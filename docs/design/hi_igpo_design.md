# Hi-IGPO 设计文档:分层信息增益 + 交替式 Planner-Executor

## 1. 目标与范围

Hi-IGPO 的目标是把 IGPO 的 turn-level 信息增益信用分配扩展到 Planner + Executor 分层检索系统。

核心问题:

- 单 Agent 在复杂问题上 rollout 长、规划和检索执行耦合。
- Multi-Agent 如果只共享最终 F1, 很难区分错误来自 Planner 拆解差, 还是 Executor 检索执行差。
- 直接联合训练 Planner 和 Executor 会引入非平稳性, 早期训练信号容易变噪。

本设计采用渐进路线:

- 先固定 Executor, 主要训练 Planner。
- Planner 稳定后再做 Planner/Executor 联合训练。
- 信用分配以外层 macro-turn 的信息增益为主, 并通过折扣累计回报传递长期收益。

## 2. 分层 Rollout

每个问题生成一条交替式分层轨迹:

```text
H_0 = question

for k = 1..K:
    Planner_k:
        input  = question + previous findings
        output = plan_k / subquestion_k

    Executor_k:
        input  = question + plan_k + optional previous findings
        output = search/visit trajectory + finding_k

    H_k = H_{k-1} + plan_k + finding_k

Final:
    Planner 或 Executor 输出 final answer
```

初始约束:

- `max_planner_turns` 控制在 3 到 6。
- Executor 每个 subtask 内允许多轮 search/visit。
- `finding_k` 应包含简洁结论和关键证据, 不只是短答案。

Executor 输入建议先用隔离式或轻量上下文式。隔离式只给 `question + plan_k`, 历史信息由 Planner 写进 plan, 信用更干净; 上下文式额外给 previous findings, 执行更强但归因更混。

## 3. 外层信息增益

对每个 macro-turn 计算一次 belief:

```text
B_0 = Bel(golden_answer | H_0)
B_k = Bel(golden_answer | H_k)
```

外层 raw IG:

```text
r_k = B_k - B_{k-1}
```

如果使用 log-prob 形式:

```text
r_k = log P(golden_answer | H_k) - log P(golden_answer | H_{k-1})
```

最终答案还有 outcome reward:

```text
r_final = F1(final_answer, golden_answer)
```

Belief 只作为奖励标量, 需要 stop-gradient。

## 4. 外层累计回报

不要把 raw IG `r_k` 直接当作最终 token advantage。应沿用 IGPO 的顺序:

```text
raw rewards
    -> group normalization
    -> turn-level discounted accumulation
    -> broadcast to owned tokens
```

设归一化后的奖励为:

```text
\hat r_1, \hat r_2, ..., \hat r_K, \hat r_final
```

从后往前计算外层累计回报:

```text
G_final = \hat r_final
G_k = \hat r_k + gamma_outer * G_{k+1}
```

直观含义:

- 当前 plan/finding 不只因本轮 belief 上升而获奖。
- 如果它为后续 turn 或最终答案铺路, 后续收益也会通过 `G_k` 回传。

当前 `compute_igpo_turn_advantage` 已实现这一逻辑: 先归一化 per-turn reward, 再从后往前累计, 最后广播到该 turn 的 token span。

## 5. Planner 信用分配

Planner 第 `k` 轮输出 `plan_k / subquestion_k`, 使用外层累计回报:

```text
Planner plan_k tokens -> G_k
```

如果 Planner 负责输出最终答案:

```text
Planner final answer tokens -> G_final
```

Planner-first 阶段冻结 Executor, Planner 使用 `G_k` 更新。这等价于把冻结 Executor 看成环境工具, Planner 是外层单 Agent。

## 6. Executor 信用分配

Executor 的处理取决于训练样本组织方式。

### 6.1 Executor 按 Macro-Turn 拆样本

如果每个 `Executor_k` 是独立 rollout:

```text
Executor_k input  = question + plan_k + context
Executor_k output = search/visit/.../finding_k
```

Executor 看不到后续 macro-turn。因此给 Executor 的不应是 raw IG `r_k`, 而应是外层累计后的:

```text
Executor_k tokens -> G_k
```

第一版最稳:

```text
Executor_k 所有 assistant tokens -> G_k
```

更细版本:

```text
finding turn-end 写 G_k
search/visit turn-end 写 0
在 Executor 内部用 gamma_inner 做 discounted accumulation
```

则内部 credit 近似为:

```text
finding tokens -> G_k
visit tokens   -> gamma_inner * G_k
search tokens  -> gamma_inner^2 * G_k
```

这样早期 search/visit 也能感知本轮 finding 对外层目标的长期贡献。

### 6.2 Executor 串成长序列

如果所有 macro-turn 的 Executor 输出被串成一条长 Executor trajectory, 则可以把 raw IG `r_k` 写在各 macro-turn boundary, 再由同一个 advantage 函数完成外层 discounted accumulation。

当前更现实的工程形态是按 `Executor_k` 拆样本, 因此主线采用 6.1。

### 6.3 局部 Executor IG

联合训练稳定后, 可给 Executor 内部增加局部 dense reward:

```text
local_r_{k,j} = Bel(finding_k | executor_history_{\le j})
               - Bel(finding_k | executor_history_{< j})
```

它用于衡量内部 search/visit 是否帮助产生本轮 finding。该局部 reward 只建议在以下条件下启用:

- 外层 `G_k` 或 raw `r_k` 为正。
- 或 judge 判定 `finding_k` 与最终问题相关。

否则可能强化无用甚至错误的检索路径。

## 7. 训练阶段

### 7.0 工具网络环境

Hi-IGPO 的 search/visit 依赖外部网页抓取。训练脚本必须显式设置工具侧代理,
不能依赖交互式 shell 自动加载 `/etc/profile.d`:

```text
http_proxy / https_proxy -> 127.0.0.1:7890
no_proxy                 -> localhost, 127.0.0.1, internal networks
```

Ray worker 会继承 launcher 的环境变量。若 launcher 未带代理, `browse_webpage`
对 Wikipedia/BBC 等站点会直连失败, 表现为 `fetch_fail=1, extract_empty=1`;
此时错误页应被丢弃, 不能作为 evidence 进入 rollout。

Search 侧使用本机 SearXNG 时, engine priority 不能只按“第一个非空结果”信任 Bing。
实测 Bing 会把 `lowest lying island nation...` 误召回到 Lowe's 商店页面。训练脚本固定采用:

```text
SEARXNG_ENGINE_PRIORITY = google,bing,duckduckgo,brave
```

注意 `research_agent.config` 会以 `override=True` 重新加载项目 `.env`; 因此 `.env` 中的
`SEARXNG_ENGINE_PRIORITY` 必须与训练脚本保持一致, 否则运行时会覆盖 launcher export。

该阶段先只做配置修复, 不在 search wrapper 内增加额外相关性过滤逻辑。
如果后续仍出现大量跑偏结果, 再单独设计 query rewrite / rerank / guard 方案。

### 7.1 Executor Cold-Start

Executor 需要先具备基本 search/visit/finding 能力。本项目主线采用:

- **单 agent IGPO 训练后的模型** (Phase 0 输出): 从 SFT 后的 `Qwen3-4B-Thinking-2507` 出发, 经单 agent IGPO RL (L1+L2) 训练得到, 冻结后作为 Executor。

备选来源 (非主线):

- Teacher rollout 生成的 multi-turn SFT。
- 其他 agentic SFT warm-start checkpoint。

目标不是最优, 而是稳定执行子任务并返回可用 finding。

### 7.1.1 Planner SFT Cold-Start

Planner-first RL 之前先做一个小规模 Planner-only SFT。该阶段的目标不是训练检索能力, 而是把 Planner 行为约束到干净的分层协议:

```text
question + previous executor findings
-> exactly one of:
   <subtask>one concrete searchable sub-question</subtask>
   <answer>short final answer</answer>
```

Planner 不调用 search/visit, 不输出 `<tool_call>`, 不把自己的搜索过程写进 `<subtask>` 或 `<answer>`。当前单 agent 训练后的模型已经强化出自主检索行为, 直接拿它当 Planner 容易复发 `Search for ...`、`Open result`、`</think>` 等单 Agent 轨迹残留。因此 Planner SFT **不能**用 `Qwen3-4B-Instruct` (非 thinking) 做 base, 必须用与 RL 一致的 thinking base (SFT 后的 `Qwen3-4B-Thinking-2507`), 再训练 Planner LoRA; Executor 仍使用冻结的单 agent 训练后模型。原因见下 §7.1.1.1。

#### 7.1.1.1 Base 一致性与空 think 蒸馏 (路径 2)

Planner SFT 的 base 必须与 RL rollout 时的 Planner base **完全一致**。当前 Phase 2b RL 配置 (`scripts/train/hi_igpo_phase2b.sh`) 是 single thinking base + dual LoRA:

```text
multi_agent.base_model = models/Qwen3-4B-Thinking-2507-SFT   # SFT 后的 Qwen3-4B-Thinking-2507, thinking-only
+algorithm.enable_think = true
```

backbone 是 `Qwen3-4B-Thinking-2507` (thinking-only), chat template 在 generation prompt 后会自动追加 think 开标签; 模型无法在权重层关闭 think。两个 base (Instruct vs Thinking) 权重不同, LoRA adapter 不通用; chat template 行为也不同 (Instruct 不自动开 think 标签)。因此早期的 `scripts/train/planner_sft_deepresearch.sh` 默认 `MODEL_PATH=Qwen3-4B-Instruct` 是错误的, 必须改回 thinking base。

但 Planner 作为决策角色 (拆子问题 / 何时停 / answer), 不需要显式推理; Executor 才是需要 think 来做 search/browse/抽取的角色。因此采用 **空 think 蒸馏**: SFT target 教 Planner LoRA 在角色层面跳过 think, 即每轮 assistant target 为 think 闭合标签 + 换行 + tag:

```text
{think_close}\n<subtask>{subtask}</subtask>
# 或
{think_close}\n<answer>{answer}</answer>
```

配合 chat template 自动开的 think 标签 (training 时 `apply_chat_template(enable_thinking=True)` 会在 assistant turn 前加 think 开标签), 组成完整的 `think...think_close subtask`。这是合法的 role-specific 行为蒸馏, 正是 dual-LoRA 的意义: 同 base, 不同 LoRA 表现不同角色 (Planner 空 think, Executor full think)。

分布一致性: SFT 与 RL 同 base、同 chat template、同 think 自动开启, 唯一区别是 SFT 教模型立刻闭合 think 然后输出 tag。Executor LoRA 仍然 full think, 不受影响。

长度预算: 空 think 后每轮 Planner response ~50 token, 4 轮 ~200 + 3×finding (~300 token/finding) ~900 + answer ~30 ≈ 1130 < `max_response_length=1536`, 留出余量。

#### 7.1.1.2 Teacher think 的处理

Teacher (MiniMax-M2.7) 是 reasoning 模型, 仍会产出长 think 块 (1500-1900 token/turn)。路径 2 下 **直接丢弃 teacher think**, 只取解析出的 subtask/answer payload 作为 SFT target 的 tag 内容。这样:

- MiniMax think 长度不再吃 SFT 轨迹预算 (SFT target think 字段为空)。
- `generate_planner_sft_rollouts.py` 的 `planner_max_tokens` 仍需保留足够预算让 MiniMax 完成 think 并吐出 tag (否则 tag 被截断 → malformed); 但这不再影响 SFT 数据长度。
- `planner_max_tokens=2048` 维持, 必要时配合 best-of-N rejection sampling 提升 tag yield。

SFT target 的精确格式 (空 think 前缀 + tag) 必须用项目实际 tokenizer 验证后再固化, 见 §7.1.1.3。

#### 7.1.1.3 Tokenizer 验证 (已完成)

用 `scripts/verify_planner_sft_target_format.py` 在 thinking base tokenizer 上验证, 结论:

1. `apply_chat_template(add_generation_prompt=True, enable_thinking=True)` 在 generation prompt 末尾自动追加 assistant think 开标签 (loss mask 0)。
2. `MultiTurnSFTDataset._message_tokens` 对每个 assistant turn 单独 slice 渲染, 使该 turn 在 slice 内为 loop.last, 模板自动包裹空 think。因此 SFT target 的 assistant content 直接写 bare tag (`<subtask>...</subtask>`) 即可, 模板自动加空 think 前缀; 显式写 think 闭合标签会被模板归一化, 三种写法等价。
3. loss mask 正确只覆盖 assistant content (闭 think + tag + im_end), system/user/finding 全 mask 0。
4. 空 think 每 turn ~13-29 loss token, 2-subtask 样本总 153 token / 42 loss token, 远低于 `max_response_length=1536`。
5. 良性副作用: `_build_sequence` 的 per-turn concat tokens 与 full-render tokens 在 think 包裹上不一致 (per-turn 每 turn 包, full-render 只最后 turn 包), 会触发 `Token mismatch in MultiTurnSFTDataset` warning。训练使用 per-turn tokens (与 RL `assemble_planner_sequence` 一致), warning 为 false alarm, 暂不处理。

基于此, `build_planner_sft_from_rollouts.py` 现有 message 构造无需修改; `planner_sft_deepresearch.sh` 的 `MODEL_PATH` 已从 `Qwen3-4B-Instruct` 改回 SFT 后的 `Qwen3-4B-Thinking-2507`。

第一版先只做 L3 500 条:

```text
DeepResearch-9K L3: 500
```

原因是 DeepResearch-9K 的 L2 `final answer` 中存在较多描述型答案, 例如正确实体是
`Paul "Bear" Bryant`, 但 golden answer 写成一段属性描述。这类样本会让 token F1/EM
过滤与后续 reward 失真。L3 的 short golden answer 更干净, 且更贴近 Planner 需要解决的
复杂 query rewrite 和长依赖链问题。L2 后续只在做 canonical answer 清洗后再混入。

当前实现入口:

```text
scripts/prepare_planner_sft_samples.py
    从 data/deepresearch_phase2.parquet 固定采样 L3 500,
    输出 data/planner_sft_seed_l3_500.parquet。

scripts/build_planner_sft_from_rollouts.py
    将 teacher / frozen-executor 产生的 interleaved rollout dump 清洗成
    data/planner_sft/train.parquet 与 data/planner_sft/val.parquet。

scripts/generate_planner_sft_rollouts.py
    使用 MiniMax 作为 teacher Planner / finding summarizer, 但 evidence 仍走
    本项目 web_search + browse_webpage, 输出 raw rollout JSON 供人工审计和清洗。

scripts/train/planner_sft_deepresearch.sh
    从 SFT 后的 Qwen3-4B-Thinking-2507 训练 Planner LoRA, 使用 data.multiturn.enable=true。
    空 think 蒸馏: SFT target 教 Planner 立刻闭 think 并输出 tag (路径 2)。
```

注意不要复用 `scripts/train/igpo_single_sft.sh` 做 Planner SFT。该脚本的语义是
`SFT base -> single-agent IGPO/RL` (单 agent 训练入口), 不是监督微调入口。

数据生成流程:

```text
for each selected sample:
    input: question, golden_answer
    1. MiniMax teacher Planner 基于 question + previous findings 生成 <subtask>;
    2. 用真实 web_search(subtask) + browse_webpage(urls, goal=subtask) 获取 evidence;
    3. MiniMax 仅把真实 search/browse 输出压缩成 executor finding;
    4. 重复 1-3, 直到 teacher Planner 输出 <answer> 或达到 max_planner_turns;
    5. 用 golden_answer 做离线 F1/EM 过滤, 不把 golden_answer 暴露给 Planner 生成过程;
    6. 轨迹形式为多轮 messages (路径 2 空 think 蒸馏, assistant target = think 闭合 + tag, teacher think 丢弃):
       system: interleaved planner prompt
       user: original question
       assistant: {think_close}\n<subtask>...</subtask>
       user: executor finding
       assistant: {think_close}\n<subtask>...</subtask> or {think_close}\n<answer>...</answer>
       ...
       assistant: {think_close}\n<answer>golden-equivalent short answer</answer>
    7. 通过规则与 answer F1/EM 过滤后写入 SFT 数据。
```

Teacher 只能生成 Planner 轨迹和 evidence compression, 不能让 Planner 自己模拟工具调用。`finding` 必须来自真实检索结果、冻结 Executor 运行结果或可追溯 evidence 摘要, 不使用纯编造 observation。

硬过滤规则:

```text
reject if any planner assistant turn:
    - does not contain exactly one complete <subtask>...</subtask> or <answer>...</answer>
    - contains <tool_call>, <tool_response>, <think>, </think>
    - contains "Search for", "Open result", raw search/browse instructions
    - puts numbered multi-step plans inside <subtask>
    - has empty <answer> or answer too long

reject if any executor finding:
    - is empty
    - contains <think>, </think>, <answer>, <tool_call>, <tool_response>
    - contains raw search process such as "Search for", "Open result"
    - is mostly failure text or unrelated search snippets
```

答案过滤:

```text
normalized EM == 1
or semantic/token F1 >= 0.8
```

若 teacher 输出解释句, 先压缩成短答案再重新打分; 仍不达标则丢弃。Planner SFT 的 loss mask 只覆盖 assistant 的 `<subtask>` / `<answer>` token; system、user、executor finding 全部 mask 为 0。

SFT 后先做离线健康检查, 不立刻进入 RL:

```text
malformed_subtask_rate < 2%
malformed_answer_rate  < 2%
answer_format_pass_rate > 95%
empty_finding_rate      < 2%
avg_planner_turns       在 2 到 4 内
final_answer_f1         明显高于未 SFT Planner rollout
```

只有这些指标通过后, Planner-first RL 才使用该 Planner LoRA 初始化。

### 7.2 Planner-First

冻结 Executor, 只训练 Planner。

Planner 初始化应优先使用 7.1.1 的 Planner SFT LoRA。若直接使用单 agent 训练后模型作为 Planner, 必须先通过 rollout 健康检查; 一旦出现大量 malformed subtask / malformed answer, 应停止 RL, 回到 Planner SFT 或 parser/finding 清洗。

奖励:

```text
raw outer IG r_k + final F1
-> normalization
-> outer discounted return G_k
```

更新:

```text
plan_k tokens -> G_k
final answer tokens -> G_final
```

这是第一阶段主线, 因为复杂 L3 问题的主要瓶颈通常是:

- 搜什么。
- 如何拆子问题。
- 何时停止。
- 如何利用已有 findings 改写下一步目标。

### 7.2.1 Phase 2b 正式小跑

当前正式小跑以 `scripts/train/hi_igpo_phase2b.sh` 为入口, 目标是验证 Planner-first 训练 (冻结的单 agent 训练后模型作 Executor) 能稳定产生:

- 正常的交替式 Planner/Executor rollout。
- 非空且可解释的 belief / IG。
- 合理的 turn-level advantage。
- 可恢复的中间 checkpoint。

当前 full run 配置:

```text
data.train_batch_size = 16
actor_rollout_ref.actor.ppo_mini_batch_size = 16
agent_grpo.n = 8
multi_agent.max_planner_turns = 4
multi_agent.agents.executor.max_turns = 6
trainer.total_training_steps = 100
trainer.save_freq = 5
trainer.resume_mode = auto
trainer.remove_previous_ckpt_in_save = true
multi_agent.freeze_executor = true
CUDA_VISIBLE_DEVICES = 1,3,4,6
```

训练侧序列长度配置:

```text
data.max_response_length = 1536
max_seq_len_for_training = 7168
actor_rollout_ref.rollout.max_model_len = 9216
actor_rollout_ref.rollout.max_num_batched_tokens = 9216
actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu = 7168
actor_rollout_ref.actor.ppo_max_token_len_per_gpu = 7168
multi_agent.planner_findings_max_chars = 1000
```

这些参数分别约束 rollout 生成侧 vLLM 最大上下文、rollout 完成后用于训练的拼接序列长度、rollout log-prob 重算长度和 actor PPO 更新长度。由于 Planner/Executor 多轮交互会不断累积 history, 生成下一轮时 vLLM prompt 也会增长; 同时 Planner 训练序列会把多轮 `plan/subtask` 与 Executor `finding` 交替拼回一个长响应帧。之前正式小跑先后暴露过三类长度/显存问题: `compute_log_prob` 阶段实际 `max_seq_len=7696` 超过 `log_prob_max_token_len_per_gpu=7168`, step 4 生成阶段 `decoder prompt length=6357` 超过 `rollout.max_model_len=6144`, 以及 actor update 阶段在 `loss.backward()` 处 OOM。第三类 OOM 发生时 step 2 最长 Planner response turn-end 已接近 `8930` token, 单样本 backward 峰值过高。因此当前保留 vLLM 生成窗口 `9216`, 但把训练侧序列和 actor/log-prob token 上限降到 `7168`, 同时降低单轮生成长度和注入 Planner 的 finding 长度, 优先保证可持续训练。

Planner 训练序列当前拼接方式:

```text
[planner_subtask_0][executor_finding_0][planner_subtask_1][executor_finding_1]...[planner_answer]
```

`executor_finding` 作为 observation 注入, loss mask 为 0; Planner 输出 token 的 loss mask 为 1。当前实现没有额外插入新的特殊分隔 token, 边界主要依赖 Planner/Executor 文本自身的标签、换行和 turn-end 位置记录。

有效 rollout 数约为:

```text
effective_rollouts_per_step = train_batch_size * agent_grpo.n = 128
```

本轮 `train_batch_size=16` 和 `agent_grpo.n=8`, effective batch 128。之前曾用 `train_batch_size=4` (effective 32), 已上调到 16 以提升梯度估计质量; 显存上因 `use_dynamic_bsz=true` + `ppo_max_token_len_per_gpu=7168` token 预算封顶, 不会 OOM (设计文档此前判断瓶颈在 search/browse/summary LLM 工具调用, 不是 GPU 显存), 主要代价是每步 tool 调用变多、step 时间变长。`max_planner_turns=4` 用于允许更像正式任务的多轮规划; 最后一轮会 force answer, 因此实际通常是 2 到 3 个子任务机会加最终回答。`executor.max_turns=6` 用于给冻结 Executor 多一轮检索/访问空间。

checkpoint 策略:

```text
只保留最新 global_step_* checkpoint
```

`trainer.remove_previous_ckpt_in_save=true` 会让 worker 删除前一次保存的 actor shard; driver 侧额外清理当前实验目录下除最新 `global_step_N` 以外的旧 `global_step_*` 目录, 避免旧 `data.pt` 残留。`latest_checkpointed_iteration.txt` 始终指向最新 checkpoint, `resume_mode=auto` 从该 checkpoint 恢复。

若出现 OOM、搜索服务压力过大或 step time 明显不可接受, 首先回退到:

```text
data.train_batch_size = 8        # 从 16 降回 8 (effective 64), 仍大于原 4
multi_agent.max_planner_turns = 3
multi_agent.agents.executor.max_turns = 5
agent_grpo.n = 4
```

若仍 OOM, 再降 `train_batch_size = 4` (effective 32)。

运行前必须检查:

- 远程 `models/Qwen3-4B-Thinking-2507-SFT` (单 agent 训练后 ckpt) 存在。
- SearXNG / search 服务可用。
- `CUDA_VISIBLE_DEVICES` 对应 GPU 空闲。
- `.env` 中 judge / search 相关配置有效。

训练中每 30s 监控:

- 主日志: `deepresearcher_qwen3_4b_hi_igpo_phase2b.log`。
- Rollout dump: `outputs/deepresearcher/qwen3_4b_hi_igpo_phase2b/rollout/planner_rollout_step_*.json`。
- Advantage audit: `outputs/deepresearcher/qwen3_4b_hi_igpo_phase2b/advantage_audit_step_*.json`。
- Checkpoint: `ckpts/deepresearcher/qwen3_4b_hi_igpo_phase2b/global_step_*/actor`, 预期只保留一个最新 `global_step_*`。

F1 日志拆分:

```text
planner/f1/mean                = planner/f1_format/mean, 保持向后兼容
planner/f1_format/mean         = 带格式惩罚的 F1, 格式坏时为 -2
planner/f1_semantic/mean       = 只看最终 answer 文本与 GT 的 token F1, 范围 [0, 1]
planner/format_error_rate      = format-penalized F1 < 0 的样本比例
```

训练 reward 仍使用 `planner/f1_format`。`planner/f1_semantic` 只用于诊断, 用来区分“答案内容错”和“答案内容可能相关但标签格式坏”。

### 7.2.2 Visit/Browse 兼容策略

search/visit 工具语义:

```text
search(query) -> 返回 URL 文本
visit(url, goal) -> 直接访问给定 URL, 使用 Jina Reader 抓取正文, 再按 goal 摘要
```

模型策略上应主要从上一轮 `search` 的结果里选择 URL 调用 `visit`; 但 `visit` 工具本身不校验 URL 是否来自上一轮 `search`, 也不依赖 search cache。当前项目的 `browse_webpage` 复用 `web_search` 产生的 `WebPageInfo`/browser 对象, 因此历史实现会在 URL 未精确命中最近 search 结果时直接返回 `[]`, 与 search/visit 原生语义不完全一致。

Phase 2b 采用兼容实现:

```text
for url in url_list:
    1. 优先复用当前样本最近 search 结果中的 WebPageInfo/browser;
    2. 如果未命中 search 结果, fallback 到直接抓取该 URL;
    3. 返回格式仍保持 [{"url": ..., "information": [...]}], 不改变 rollout 消费方。
```

日志需要区分:

- `cache_hit`: URL 命中 search 结果并复用已有上下文。
- `direct_fetch`: URL 未命中 search 结果, 走直接抓取 fallback。
- `fetch_fail`: 直接抓取失败。
- `extract_empty`: 抓取成功但 ReadingAgent 没抽出有效信息。

这样既保留“search 后 browse”的主路径, 又避免模型生成合法 URL 但因实现过严而得到空工具响应。

抓取失败判断不能只看 browser 对象是否存在。若页面正文实际是 `## Error`、`HTTPSConnectionPool`、`ConnectTimeoutError`、`Max retries exceeded` 等网络错误文本, 应直接标记为 `browser="error"` 并计入 `fetch_fail`, 不再送入 visit extractor。

`visit(url, goal)` 的 goal-directed extraction 必须与普通 `browse_webpage(url_list)` 区分处理。普通 browse 可继续使用本项目分页式 `EXTRACT_NEW_INFO_PROMPT` 和 `<extracted_info>` 解析; 但原生 visit 是“整页内容 + 用户 goal -> JSON evidence/summary”, 不应复用分页 prompt 直接把 `goal` 塞成 `sub_question`。否则 thinking 模型容易格式漂移, 产生空 `extracted_info` 或无效片段, 污染 finding 与后续 IG/advantage。

Phase 2b 的 `goal` 路径修复为:

```text
if goal:
    1. 抓取 URL 正文, 转 markdown, 截断到 WEBCONTENT_MAXLENGTH;
    2. 使用 goal-directed EXTRACTOR_PROMPT(webpage_content, goal);
    3. 要求 LLM 输出 JSON: {"rational": ..., "evidence": ..., "summary": ...};
    4. 从 raw response 中抽取 JSON 对象, 支持 ```json fence 和 thinking 前后缀;
    5. JSON 解析失败或 evidence/summary 过短时, 缩短正文后重试;
    6. 成功后写入 page_summary:
       The useful information in {url} for user goal {goal} as follows:
       Evidence in page: ...
       Summary: ...
```

失败处理只返回明确的工具失败摘要, 不把原文截断伪装成有效 finding。这样可以让训练监控据 `extract_empty` / failure 文案识别工具质量问题, 避免“非空但不可信”的奖励污染。

验收标准:

- `browse_webpage(url_list=["https://example.com"])` 仍能正常返回非空摘要。
- `browse_webpage(url_list=["https://example.com"], goal=...)` 在 30 秒级别内返回与 goal 相关的 `Evidence` 和 `Summary`, 不出现模板残片如 `Should contain the new information`。
- Wikipedia 等真实页面如果抓取正文为空或抽取失败, 日志应明确标记失败, 不静默产出空 finding。

### 7.3 Joint V1

Planner-first 稳定后, 开始联合训练。

建议先做交替更新, 降低非平稳性:

```text
rollout 一批完整 Planner+Executor 轨迹
计算 outer IG 与 G_k
固定 Executor, 更新 Planner
固定 Planner, 更新 Executor
```

或每 N step 只更新一个 agent。

奖励分配:

```text
Planner plan_k tokens -> G_k
Executor_k tokens     -> G_k
```

第一版不拆 Planner/Executor 对 raw IG 的相对贡献, 先验证 shared outer return 是否能提升。

### 7.4 Joint V2

在 Joint V1 稳定后, 再增加更细分的 credit:

- Executor 内部 local IG。
- Planner/Executor 的反事实贡献拆分。
- 不同 agent 的 reward 权重或可学习 credit split。

这些属于增强项, 不应阻塞第一版训练。

## 8. Advantage 计算约定

统一约定:

- 写入 reward tensor 的是 raw reward 或外层已累计的 `G_k`, 具体取决于样本组织。
- 如果样本本身包含完整外层时间轴, 写 raw `r_k`, 由 advantage 函数累计。
- 如果样本是单个 `Executor_k`, 写外层累计后的 `G_k`, 因为该样本看不到后续 macro-turn。

Planner:

```text
完整 Planner trajectory 内含所有 macro-turn
-> 写 raw r_k 和 final F1
-> compute_igpo_turn_advantage 得到 G_k
```

Executor:

```text
每个 Executor_k 单独训练
-> 写外层 G_k
-> 可直接 broadcast, 或在内部 boundary 上做 inner accumulation
```

不要把同一层的 discounted return 重复累计两次。

## 9. 归一化策略

沿用 IGPO 的关键设计:

- IG reward 和 F1 reward 分开归一化。
- 同一 prompt 的 rollout group 内做 group-relative advantage。

默认:

```text
info_gain_norm_mode = separate
gamma_outer = 0.95 或 1.0
```

可选改进:

- `global`: 同一 prompt 下所有 turn 的 IG 混合归一化。
- `turn_group`: 按 `(prompt, turn-index)` 归一化, 缓解不同深度 turn 的尺度不可比。

第一版先用 `global` 对齐 IGPO, 再用 `turn_group` 做消融。

## 10. 判断 Planner-First 是否足够

不要只看最终 F1。建议同时看:

- L3 validation F1/EM 是否 plateau。
- 平均 macro-turn 数是否下降或稳定。
- invalid Planner output rate 是否低。
- repeated subquestion rate 是否低。
- empty/useless finding rate 是否下降。
- 每轮平均 IG 是否还有提升。
- answer-before-evidence 比例是否下降。
- L1/L2 是否明显退化。

一个实用停止标准:

```text
连续 N 次 eval, L3 F1 无明显提升;
invalid/repeat rate 稳定在低位;
平均 IG per macro-turn 不再上升;
L1/L2 不明显退化。
```

满足后进入 Joint V1。

## 11. 风险与回退

主要风险:

- Executor cold-start 不足, Planner reward 被 Executor 噪声污染。
- Joint training 非平稳, 两个 agent 同时变化导致 credit assignment 变噪。
- Executor 按 macro-turn 拆样本时, 误把 raw `r_k` 当训练信号, 导致它感知不到后续收益。
- local IG 过早启用, 强化无用 finding 的内部检索路径。

回退策略:

- 先训 Planner, Executor 冻结。
- Joint 阶段使用交替更新。
- Executor 第一版直接吃外层 `G_k`, 不上 local IG。
- 若 multi-agent 不稳定, 回退到单 Agent IGPO 或 Planner-first 作为主要结果。
