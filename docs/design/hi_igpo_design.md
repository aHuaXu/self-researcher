# Hi-IGPO 设计文档:分层信息增益 + 交替式 Planner-Executor

> 日期:2026-06-19
> 状态:设计草案(待 review → writing-plans)

## 1. 背景与动机

本项目当前是 DeepResearcher 的改版:Planner + Executor 双 agent(共享 Qwen3-4B base、各自 LoRA),
Executor 走一次性 DAG(子任务带依赖、分 wave 并行),用 **GRPO(无 critic)** 训练。

**核心痛点(简历/论文缺乏算法新意):**

1. **奖励稀疏**:outcome 信号(F1)只在最终答案出现一次。
2. **信用混淆**:当前 `MultiAgentRewardManager` 让两个 agent **共享同一个最终 F1**:
   - `reward_planner  = α·rule_p + (1-α)·F1`
   - `reward_executor = β·rule_e + (1-β)·F1`
   - `rule_p/rule_e` 只是格式/轮数约束。答错时无法区分是 Planner 拆解差还是 Executor 执行差。

**机会**:IGPO(arXiv:2510.14967,Information Gain-based Policy Optimization)用"信息增益"给单 agent 多轮
提供稠密内生奖励,且 IGPO 与本项目**同源**(都基于 verl + Search-R1 + DeepResearcher)。IGPO 原文明确把
"扩展到多智能体分层规划检索系统"列为开放问题。

## 2. 目标与贡献

**一句话**:把 IGPO 的 turn-level 信息增益从单 agent 推广到**交替式分层多 agent**——
Planner 每提出一个子任务是一个决策步、Executor 每次 tool call 是一个决策步,**两层都用信息增益做信用分配**,
解决多 agent deep research 的"稀疏奖励 + 信用混淆"。命名:**Hi-IGPO(Hierarchical IGPO)**。

**两个算法贡献:**
1. **交替式分层 IGPO**:把 turn-level IG 用到交替式 Planner-Executor;冻结 Executor 时 Planner 退化为
   标准单 agent IGPO(见 §5.0),命中 IGPO 原文留的"多智能体分层规划检索"开放问题。
2. **变长 Planner 的 turn-group 归一化**:把 A²TGPO 的 (prompt, turn-index) 归一化适配到轮数可变的
   Planner(含 <2 样本回退),解决原生 IGPO 全局归一化下"不同深度 turn 不可比"的偏置(见 §5.2)。

**范围约束(本设计 story 优先,不追 SOTA):**
- 算力上限:**4×V100 32GB**。小 batch、`grpo_n` 4~8、`max_planner_turns ≤ 5`。
- 成功标准:一个站得住的新机制 + 一组干净消融,证明优于"共享 F1"基线;小规模能讲清楚机制即可。

## 3. 架构:交替式 Planner ↔ Executor

将"一次性出 DAG"改为**交替循环**,使 Planner 本身成为外层多轮 agent:

```
P_0 ← belief(question)                      # 还没任何子任务时的信念
for t = 1 .. T_p (T_p ≤ max_planner_turns):
    planner: 输入 question + 历史 findings_{<t}
             → 输出 [子任务_t]  或  [<answer>]
    if 输出是 <answer>:
        break
    executor: 多轮 tool call 执行 子任务_t  → findings_t      # 复用现有 run_llm_loop
    P_t ← belief(question, findings_{≤t})    # 一次 teacher-forcing forward
最终 answer → F1(answer, golden)
```

- **Planner**:外层多轮 agent(新增的外层循环)。
- **Executor**:内层多轮 agent(复用现有 `LLMGenerationManager.run_llm_loop`)。
- **限制**:`T_p ≤ 5` 控成本;Executor 每子任务沿用现有 `max_turns`。
- 相比 DAG:失去并行、轨迹变长(串行),但换来 Planner 的自适应决策(适时停止/换方向)与原生的分层信用。
- **最终答案归 Planner**:Executor 只产中间 `findings_t`;Planner 攒够信息后自己输出 `<answer>`(F1 对它算)。
- **`findings_t` 的内容(定:两者都要)**:Executor 把 `子任务_t` 当成小问题跑搜索循环,返回
  **(a) 它对该子任务的 `<answer>`(简洁结论)+ (b) 沿途检索到的关键证据**。中间过程信息更全,
  避免只取简短 `<answer>` 丢信息;`findings_t` 灌回 Planner 上下文(`response_mask=0`,不训),供下一轮决策与 belief。
- **Executor 输入(定:隔离式 A)**:Executor 执行 `子任务_t` 时**只看 `子任务_t`(+ 原始 question 做锚定),不看历史 findings**。
  理由:① 契合分解(Planner 持全局、把子任务写自包含;Executor 专注执行一个);② IG 信用更干净——`findings_t`
  是"这个子任务带来的新信息",避免 Executor 重捞历史信息污染 IG_t 归因;③ 实现简单(Executor 无状态)。
  需要前置结论时由 Planner 把它写进子任务描述(这是 Planner 要学的、IG 会奖励的能力)。
- **架构(定:保留双 rollout,不复用单 agent 管线)**:Planner 与 Executor **各产独立 DataProto**(`planner_outputs` /
  `executor_outputs`)。**不**把 Executor 降级成 `execute_predictions` 里的黑盒工具(那样会丢掉 Executor 的可训轨迹、
  foreclose Phase 2a)。Phase 2b 只训 Planner、Executor 冻结,但**其 rollout 照样捕获保留**,Phase 2a 将来可直接在其上
  加 IG/优势,不用改架构。Planner 的 IG 仍复用 `compute_all_turns_vectorized`(belief),但两条 rollout 分开管理。

## 4. 信念与信息增益(双层共享一条 belief 轨迹)

沿用 IGPO 定义。golden 答案 `a = (a_1..a_L)` 已有(F1 即用它)。第 t 个子任务执行后的信念:

```
P_t = exp( (1/L) · Σ_j log π_θ(a_j | q, findings_{≤t}, a_{<j}) )
```

- 几何平均消除答案长度影响;
- **stop-gradient**:P_t 只作奖励标量,不让"提高答案概率"路径直接回传梯度;
- 子任务 t 的信息增益:`IG_t = P_t − P_{t-1}`;
- 一条轨迹算 `T_p + 1` 次 belief forward;golden 短,4 卡可负担。

## 5. 信用分配(优势计算)

### 5.0 关键简化:冻结 Executor ⇒ Phase 2b 退化为单 agent IGPO

主线(Phase 2b)**冻结 Executor**。此时 Planner 单独看就是一个标准的**单 agent 多轮 IGPO**:

| IGPO 单 agent 概念 | 本项目 Phase 2b 对应 |
|---|---|
| 一个 turn | Planner 提出一个子任务 `p_t` + 冻结 Executor 执行它 |
| turn 的 observation / tool_response | 冻结 Executor 的输出 `o_t`(属于环境动态) |
| turn-level IG 奖励 | `IG_t = Bel_t − Bel_{t-1}` |
| 末轮 outcome | Planner 输出 `<answer>` → F1 |

**推论:Phase 2b 的优势计算可直接复用 IGPO 的 `compute_grpo_outcome_advantage`,不需要任何自定义"多 agent 优势"数学。**
之前提的 AT-GRPO 角色分组只在未来"联合训 Planner+Executor"(future work)时才需要,2b 用不上。

### 5.1 优势计算的固定骨架(先归一化、再折扣累加)

三者共用同一骨架,差异只在"归一化的分组维度":
1. **per-turn 奖励向量**:turn t 的奖励 = `IG_t`(中间步)或 F1(末轮),写到该 turn 末 token 位置。
2. **先归一化**:对 per-turn 奖励做 `(r − mean)/std`(分组维度见 5.2)。
3. **IG 与 F1 分开归一化**(`info_gain_norm_mode=separate`):不同 mask 分别统计 IG 部分与 F1 部分,
   **避免终奖尺度淹没 IG 信号**(IGPO 关键设计)。
4. **再折扣累加**(turn 级),advantage 广播到该 turn 全部 planner token:
   ```
   Ã_{i,t} = Σ_{k≥t} γ^(k-t) · A_{i,k}        (默认 γ=1.0)
   ```

> 对应 IGPO `compute_grpo_outcome_advantage`(含 `gamma`、`info_gain_norm_mode`)+
> `_compute_turn_level_advantage`。注意:三者都是"**先归一化 per-turn 奖励、再累加**";
> "先把 raw reward 累加成 `G_t` 再减基线"既非 IGPO 也非 A²TGPO,因终奖会淹没 IG,**不采用**。

### 5.2 归一化分组维度:`global`(基线) vs `turn_group`(改进,主线)

这是本设计的**第二个算法贡献点**,也是一条干净消融轴。已核对三种方法的实际做法:

| 模式 | 分组维度 | 出处 | 评价 |
|---|---|---|---|
| `global` | 按 **prompt**,所有 turn 混池(同一组 mean/std) | **原生 IGPO**(已查代码:按 `index` 分组) | 不同深度 turn 不可比(早期 IG 大、后期小)→ 系统偏置 |
| `turn_group` | 按 **(prompt, turn-index)**,每个 turn 只和同深度同伴比 | **A²TGPO**(arXiv:2605.06200) | 解决跨时序不可比 + "variable trajectory depth" 漂移 ✅ |
| anchor-state | 按跨轨迹重复出现的相同 state 分组 | GiGPO(arXiv:2505.10978) | research 中 state(question+findings)几乎不重复 → **不适用** |

- **主线取 `turn_group`**(A²TGPO):每个子任务深度独立基线,契合变长 Planner。
- **实现注意(变长)**:深 turn 处该 (prompt, turn-index) 组样本可能 <2 → **回退到 `global` 的 mean/std**,
  避免单样本 std 退化(A²TGPO 处理 variable depth 的工程要点)。
- **(可选)自适应 turn 级 clipping**(A²TGPO):按归一化 IG 调 clip 区间,informative turn 放宽、
  uninformative 收窄;作为第二档消融。
- COMA 单步反事实(每步 M 次替换推演)算力高,归 future work(§10)。

### 5.3 Executor(内层)
- 主线(Phase 2b)**冻结**,不更新 LoRA,仅复用其执行产生 observation 供 belief 计算。
- "Executor 内层 turn-level IG 训练 + 联合优化"随 Phase 2a 标记为 **future work**(§10)。

> **⚠ 待讨论(核心流程跑通后再回看):Phase-1 模型当 Executor 的角色不匹配。**
> Phase-1 训出的是"对**原问题**端到端出答案"的研究 agent,**没专门训"执行一个子任务并返回有用 findings"这个角色**;
> 略过 Phase 2a 是赌它"够用"(已会多轮检索)。潜在风险:① QA 模型给子任务的 `<answer>` 可能太短/丢信息
> →靠 `findings_t` 取(a)+(b) 两者缓解(§3);② 子任务文本与训练时的"原问题"分布有偏移,Executor 表现可能打折。
> **回填选项(若 findings 质量太差导致 IG 信号弱)**:(i) 轻量做 Phase 2a 给 Executor 内层加 IG 训练;
> 或 (ii) 不训练、只给 Executor 换一套"执行子任务"的 prompt,让产出更适合当 findings。**先把核心交替式流程跑通,再据 IG 信号强弱决定是否回填。**

## 6. 训练流程(两个有效阶段)

| 阶段 | 训练对象 | 奖励 | 状态 |
|---|---|---|---|
| **Phase 1:单 agent IGPO** | 单 agent | turn-level IG + F1(移植 IGPO) | 升级现有单 agent GRPO;= 基线 + 冻结 Executor warm start |
| ~~Phase 2a:Executor IG~~ | — | — | **跳过(future work)** |
| **Phase 2b:Planner 交替式 IGPO** | **Planner LoRA**(冻结 Executor = Phase-1 模型) | 外层 turn-level IG + F1 | 主线贡献 |

- 跳过 2a 的理由:Phase-1 单 agent 模型已具备多轮搜索能力,直接冻结当 Executor;减少阶段数、降风险、省算力。
- 冻结 Executor 的理由:规避 MARL 非平稳性——Planner 的 IG 信用基于 Executor 跑出的 belief 轨迹,
  Executor 不动则信用干净。

## 7. 实现策略:从 IGPO 移植 IG 核心(可执行步骤)

IGPO 与本项目同源(verl + Search-R1 + DeepResearcher),文件结构几乎一一对应。
**不整库替换,按文件移植**(优势/belief 整段复制;generation 整文件 port 成并行路径,见下)。已核对的关键差异如下。

### 7.0 已核对的源/目标差异

**belief 计算(IGPO 新增,本仓库没有 → 整文件复制):**
- `scrl/llm_agent/vectorized_gt_logprob.py`,核心:
  ```text
  compute_all_turns_vectorized(self, model, original_input_ids, original_attention_mask,
                               original_position_ids, ground_truth_text,
                               turn_end_positions, temperature=1.0)
      -> (gt_log_probs_per_turn, gt_answer_ranges)
  ```
  - 单次 forward:把 golden 答案的 T 份拷贝拼到序列尾,用 **4D causal mask** 防跨 turn 串扰;
    `cur_value = exp(answer_log_probs.mean())` 即 `Bel_t`。
  - 全程 `torch.no_grad()` → belief 天然是常量(= IGPO 的 stop-gradient)。
  - 自包含,依赖少,可整文件拷入本仓库 `scrl/llm_agent/`,改 import 即可。
- `scrl/llm_agent/prealigned_vectorized.py`:把 per-turn IG 奖励对齐到 token 位置(配合
  `turn_end_positions`)→ 一并复制。

**优势函数(本仓库有同名,但是 vanilla 版 → 替换/扩展):**

| | 本仓库现状 `core_algos.py:111` | IGPO 版(目标) |
|---|---|---|
| 签名 | `compute_grpo_outcome_advantage(token_level_rewards, eos_mask, index, epsilon)` | `(... , norm_adv_by_std_in_grpo=True, gamma=1.0, info_gain_norm_mode='joint', curriculum_f1_weight, curriculum_ig_weight)` |
| 逻辑 | `scores = rewards.sum(-1)` → 单标量广播到所有 token(**无 turn / 无 IG / 无 gamma**) | per-turn 奖励**分组归一化**(IG/F1 separate)→ `_compute_turn_level_advantage()` 折扣累加广播 |

→ 动作:把 IGPO 版 `compute_grpo_outcome_advantage` 与 `_compute_turn_level_advantage`
移入本仓库 `core_algos.py`(保留旧函数或加 `adv_estimator` 分支),适配 `eos_mask`↔`response_mask` 命名。

> **优势函数的输入表示(本次确认):保持 IGPO 原生 token 级签名,不引入 per-turn dict 适配层。**
> 即 `(token_level_rewards:(bs,L), response_mask:(bs,L), index)`——IG 写在每个 turn 末 token、F1 写在
> 最后一个有效 token,turn 边界从张量非零位置 / `turn_boundary_mask` 反推。这样**单 agent 与 Phase 2b
> 的 Planner 共用同一个 token 级优势函数,逐函数等价于 IGPO**(§5.0)。本设计的 `turn_group`(§5.2)只作为
> 该函数内部的一个**分组模式扩展**:turn-index 由每行 turn 边界的出现顺序现场推出(第 k 个 IG 边界 = turn k),
> 同样不需要外部 per-turn 结构。

**生成逻辑(本仓库与 IGPO 已分叉 → 整文件 port 成并行路径,不动现有文件):**

> **设计原则(本次确认):单 agent 流程必须与 IGPO 项目逐函数一致。**
> 因此生成路径**不 cherry-pick**——已核对两边 `generation.py` 结构分叉显著:
>
> | | 本仓库 `generation.py` | IGPO `generation.py` |
> |---|---|---|
> | 行数 | 590 | 1006(多 ~400 行 IG/pseudo-forward/vectorized 收集) |
> | `run_llm_loop` 签名 | `(gen_batch, global_steps, lora_adapter_name) -> (Dict, Dict)` | `(gen_batch, global_steps, ground_truths) -> (str_list, tensor, info_gain_rewards)` |
> | IG 钩子 | 无 | 每轮 `pseudo_generate_sequences` + belief + `info_gain_rewards` |
>
> 手工拼接 ~400 行还要调和签名/返回,既不小也极易与 IGPO 产生细微偏差,违背"流程一致"前提。
>
> **取而代之:把 IGPO 的 `generation.py` 整文件搬为 `scrl/llm_agent/igpo_generation.py`(只改 import / 对齐本仓库 verl 的 DataProto·tokenizer 接口,不改其数学与控制流),做成一条 `adv_estimator=igpo` 时才走的并行路径。现有 `scrl/llm_agent/generation.py` 零改动(仍作冻结 Executor 的内层 rollout)。`multi_agent_generation.py`(一次性 DAG)**已废弃删除**——multi-agent 路径改由交替式 `interleaved_generation.py` 承载。**
>
> - 一致性:单 agent 跑的就是 IGPO 原代码三件套(`igpo_generation.py` + `vectorized_gt_logprob.py` + `compute_igpo_turn_advantage`),可逐函数核对。
> - 改动面:几乎全是新增文件;现有代码只新增一个"按 `adv_estimator` 选 generation manager"的选择器开关。基线路径回归不破。
> - 代价:IGPO `generation.py` 依赖本仓库 verl 版本接口,搬来需对齐少量 import / API 漂移(同源,漂移小)。
>
> **工具层必须替换(已核对):IGPO 只有 `web_search`(走独立 `tools_server`/`MessageClient`),本项目是 `web_search` + `browse_webpage`(进程内直调 `research_agent.tools`)。**
> 我们要的是 IGPO 的 **IG/belief/rollout 算法**,不是它的工具环境(工具属任务/数据,本项目是联网 research)。
> 好在两边工具调用**格式一致**(都是 `<tool_call>{name,arguments}</tool_call>` + `<answer>`),且 `parse_response`
> 返回 `[(is_stop, reasoning, answer/tool_call)]`、`execute_predictions(tool_call_list, total_number)` 返回
> `{idx,question,think,tool_call,content}` —— **签名与返回结构两边相同**。故 port 时**只替换 `execute_predictions`
> 与 `parse_response` 两个方法为本项目版本**(保留 browse_webpage、去掉 `self.client`/tools_server 依赖),
> 其余 IG/belief/控制流原样保留。替换隔离在两个 drop-in 方法里,不动算法。

**config 开关(来自 IGPO `train.sh`):**
`+algorithm.info_gain_type=log_prob_diff`、`+algorithm.info_gain_norm_mode=separate`、`algorithm.gamma=1.0`。

> 本地 verl `0.2.0.dev`;IGPO 精确版本未取到,以**文件级 diff 为准,注意 API 漂移**(如 `eos_mask` 命名)。

### 7.1 实现顺序(每步可独立验证)

1. **clone 参考库**:`git clone https://github.com/GuoqingWang1/IGPO /tmp/IGPO_ref`(仅作 diff 参考,不进项目)。
2. **移植 belief**:复制 `vectorized_gt_logprob.py` + `prealigned_vectorized.py` 到 `scrl/llm_agent/`,
   改 import 跑通。单测:给定固定 context + golden,`Bel_t ∈ (0,1)`、随检索增多上升。
3. **移植优势**:把 IGPO 的 `compute_grpo_outcome_advantage` + `_compute_turn_level_advantage` 并入
   `core_algos.py`,适配命名;在 `ray_trainer.py:compute_advantage` 增加 IGPO 分支,
   传 per-turn 奖励张量(非 sum)+ `gamma` + `info_gain_norm_mode`。
4. **整文件 port 生成路径**:把 IGPO `generation.py` 搬为 `scrl/llm_agent/igpo_generation.py`(只改 import / 对齐
   本仓库 verl 接口,不改数学与控制流);在 `ray_trainer` / 生成入口加"按 `adv_estimator=igpo` 选 manager"的选择器。
   现有 `generation.py` 不动。**目标:单 agent rollout 跑的就是 IGPO 原代码,逐函数可核对。**
5. **跑通单 agent IGPO(Phase 1)**:小规模 2-step smoke。验证 IG 非零、各 turn advantage 不同、无 NaN;
   抽一条样本手算 belief 轨迹核对。**此步通过 = IG 核心可信,再进多 agent。**
6. **新增** `scrl/llm_agent/interleaved_generation.py`:交替式主循环(外层 Planner 循环 + 每轮 belief;
   内层复用现有 `LLMGenerationManager.run_llm_loop` 跑冻结 Executor)。
7. **改造** `verl/workers/reward_manager/multi_agent.py`:从"共享 F1"改为输出 **Planner per-turn 奖励向量**
   (各 turn IG / 末轮 F1),复用步骤 2 的 belief。
8. **接线 Phase 2b**:Planner LoRA 走步骤 3 的 IGPO 优势(group=prompt index);Executor 冻结。
9. **config**:`multi_agent.interleaved.enable`、`multi_agent.max_planner_turns`、
   `algorithm.info_gain_type=log_prob_diff`、`algorithm.info_gain_norm_mode=separate`、`algorithm.gamma`、
   `algorithm.ig_group_mode={global|turn_group}`(5.2 的归一化分组开关,新增)、
   `algorithm.adaptive_turn_clip={true|false}`(可选,A²TGPO 自适应 clipping)。

> 关键:**步骤 2–5 先在单 agent 路径打通并验证数值**,把"belief/优势是否正确"与"交替式架构是否跑通"
> 两类风险解耦,避免在脆弱的多 agent rollout 上同时 debug 数值和架构。

## 8. 实验与消融(论文主线)

固定架构 = 交替式(**一次性 DAG 已废弃,multi-agent 路径即交替生成**),**消融奖励与归一化**(干净对照):

1. **基线 B**:单 agent IGPO(Phase 1)。
2. **Hi-IGPO + global 归一化**:交替式 + Planner turn-level IG,原生 IGPO 全局归一化。
3. **Hi-IGPO + turn_group 归一化(主方法)**:在 2 基础上换 A²TGPO 的 (prompt, turn-index) 归一化。
4. **消融**:`info_gain_norm_mode` separate vs joint;自适应 turn 级 clipping on/off。

> 注:原"DAG + 共享 F1"基线 A 已**移除**(决定放弃 DAG,multi-agent 直接=交替式)。对照主要落在
> 单 agent IGPO(B)↔ 交替式 Hi-IGPO,以及归一化/separate-joint 的消融轴。

**指标**:hard QA 的 F1/EM;收敛速度 / 数据效率;平均 Planner 轮数(看是否学会适时停止)。
**规模**:Qwen3-4B,小 batch,`grpo_n` 4~8,各变体从同一 warm start 分叉。

## 9. 风险与回退

- **最大风险**:交替式重写碰现有较脆弱的 multi-agent rollout(OOM / 权重同步 / 长度对齐,见
  `docs/design/dual_agent_smoke_test.md`)。**回退**:DAG 已废弃、不再作为回退;若 Phase 2b 交替式跑不通,
  回退到 **Phase 1 单 agent IGPO**(已验证可跑通)作为可发表的最小成果,交替式作为加分项继续 debug。
- **成本**:串行变长。**对策**:严格限 `T_p`、小 batch、先在 2-step smoke 上验证。
- **非平稳**:靠冻结 Executor(Phase 2b)规避。
- **数值**:belief/优势复用 IGPO 已验证实现降低风险;Phase 1 先验证数值正确再进 Phase 2b。

## 10. Future Work(写进论文留白)

- Phase 2a:Executor 内层 turn-level IG 训练 → 联合优化。
- 跨区间因果信用回流、Shapley 细粒度子任务分摊、熵去噪。
- 无 ground-truth 的语义信息增益(Self-Induced Outcome Potential / Cycle-Consistent Search 思路)。
- **超长程深度研究 + 真实浏览(对标 DR-Venus)**:IGPO 团队已把同一信息增益信用分配扩到 **200+ turn 的 deep research**
  并训出 4B 的 **DR-Venus**(arXiv:2604.19859,代码在 inclusionAI/DR-Venus/RL),在 **BrowseComp / BrowseComp-ZH**
  等高难 benchmark 上验证有效。我们当前基础 IGPO 仓库是 **snippet-only 多跳 QA** 版;后续可沿 DR-Venus 方向把
  Hi-IGPO 的分层 IG 用到**带 `browse_webpage` 的长程开放网页研究**(我们已保留 browse 工具,IG 与工具无关 → 天然可延伸),
  并在 BrowseComp 类 benchmark 上评测。属"换更难任务 / 更长 horizon"的纵深,与本设计的"分层信用"创新正交、可叠加。

## 11. 统一框架:冻结(2b)与联合训练(2a)是同一框架的两个取值

把 Phase 2b 和 Phase 2a 看成**同一个联合训练框架**的两个取值,而不是两套独立设计。
**rollout 生成流程完全不变**(交替生成 + 每 turn 算 `Bel_t` → `IG_t`);变的只有「IG 怎么分」和「谁更新参数」两个**正交旋钮**:

- **旋钮 1 — IG 分配权重 λ**:planner 拿 `λ·IG_t`,executor 拿 `(1−λ)·IG_t`,各自 scatter 到
  **自己那条 DataProto**(我们已保留的双 rollout)上,各算各的 group-relative turn advantage。
- **旋钮 2 — executor 是否进 optimizer**(`freeze_executor`)。

| 配置 | λ | executor 更新? | = |
|---|---|---|---|
| **Phase 2b(主线/现在)** | 1.0(IG 全给 planner) | ❌ 冻结 | 当前设计(§5.0) |
| **Phase 2a(联合,future)** | (0,1) | ✅ | 一般情形 |
| 退化校验 | 1.0 | — | + 单 agent ⇒ 原生 IGPO |

**关键洞察:冻结是「executor 的 IG 份额无所谓」的角点**——它不进 optimizer,给它的 reward 算了也白算。
所以代码上 2b 就是 `freeze_executor=True` 让 `scatter_planner_token_rewards` 吃下全部 IG;2a 只要把 λ 调出来、
给 executor 那条 DataProto 也接上 advantage 即可,**无需重写数据流**。

### 11.1 实证锚:冻结是下界档,不是终点
**M-GRPO**(arXiv:2511.13288,vertical 多 agent deep research,benchmark = GAIA / XBench-DeepSearch / WebWalkerQA,
与本项目同赛道)实测:**联合训练 > single-agent GRPO > multi-agent GRPO with frozen sub-agents**。
即「冻结 Executor 的分布偏移固化」是被测出来的真实瓶颈,而非纯理论担忧。
**边界条件**:**CODA 双脑**(arXiv:2508.20096)选择冻结 executor 反而好——但其 executor 是经海量 grounding 预训练、
强泛化的成熟模型。冻结能成立的前提是**执行器本身足够通用**;我们 Phase-1 的 executor 只见过「一体化单 agent」的 plan 分布,
恰好不满足 → 这正是 Phase 2a 要解的问题。

### 11.2 λ 取什么:从固定标量到反事实
「按权重分」里的权重本身就是 Phase 2a 的核心研究点,有一条升级谱(不破坏框架):
```
固定标量 λ   →   反事实估计 λ_t(CCPO 式,arXiv:2603.21563)   →   可学习 λ
  省                我们已有 belief 机制,加一次反事实前向即可拆          贵
```
- 反事实拆分回答「这一跳 belief 上升,是 planner 问得准还是 executor 查得好」:
  planner 贡献 `≈ Bel(task_t, findings_t) − Bel(弱化 task_t, findings_t)`;
  executor 贡献 `≈ Bel(task_t, findings_t) − Bel(task_t, 弱化 findings_t)`。代价:每 turn 多一次 belief 前向。
- **耦合点(务必同时上)**:`IG_t` 是 (task_t + executor findings) 的**联合产物**。冻结时全归 planner 无碍;
  **一旦解冻,不拆 IG 就是 credit 串扰**(planner 因 executor 查得好而被奖励,反之亦然)。故「解冻 executor」与「IG 归因拆分」是一对。

### 11.3 Phase 2a 才会撞上的工程坑(现在不做,留 TODO)
- **executor advantage 的 group 归一化**:executor 每个 subtask 的调用次数不齐,需 M-GRPO 式
  **trajectory-alignment**(padding/mask 成 fixed-size batch,不破坏 group baseline)。
- 备选 credit-assignment 路线(与上面 λ 谱并行):**HiPER**(arXiv:2602.16165)的分层优势 HAE
  (planner 优势 = subgoal 执行段聚合 return,executor = 段内细粒度;证明无偏 + 低方差)。

### 11.4 渐进路线(不 foreclose)
```
Phase 2b (现状)   : freeze + λ=1，IG_t 全给 Planner。站得住的 baseline（M-GRPO 已证为下界档）。
   │  低成本补丁    : ①离线缓存刷新 Executor；或 ②只惩罚"执行失效"的轻量正则（非无差别 KL，避免压制 planner 进化）。
   ▼
Phase 2a (目标)   : 解冻 Executor，双通道联合训。λ 先用固定值跑通，需要精度再升级到反事实拆分（11.2）。
                    依托已保留的双 rollout 架构，rollout 流程不变。
```

## 附:远程同步

按 `CLAUDE.md`:先改本地 → rsync 到 `zjx@10.35.2.238:/home/zjx/self_llm/self-researcher`;
训练前检查 GPU 空闲;训练中每 30s 监控。
