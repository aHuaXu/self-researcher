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
**不整库替换,按文件 diff/cherry-pick 增量移植。** 已核对的关键差异如下。

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

**生成逻辑(本仓库有,diff 出 IG 钩子):**
- `scrl/llm_agent/generation.py`:diff IGPO 版,cherry-pick "每轮调用 belief → 算 `IG=log_prob_diff` →
  写入 per-turn 奖励张量 + `turn_end_positions`" 的代码块。

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
4. **diff 生成逻辑**:cherry-pick IGPO `generation.py` 的 IG 钩子,使单 agent rollout 每轮算 IG 并写入奖励张量。
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

固定架构 = 交替式,**消融奖励与归一化**(干净对照):

1. **基线 A**:DAG + 共享 F1(当前代码)。
2. **基线 B**:单 agent IGPO(Phase 1)。
3. **Hi-IGPO + global 归一化**:交替式 + Planner turn-level IG,原生 IGPO 全局归一化。
4. **Hi-IGPO + turn_group 归一化(主方法)**:在 3 基础上换 A²TGPO 的 (prompt, turn-index) 归一化。
5. **消融**:`info_gain_norm_mode` separate vs joint;自适应 turn 级 clipping on/off。
6. **(可选)架构对比**:交替式 vs 一次性 DAG。

**指标**:hard QA 的 F1/EM;收敛速度 / 数据效率;平均 Planner 轮数(看是否学会适时停止)。
**规模**:Qwen3-4B,小 batch,`grpo_n` 4~8,各变体从同一 warm start 分叉。

## 9. 风险与回退

- **最大风险**:交替式重写碰现有较脆弱的 multi-agent rollout(OOM / 权重同步 / 长度对齐,见
  `docs/design/dual_agent_smoke_test.md`)。**回退**:若 Phase 2b 交替式跑不通,退回"保留 DAG + P-fine
  (把子任务 IG 回传到提出它的 planner token 段)",仍保留信用分配创新。
- **成本**:串行变长。**对策**:严格限 `T_p`、小 batch、先在 2-step smoke 上验证。
- **非平稳**:靠冻结 Executor(Phase 2b)规避。
- **数值**:belief/优势复用 IGPO 已验证实现降低风险;Phase 1 先验证数值正确再进 Phase 2b。

## 10. Future Work(写进论文留白)

- Phase 2a:Executor 内层 turn-level IG 训练 → 联合优化。
- 跨区间因果信用回流、Shapley 细粒度子任务分摊、熵去噪。
- 无 ground-truth 的语义信息增益(Self-Induced Outcome Potential / Cycle-Consistent Search 思路)。

## 附:远程同步

按 `CLAUDE.md`:先改本地 → rsync 到 `zjx@10.35.2.238:/home/zjx/self_llm/self-researcher`;
训练前检查 GPU 空闲;训练中每 30s 监控。
