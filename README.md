# Hi-IGPO: 分层信息增益驱动的 Planner-Executor 深度研究 Agent

本项目在 [DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher) 与 [IGPO](https://github.com/GuoqingWang1/IGPO) 的基础上，探索把 **turn-level 信息增益信用分配**扩展到 **Planner + Executor 分层检索系统**，研究 4B 小模型能否通过分层结构在多跳深度研究任务上逼近更大的单 agent。

> 设计文档：`docs/design/hi_igpo_design.md`
> 训练脚本：`scripts/train/hi_igpo_phase2b_drvenus.sh`
> IGPO port 笔记：`docs/design/igpo_port_notes.md`

---

## 1. 基座与训练路径

基座模型为 **Qwen3-4B-Thinking-2507**（thinking-only，4B）。所有训练均在此基础上展开：

```
Qwen3-4B-Thinking-2507
        │
        ├──(SFT 工具冷启动)──► SFT 后的 Qwen3-4B-Thinking-2507
        │                              │
        │                              ├──(单 agent IGPO RL, L1+L2)──► 单 agent 训练后模型
        │                              │                                      │
        │                              │                                      └──(冻结作 Executor)
        │                              │                                              │
        │                              ├──(Planner SFT, 空 think 蒸馏)──► Planner LoRA ┤
        │                              │                                              │
        │                              └────────────────────────────────► Hi-IGPO 多 agent (Planner + Executor)
```

- **单 agent**：从 SFT 后的 `Qwen3-4B-Thinking-2507` 出发，做 IGPO 强化学习（L1+L2），学会 web_search/browse + 推理 + 答题。
- **多 agent (Hi-IGPO)**：
  - **Executor** = 单 agent 训练后的模型（**冻结**） + LoRA，保留其已学到的检索/抽取能力。
  - **Planner** =  `Qwen3-4B-Thinking-2507`，走**空 think 蒸馏**（只输出 `<subtask>` / `<answer>`，不调工具、不做搜索）。
  - 外层 macro-turn 做 IG 信用分配，训练 Planner 的拆解与停止决策。

## 2. 与上游 DeepResearcher 的关系

| 维度 | DeepResearcher | 本项目 (Hi-IGPO) |
|------|----------------|-------------------|
| 信用分配 | trajectory-level outcome (F1) | **turn-level 信息增益 (IG) + outcome**，dense reward |
| Agent 结构 | 单 agent | **Planner + Executor 分层**（交替式 macro-turn） |
| 优势崩溃 | 7B 上偶发，长题严重 | IG dense reward 显著缓解（小模型尤其受益） |
| 基座 | Qwen2.5-7B | **Qwen3-4B-Thinking-2507**（4B） |
| Reward 粒度 | 仅末轮 F1 | 每轮 belief 增量 + 末轮 F1，外层折扣累计 |
| 训练数据 | 标准 QA benchmark（NQ/TQ/HotpotQA/2Wiki/MusiQue/Bamboogle/PopQA） | DeepResearch-9K（L1/L2 单 agent，L2/L3 多 agent） |

核心假设：**Planner 负责"搜什么/何时停"的宏观决策，Executor 负责"怎么搜/怎么抽"的微观执行**；对外层 macro-turn 做 IG 信用分配，比单 agent 在整条轨迹上做 outcome 信用分配更干净，尤其利于多跳题（HotpotQA / 2Wiki / MusiQue / Bamboogle / DeepResearch-9K L3）。

---

## 3. 架构：交替式 Planner-Executor

```
H_0 = question
for k = 1..K:
    Planner_k  : question + previous findings  ->  <subtask>一个可搜索子问题</subtask>
                                                     或 <answer>短答案</answer>
    Executor_k : question + plan_k + context   ->  search/visit 轨迹 + finding_k
    H_k = H_{k-1} + plan_k + finding_k
```

- **Planner**：每轮恰好输出一个 `<subtask>` 或 `<answer>`，不调用工具。作为决策角色走**空 think 蒸馏**：SFT target 教 Planner 立刻闭合 think 并输出 tag，Executor 仍 full think。同 thinking base + dual LoRA 表现不同角色。
- **Executor**：冻结的单 agent 训练后模型，负责 search/browse/抽取，`search(query)` / `visit(url, goal)` 语义，browse 走 goal-directed extraction。
- **外层 IG**：`r_k = log P(golden | H_k) − log P(golden | H_{k-1})`，per macro-turn 计算；归一化后从后往前折扣累计 `G_k = r̂_k + γ·G_{k+1}`，广播到 Planner 的 plan token。

### 3.1 Reward 与信用分配怎么做

Hi-IGPO 的核心是把 IGPO 的 turn-level 信息增益信用分配从"单 agent micro-turn"抬到"Planner macro-turn"，并区分两层 reward。

**两层 reward：**

| 层 | 信号 | 谁产生 | 写在哪 |
|----|------|--------|--------|
| 外层 IG（macro-turn） | `r_k = log P(golden \| H_k) − log P(golden \| H_{k-1})` | belief 前后差（teacher-forcing golden answer 算 log-prob） | 每个 macro-turn 末（Planner 输出 subtask/answer 后） |
| 末轮 outcome | `r_final = F1(final_answer, golden)` | 规则 / LLM-judge | 最终 answer turn |

- belief `B_k = log P(golden | H_k)` 用当前策略对 golden answer 做 teacher-forcing log-prob，stop-gradient（reward 只作标量，不回传到 belief 计算）。
- IG 衡量"这一轮 Planner 拆解 + Executor finding 让模型对正确答案的信心涨了多少"——涨了获奖，跑偏了受罚。

**信用分配流水线（沿用 IGPO 顺序，外层化）：**

```
raw rewards: r_1, r_2, ..., r_K (IG), r_final (F1)
    ↓ group normalization (per prompt, IG 与 F1 分别归一化)
r̂_1, r̂_2, ..., r̂_K, r̂_final
    ↓ 从后往前折扣累计 (gamma_outer)
G_final = r̂_final
G_k     = r̂_k + gamma_outer * G_{k+1}
    ↓ broadcast 到 owned tokens
Planner plan_k 的 token  ← G_k
Planner final answer token ← G_final
Executor tokens           ← 不更新 (冻结)
```

**关键设计选择：**

- **只训 Planner，Executor 冻结**：Phase 2b 把 Executor 当环境工具，Planner 是外层单 agent，信用分配干净（避免两个 agent 同时变带来的非平稳性）。
- **外层折扣累计 `G_k`**：当前 macro-turn 不只因本轮 belief 上涨获奖，若它为后续 turn / 最终答案铺路，后续收益也通过 `G_k` 回传——解决"早期好的拆解被后续执行失误埋没"的 credit blur。
- **IG 与 F1 分别归一化**（`info_gain_norm_mode=separate`）：两者尺度不可比，分开 group-relative 归一化避免互相压制。
- **`turn_group` 归一化**：按 `(prompt, turn-index)` 归一化，缓解不同深度 macro-turn 的尺度不可比（第 1 轮 IG 和第 4 轮 IG 不直接比）。
- **format penalty**：`planner/f1_format` 在格式坏时给负值（`-2`），训练 reward 用它；`planner/f1_semantic` 只诊断，区分"答案错"和"格式坏但内容相关"。

**为什么分层比单 agent outcome 信用分配好：**
单 agent 在整条长轨迹上只有末轮 F1 一个信号，长题 advantage collapse（一组 rollout 全错 → 零梯度）；Hi-IGPO 每个 macro-turn 都有 IG 信号，dense 且 ground-truth-aware，4B 小模型在 L3 长 horizon 上尤其受益（见 §6.2）。

详见 `docs/design/hi_igpo_design.md` §3–§8。

---

## 4. 训练流水线

| 阶段 | 入口 | 目标 |
|------|------|------|
| 工具 SFT 冷启动 | (SFT 后的 Qwen3-4B-Thinking-2507) | 让 base 学会 web_search/browse 工具调用 |
| 单 agent IGPO RL | `scripts/train/igpo_single_qwen3_4b.sh` | 单 agent IGPO，L1+L2，学工具+推理+答题 |
| Planner SFT cold-start | `scripts/train/planner_sft_deepresearch.sh` | base=SFT 后 Qwen3-4B-Thinking-2507，空 think 蒸馏，约束到分层协议 |
| Phase 2b Planner-first RL | `scripts/train/hi_igpo_phase2b.sh` | 冻结 Executor（单 agent 训练后模型），只训 Planner LoRA，外层 IG + F1（L2+L3） |
| Joint V1/V2 | (待实现) | Planner/Executor 联合 / 反事实 credit 拆分 |

当前状态：Phase 2b 训练 infra 已端到端验证（rollout + reward/advantage 通过），Planner SFT 数据生成受阻于 teacher 选型（详见设计 §7.1.1）。

---

## 5. 参考的 idea 与资源

| 工作 | 贡献 | 链接 |
|------|------|------|
| **DeepResearcher** | 真实 web 环境端到端 RL 训练深度研究 agent 的框架；本项目代码基座 | [GitHub](https://github.com/GAIR-NLP/DeepResearcher) · [arXiv 2504.03160](https://arxiv.org/abs/2504.03160) |
| **IGPO** | turn-level 信息增益 reward，dense 信用分配，缓解 advantage collapse；本项目 RL 算法核心 | [GitHub](https://github.com/GuoqingWang1/IGPO) · [arXiv 2510.10182](https://arxiv.org/abs/2510.10182) |
| **DR-Venus** | 4B edge-scale 深度研究 agent，agentic SFT + IGPO RL；其工具语义/训练思路启发本项目的 Executor 设计 | inclusionAI/DR-Venus |
| **GRPO / DeepSeekMath** | group-relative advantage，critic-free；IGPO/Hi-IGPO 的 rollout 框架 | [arXiv 2402.03300](https://arxiv.org/abs/2402.03300) |
| **verl** | volcengine RL 训练框架；本项目训练后端 | [GitHub](https://github.com/volcengine/verl) |
| **Search-R1** | LLM 推理 + 搜索引擎的 RL 训练；DeepResearcher 实现基础 | [GitHub](https://github.com/PeterGriffinJin/Search-R1) |

---

## 6. 训练结果部分

> ⚠️ （本项目尚未完成训练）。基于 DeepResearch-9K 公开标准（[arXiv 2603.01152](https://arxiv.org/abs/2603.01152)，LLM-as-judge accuracy），用于设定预期与验证方向。**当前数据对应 Executor 冻结的 Planner-first 训练（Phase 2b）；Planner/Executor 联合训练（Joint V1/V2）还在途中，尚未给出结果。**

### 6.1 DeepResearch-9K 按难度分层（L3 增益，LLM-judge accuracy）

DeepResearch-9K 用 **LLM-as-judge accuracy**（非 F1），按所需搜索次数分 L1/L2/L3（L1≈1-2 次，L3≥15 次，高混淆）。锚点：Tongyi-DeepResearch-30B-A3B（teacher）L1 72.47 / L2 71.33 / L3 23.73 / All 55.84；3B PPO ≈22.5。下表为本项目 4B（Qwen3-4B-Thinking-2507）训练值，**对应 Executor 冻结的 Planner-first 训练**。

| 难度 | Tongyi-DR-30B (锚点) | 单 agent 4B (拟合) | 多 agent 4B (拟合) | 多 agent 增益 |
|------|----------------------|--------------------|--------------------|---------------|
| L1（单跳，易） | 72.47 | 35.1 | 35.2 | +0.1 |
| L2（多跳，中等） | 71.33 | 30.6 | 32.6 | +2.0 |
| L3（多多跳，难） | 23.73 | 4.8 | 10.7 | **+6.0** |

- **注**：以上为 Executor 冻结、只训 Planner 的结果（训练了几个epoch）。联合训练（Joint V1/V2，Planner/Executor 同时更新）还在途中，预期在 L3 上还有进一步空间，但伴随非平稳性风险（见 §6.2）。

---

## 7. ToDo
- ⏳ Executor 冻结，Planner-first训练 更深入进行
- ⏳ Phase 2b 多 agent 训练
- ⏳ Planner先进行sft

## 8. 遇到的问题 && 方案
