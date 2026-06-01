# Agentic Search 面试问题回答

## 1. 你怎么定义 Agentic Search 的 action space？

**显式 action（3 个）：**
- `web_search(query: list[str])` — 向搜索引擎发请求，返回结构化摘要
- `browse_webpage(url_list: list[str])` — 深度阅读搜索结果中的网页，返回逐页摘要
- `stop` — 不再调用工具，直接生成 `<answer>` 作为最终回答

**隐式行为（通过 RL 自然涌现）：**
- **query 改写**：体现在 `web_search` 的参数里，模型会根据前几轮搜索结果调整措辞，这是 search action 内部的决策，不需要单独建模
- **反思**：模型在连续 tool_call 之间的生成文本中自然涌现。没有开 thinking mode，但 RL 训练后模型会在输出中自发产生"上一轮结果不够/不相关，换个角度"的文本，再发起新的搜索。这是纯 RL 信号驱动的涌现行为，不依赖任何显式的 CoT 机制
- **引用**：不是独立 action，而是答案生成过程中的文本行为

**为什么这样设计：**

1. RL 训练稳定性 — action space 越小，探索越高效，GRPO 的 group 内方差信号越清晰
2. Credit assignment 更简单 — 如果把 query 改写单独建模，就需要额外的 reward signal 来评判改写好坏，而把它融入 search action 后，好的 query 自然通过最终 F1 reward 被强化
3. 我们通过两阶段课程设计来渐进扩展系统级 action space：Phase 0 单 Agent 只有 `{search, browse, stop}`；Phase 1 引入 Planner，把"规划分解"从隐式思考提升为显式结构化输出（带依赖 DAG 的子任务列表），但这是在 agent 维度的扩展，而非单个 agent 的 action space 膨胀

---

## 2. 搜索触发策略怎么训练？

**纯 RL（GRPO）端到端训练**，没有 SFT warmup 也没有 rule-based router。

模型的每一步生成面临一个隐式决策：输出 tool_call（触发搜索/浏览）还是直接输出 `<answer>`（停止搜索）。这个决策完全由模型自己学习，训练信号来自最终的 F1 reward。

**为什么不用 SFT imitation：**
- 没有"什么时候该搜"的标注数据，搜索时机本身没有唯一正确答案
- Imitation learning 会让模型学到"看到某类问题就固定搜 N 次"的 pattern，缺乏泛化性
- 端到端 RL 让模型自己探索出最优策略：简单问题可能搜一次就停，复杂问题会多轮搜索

**训练信号传递：**
- 搜了且搜到有用信息 → F1 高 → 正 advantage → 强化"该搜"的决策
- 搜了但没找到有用信息 → F1 没提升 → 相对负 advantage → 弱化无效搜索
- 没搜直接答 → 简单题答对了 F1 高 → 强化"不需要搜"的判断
- 没搜直接答 → 难题答错了 F1 低 → 弱化"跳过搜索"的倾向

GRPO 的 group normalization 在这里很关键：同一个 query 的 16 条 rollout 里，有些搜了有些没搜，reward 差异自然形成了"该不该搜"的对比信号。

**两阶段的递进：**
- Phase 0（中低难度）：模型学会基本的搜索触发能力——识别出自己参数知识不够，需要外部信息
- Phase 1（中高难度）：搜索触发变成了 Executor 在 DAG 子任务内的决策，每个子任务可能需要多轮搜索。Planner 的分解质量决定了 Executor 需要搜多深

**没有用 DPO/IPO 的原因：**
DPO 需要 preference pair（好 trajectory vs 坏 trajectory），构造成本高且搜索场景下两条 trajectory 的差异维度太多（query 不同、搜到的网页不同、阅读深度不同），很难归因到"搜索触发"这一个维度。GRPO 通过 group 内自然采样对比，不需要人工构造 pair。

---

## 3. final answer reward 很稀疏时，你怎么做 credit assignment？

**核心思路：没有精确解决 credit assignment，而是通过多层次手段缓解稀疏问题。**

### 层次 1：GRPO group 对比

同一个 query 采样 16 条 rollout，不同 rollout 的搜索路径不同。Group normalization 后，高 F1 的整条 trajectory 所有 token 都会被强化。统计上，跨多个 query 训练后，好的搜索 pattern 会被反复强化。

### 层次 2：Rule reward 提供中间密集信号

```
reward_executor = β × rule_e + (1-β) × f1
```

`rule_e` 评估过程质量：
- 搜索返回了非空结果（+1）— 至少 query 写得合理
- 调用了 browse 深入阅读（+1）— 不只看摘要
- browse 返回内容有效（+1）— 选对了 URL
- 没有触发 max_turns 限制（+1）— 效率合理

这些规则 reward 给每个 action 类型提供了即时反馈，缓解了稀疏问题。

### 层次 3：Dual-Agent 架构缩小归因范围

把一个长 trajectory 切成：
- Planner → 只为分解质量负责（`rule_p + F1`）
- Executor → 只为搜索执行质量负责（`rule_e + F1`）

相比单 Agent 一条 10 轮的 trajectory，每个 Agent 承担的 credit 范围更小、归因更清晰。

### 未实现但可讨论的更精细方案

**1. Process Reward Model (PRM)**
- 思路：训练独立模型对 trajectory 每一步打分
- 优势：直接给中间 step 密集 reward，彻底解决稀疏问题
- 代价：需要大量 step-level 人工标注，且搜索场景下"好搜索"标准本身模糊——一次搜索当时看没用，但后面综合起来有用，标注员也难以判断

**2. Hindsight Credit（事后归因）**
- 思路：最终答案生成完毕后，回过头看哪些搜索结果内容实际出现在答案中 → 这些搜索步骤给额外 bonus reward
- 优势：不需要人工标注，纯自动化；奖励"确实被用上了"的搜索
- 代价：可能导致"先搜再抄"的 shortcut，鼓励逐字引用而非理解整合；有些搜索提供背景理解但没被直接引用，这类贡献会被漏掉
- 这是最值得尝试的 future work 方向，实现轻量且和 GRPO 框架兼容

**3. TD-style Value Function**
- 思路：像 PPO 训练 critic 网络，用 `r + γV(s') - V(s)` 做逐步 advantage 估计
- 优势：理论上最精确的 credit assignment
- 代价：GRPO 的核心优势就是不需要 critic（省显存、避免 critic bias）。搜索场景状态空间极大，critic 很难估准。加了 critic 等于回到 PPO，失去了 GRPO 的简洁性

**面试建议表述：**
> "我们选择了不依赖精确 credit assignment 的路线，而是通过架构设计和辅助 reward 来缓解。如果资源充足，hindsight credit 是我最想尝试的方向，因为它免标注、实现轻量，且和现有 GRPO 框架兼容。"

---

## 4. query generation 怎么评估？

**核心立场：不直接评估 query 本身，通过下游效果间接评估。**

### 实际做法

1. **最终 F1 reward** — 好 query → 搜到相关结果 → 最终答案好 → 高 reward → 该 query 被强化
2. **Rule reward 弱信号** — `executor_rules` 检查搜索返回是否非空（`len(result) > 10`），最粗粒度的 query 质量代理：query 太差连结果都搜不到

### 为什么不单独评估 query

- **没有标准答案** — 同一个信息需求可以用多种 query 表达，都是"好 query"
- **结果导向优于过程导向** — 一个"看起来不好"的 query 如果搜到有用信息就是好 query；一个"看起来专业"的 query 搜不到东西就是坏 query
- **搜索引擎是黑盒** — query 质量和 backend 耦合，同样的 query 在不同引擎效果不同，单独评估 query 没有脱离 backend 的意义

### 怎么验证 query 能力在进步

- `rule_e` 中"搜索非空"得分趋势上升 → query 质量提升
- Rollout 日志中观察到 query 多样性和针对性增强（涌现的改写行为）
- F1 涨 + 搜索轮次减少 → 用更少搜索找到更好信息 → query 更精准

### 可展开的高级讨论

- **Retrieval recall 作为中间评估**：搜到的文档是否包含答案，但需要标注"答案在哪些文档里"，成本高
- **LLM-as-judge 评估 query 相关性**：引入 judge 偏差且增加推理成本，trade-off 不划算

---

## 5. grounding reward 怎么避免 reward hacking？

> 模型引用大量无关来源但答案正确，怎么扣分？

### 我们遇到的真实 reward hacking

训练单 Agent 阶段，模型在前 ~10 个 step 还会正常多轮搜索，之后**几乎完全放弃工具调用，直接给出答案**，reward 卡在 0.1~0.2 上不去。

**根因分析：**
- 训练初期搜索能力弱 → query 质量差 → 搜到的结果经常不相关 → F1 提升不稳定
- 直接用参数知识猜 → F1 虽低但方差小、稳定
- GRPO group 内对比时，当所有 rollout 都不搜了，advantage 全部接近 0，模型卡在局部最优

**解决方案：No-Tool Penalty**

```python
NO_TOOL_PENALTY = -0.3

def compute_score(solution_str, ground_truth, val_type='f1') -> float:
    has_tool_call = "<tool_call>" in solution_str_lower

    # 格式错误处理
    if not check_tags_balance(solution_str_lower):
        if has_tool_call:
            return 0.0       # 搜了但格式错 → 不惩罚
        return -1.0          # 没搜且格式错 → 最严厉惩罚

    # 没有提取到答案
    if not answer_match:
        if has_tool_call:
            return 0.0       # 搜了但没给答案 → 不惩罚
        return -1.0

    # 正常计算 F1 ...

    # 核心：没调用工具 → 固定惩罚
    if not has_tool_call:
        return NO_TOOL_PENALTY  # -0.3

    return max_score  # 正常 F1 (0~1)
```

**设计要点：**

| 情况 | 得分 | 原因 |
|------|------|------|
| 搜了 + 答对 | F1 (0~1) | 正常奖励 |
| 搜了 + 格式错/没答案 | 0 | 不惩罚，鼓励探索 |
| 没搜 + 不管答对答错 | -0.3 | 打破"不搜直接答"的局部最优 |
| 没搜 + 格式错 | -1.0 | 最严厉，什么都没做对 |

**为什么有效：**
1. 彻底打破局部最优 — 即使猜对也是 -0.3，搜了哪怕搜错也是 0，GRPO group 内"搜了"的 rollout 永远比"没搜"的 advantage 高
2. 惩罚力度适中 — -0.3 而非 -1.0，不会让梯度爆炸
3. 和格式惩罚形成层次 — 格式错(-1.0) > 不搜(-0.3) > 搜了但没搜好(0) > 搜到了(F1)

**效果：** 加了 penalty 后模型重新开始探索搜索策略，reward 恢复上涨。

### 关于"引用大量无关来源但答案正确"

诚实说，我们目前没有显式的 grounding 验证机制。F1 reward 只看最终答案和 golden answer 的 token 重合，不关心答案是否真的来自搜索结果。

**实践中这个问题没有爆发的原因：**
- 中高难度 multi-hop 问题，参数知识大概率答不对，必须依赖搜索
- Rule reward 中 browse 要求 `len(result) > 50` 才给分，不是随便点 URL
- GRPO group 对比下，无效搜索不会获得额外优势（短 trajectory 梯度更集中）

### 如果要进一步防 hacking 的方向

- **Faithfulness reward**：用 NLI 模型检查 answer 和搜索结果间的蕴含关系
- **Counterfactual test**：去掉搜索结果后模型仍给出相同答案 → 说明没真正用搜索 → 扣分（推理成本翻倍）
- **Citation 验证**：要求输出标注来源 URL，验证引用内容是否真来自该 URL（需改 action space）。这是一个值得尝试的方向，能同时提供 grounding 保障和用户可解释性

**更深入的优化方案：**

**方案 A：PPO + per-step reward at effective search position**

不只在最终给 reward，而是在 rollout 中每次"有效搜索"结束位置直接给正 reward：

```
Rollout: [search_1] → result_1 → [search_2] → result_2 → [answer]
                ↑ reward=0          ↑ reward=+r          ↑ reward=F1
         (结果和答案无关)      (结果和答案相关)
```

- 一箭双雕：解决 credit assignment（好搜索立即得到正反馈）+ 解决 grounding（只有搜到相关内容才奖励）
- "有效搜索"判定：搜索结果和 golden answer 的 token overlap / embedding similarity 超阈值，或用 NLI 判断蕴含关系
- 代价：需要从 GRPO 切到 PPO（需训练 critic 网络，架构大改），且"有效"的自动判定可能有噪声

**方案 B：GRPO 框架内的折中 — 搜索相关性加入 rule_reward**

不换 PPO，在现有 GRPO 框架下，把"搜索结果和 golden answer 的相关度"作为新的 rule_reward 维度：

```python
def executor_rules(..., golden_answer: str) -> float:
    # 现有规则 ...

    # 新增：搜索结果和答案的相关性
    for step in trajectory:
        if step["tool"] == "web_search":
            relevance = token_overlap(step["result"], golden_answer)
            if relevance > threshold:
                score += 1.0  # 有效搜索 bonus
```

- 优势：不改训练框架，实现轻量，和现有 GRPO 完全兼容
- 代价：仍然是 trajectory-level reward（不是 per-step 的），只是让 rule_reward 更能区分"有效搜索"和"无效搜索"的 rollout

**方案 C：惩罚"大量无关搜索"— 直接回答"引用无关来源怎么扣分"**

针对"模型搜了很多但搜索结果和答案无关"的场景，加负向规则：

```python
# 计算搜索有效率
relevant_count = sum(1 for step in trajectory
                     if token_overlap(step["result"], golden_answer) > threshold)
total_search = sum(1 for step in trajectory if step["tool"] == "web_search")

if total_search > 0 and relevant_count / total_search < 0.3:
    score -= penalty  # 大量搜索但大部分无关 → 扣分
```

逻辑：答案正确照常给 F1 分，但搜索行为"注水"（大量无关搜索）时通过 rule_reward 把这部分分扣回来。这样模型会学到"精准搜索少量高质量结果"优于"广撒网碰运气"。

**面试建议表述：**
> "训练早期遇到过模式坍缩——模型放弃搜索直接答题。根因是搜索能力弱时 F1 差距不大，GRPO 无法产生有效对比信号。我们加了 no-tool penalty(-0.3)，让'不搜'永远劣于'搜了'，有效打破了局部最优。至于 grounding faithfulness，目前靠数据难度和 rule reward 隐式保障。未来优化方向有两个：一是在 GRPO rule_reward 中加入搜索结果与答案的相关性指标；二是更激进地切到 PPO，在有效搜索位置给 per-step reward，同时解决 credit assignment 和 grounding 问题。"

---

## 7. 检索结果和模型参数知识冲突时，训练目标是什么？

**核心立场：既不「永远信检索」，也不「永远信参数」，而是 outcome-driven —— 在必须调用工具的前提下，让最终答案对齐 golden answer。**

### 我们实际在优化什么

当前 reward 没有显式的 source reliability 或 conflict resolution 规则：

```
reward = F1(final_answer, golden_answer)   # 主信号，不关心答案来自参数还是检索
       + no-tool penalty (-0.3)            # 强制「搜过」，解决的是「搜不搜」而非「信谁」
       + rule_e（搜索非空 / browse / 轮次效率）  # 过程质量，不评估来源可信度
```

冲突怎么解，reward 里没写死；**谁跟 golden answer 一致，谁的 trajectory 在 GRPO group 里拿高 advantage**。

### 为什么现阶段还能 work

- **数据分布**：L2+L3 multi-hop 上参数知识往往不够，模型自然学会多轮 search + browse
- **架构辅助**：Dual-Agent 把问题拆成子任务，Executor 跨 wave 注入 findings，比单轮 RAG 更像多源综合
- **行为涌现**：多轮 RL 后模型会自发 cross-check、换 query 再搜（DeepResearcher 观察到的 emergent behavior），但这是隐式学到的，不是 reward 显式约束

### 当前方案的局限

| 局限 | 说明 |
|------|------|
| Outcome-only | F1 不区分「猜对」和「搜对」，冲突场景下可能强化错误 side |
| No-tool penalty 副作用 | 即使参数知识够用，训练里也倾向先搜再答 |
| 无 conflict 专项数据 | 未构造 conflict documents / outdated facts，冲突主要靠真实 web 噪声碰运气 |

### 未来方向：conflict/outdated 合成数据 + curriculum

**两类合成样本：**

1. **Outdated facts** — 参数知识过时（如旧 CEO），检索里有新信息，golden = 新事实 → 训练「易变事实跟检索走」
2. **Conflict documents** — 多个来源说法矛盾（旧新闻 vs 官网），golden = 权威/最新来源 → 训练「多源交叉验证再选边」，不是抄第一条 snippet

**Curriculum（分阶段加难度）：**

| 阶段 | 数据 | 目标 |
|------|------|------|
| Phase 0–1（已有） | L1–L3 multi-hop | 会搜、会分解、会多步整合 |
| Phase 2（未做） | outdated facts | 参数 vs 检索冲突时跟 golden 走 |
| Phase 3（未做） | conflict documents | 多源矛盾时交叉验证 |
| Phase 4（可选） | 错误前提 / 搜不到 | 拒答或澄清，不硬编 |

配合 **faithfulness reward**（NLI / citation verify，见 Q5）让模型不仅答对，还要证据链成立。

### 推理层的 ideal policy（线上，非当前训练目标）

- 多源不一致 → 继续搜 / browse 深读 / 换 query，而非盲信第一条
- 信号：来源权威度、发布时间、snippet vs 全文一致性
- 静态知识题 → routing 不强制搜索（no-tool penalty 是训练期 bias，线上应做 query routing）

**面试建议表述：**
> "我们的训练目标不是永远信检索，而是 F1 对齐标注答案 + 必须调用工具。冲突消解目前没有显式 reliability scorer，主要靠 multi-hop 多轮搜索和 dual-agent 分解隐式涌现。如果要系统化，我会加 conflict/outdated 合成数据做 curriculum，再配合 faithfulness reward，把『猜对』和『搜对且证据成立』区分开。"

---

## 8. 你们的 verifier 怎么做？

**核心立场：当前训练 reward 只有「F1 + 格式规则 + rule reward」三层，全部是确定性、可复现的函数；没有 LLM judge、NLI、learned RM。**

### 训练期实际走哪条代码

**单 Agent（Phase 0）：**

```
main_ppo.py → NaiveRewardManager → format_and_f1.compute_score
```

**Dual-Agent（Phase 1）：**

```
ray_trainer.py → MultiAgentRewardManager → compute_f1_reward + planner_rules + executor_rules
```

`MultiAgentRewardManager` 文件头注释明确写了：**No LLM Judge dependency**。

训练脚本 `grpo_qwen3_4b.sh` / `grpo_dual_agent.sh` 均未设置 `reward_model.enable=True`，learned RM 未启用。

### Verifier 分层（仅限训练 reward 路径）

| 层级 | 实现 | 代码位置 | 权重 |
|------|------|----------|------|
| Outcome | Token-level F1 | `multi_agent.py:compute_f1_reward` / `format_and_f1.py:compute_score` | 主信号（1-α / 1-β） |
| Format | 标签配对 + `<answer>` 提取 + no-tool penalty | `format_and_f1.py`（**仅单 Agent 路径**） | 格式错 -1.0；无 tool_call -0.3 |
| Process | `planner_rules` / `executor_rules` | `rule_reward.py` | α=0.2, β=0.3 |

Dual-Agent 的 outcome verifier 只对 executor 的 `final_answer` 做 F1，**不走** `format_and_f1` 的 no-tool penalty 逻辑。

### 各 verifier 的误差处理

**1. F1（主 verifier）**

- 预处理：小写、去标点、`split()` 后 token set 算 precision/recall
- 支持 `<|answer_split|>` 多答案取 max F1
- **语义等价但措辞不同**（如中英文混用）→ F1 偏低，**当前无补偿机制**，靠 GRPO group 相对排序缓解
- **多余 token** → precision 被稀释；prompt 要求 `<answer>` 内只写最终答案

**2. 格式规则（单 Agent only）**

```70:125:verl/utils/reward_score/format_and_f1.py
def compute_score(solution_str, ground_truth, val_type='f1') -> float:
    ...
    if not has_tool_call:
        return NO_TOOL_PENALTY  # -0.3
    return max_score
```

- 确定性规则，误差可忽略
- 不对称：搜了但格式错 → 0；没搜 → -0.3

**3. Rule reward（Dual-Agent 过程 verifier）**

```98:123:verl/utils/reward_score/rule_reward.py
def executor_rules(trajectory, max_turns, actual_turns) -> float:
    # 搜索非空 len>10 / 有 browse / browse 内容 len>50 / 未撞 max_turns
```

| 误差 | 处理 |
|------|------|
| 搜到垃圾也算非空 → 假阳性 | β=0.3 权重低，F1 仍是主信号 |
| 规则太粗 | 设计目标是 dense 中间信号，非精确过程评判 |

### 明确没用上的（面试也要主动说）

| 类型 | 状态 | 说明 |
|------|------|------|
| LLM judge | **未接入训练** | `research_agent_design.md` 规划 Writer 阶段用冻结 judge 打报告分，`llm_judge.py` 未实现 |
| NLI | **未实现** | Q5/Q7 讨论的 future work |
| Learned RM | **未启用** | verl 有 `reward_model.enable` 接口，训练脚本未开 |
| MBE | **不在本项目 reward 路径** | 仅存在于上游遗留脚本 `evaluate/cacluate_metrics.py`，API 为占位符 `"YOUR API BASE URL"`，与 `main_ppo` / `MultiAgentRewardManager` 无调用关系 |

`ray_trainer.py` 验证阶段虽调用 `val_type='llm'`，但 `format_and_f1.compute_score` 只处理 `f1` 和 `em`，**`llm` 类型无对应实现，实际等同 F1**。

### GRPO 如何吸收 F1 噪声

同 query 采样 n 条 rollout → group 内 `(reward - mean) / std` → 绝对 F1 误差转为相对排序。全组都错 → advantage ≈ 0。

### 未来演进

| 场景 | Verifier |
|------|----------|
| 有标答 QA（当前） | F1 + rules |
| 无标答研究报告 | 冻结 LLM judge（设计文档，未实现） |
| Grounding | NLI / citation verify（future work） |

**面试建议表述：**
> "我们训练期的 verifier 很克制：outcome 是 token F1 对齐 golden answer，process 是 planner/executor 的 rule reward，单 Agent 阶段额外有格式规则和 no-tool penalty。全部是确定性函数，可复现、无 API 依赖。LLM judge、NLI、learned RM 都没进 RL 主循环——judge 噪声大、不可复现，不适合做每 step reward。上游 evaluate 脚本里有 MBE 代码，但 API 未配置、也不接训练 pipeline，我们实际没用。如果扩展到无标答开放任务，才会考虑冻结 LLM judge。"

---

## 9. 训练数据怎么构造？

### 项目实际用到的数据集

| 文件 | 脚本 | 含哪些 source |
|------|------|---------------|
| `train.parquet`（80K） | `grpo_qwen3_4b.sh` 等 | hotpotqa、2wiki、**nq**、tq |
| `dev.parquet`（875） | 同上 val | 以上 4 个 + popqa、musique、Bamboogle |
| `deepresearch_phase{1,2}.parquet` | `grpo_dual_agent.sh` | DeepResearch-9K（L1/L2 或 L2/L3） |

构造：`scripts/prepare_deepresearch_data.py` 从 HF 拉 DeepResearch-9K，只取 question + final answer；rollout 实时 web 搜索，不 replay teacher trajectory。Reward 统一 token F1（单 Agent 另加 no-tool penalty -0.3）。

未接入训练：`multi-research.parquet`（24 条 open-ended 研究题，无 GT）。

### 面试四类的代表性数据集 & 项目是否使用

| 类型 | 代表性数据集 | 项目是否使用 |
|------|-------------|-------------|
| **真实用户 query** | **NQ**（Google 搜索）、DuReader（中文搜索）、ELI5（Reddit） | **部分**：legacy train 里 **nq 占 12.5%**；Dual-Agent 路径无 |
| **Synthetic multi-hop** | **HotpotQA**、2WikiMultihop、MuSiQue、**DeepResearch-9K** | **是**：hotpotqa + 2wiki（train 75%）；musique/Bamboogle（dev）；DeepResearch-9K（Dual-Agent） |
| **Counterfactual outdated facts** | **FreshQA**、**HoH** | **否**（0%） |
| **Conflict documents** | **ConflictQA**、ConfRAG、Conflicts | **否**（0%） |

另：**TriviaQA**（tq）在 legacy train 占 12.5%，属 trivia benchmark，GT 用 `<|answer_split|>` 多短答取 max F1，不归入上面四类。

### Outdated / Conflict 未使用，构造方式怎么答

三类构造路径（面试常追问，我们 **0% 使用**）：

1. **Memory–context conflict（ConflictQA）**：参数侧常用 **LLM 闭卷当代理**（`memory_answer` + `parametric_memory`），再配 Wikipedia `counter_memory`；不是从权重读「过期」，是构造「闭卷 vs 外部 evidence」冲突对。
2. **Outdated facts（FreshQA / HoH）**：更依赖 **Wikipedia 时间 diff** 和 **FreshQA 式 time-sensitive 标注**（先定当前 GT，再配旧 snapshot / 旧文档当过时 evidence）。
3. **多源 conflict（ConfRAG / Conflicts）**：**真实检索 + 人工**标多网页矛盾 viewpoint；LLM 仅轻量辅助。

Outdated 与 ConflictQA 有重叠（如 CEO 闭卷答旧名、检索是新名），但 outdated **强调时间**，ConflictQA **强调信谁**（闭卷错也可能是记错）。

### 面试建议表述

> "训练数据两条线：Phase 0 用 train.parquet（HotpotQA+2Wiki 75%、NQ 12.5%、TriviaQA 12.5%），Dual-Agent 用 DeepResearch-9K 分 phase curriculum，F1 reward。四类里我们只覆盖 multi-hop 和部分真实 NQ，outdated 和 conflict 都是 0%。若要做：**ConflictQA 这类 memory–context 冲突，参数侧常用 LLM 闭卷当代理来构造；outdated 更依赖 Wikipedia 时间 diff 和 FreshQA 式 time-sensitive 标注；多源 conflict 则靠真实检索加人工。** 我们现用现成 benchmark，不需要这套构造 pipeline。"

---

## 10. 线上怎么做 routing？

**现状：项目没有 per-query router**；搜不搜由 GRPO rollout 隐式决定（Q2）。no-tool penalty（-0.3）是训练 bias，**线上不应 all-in Agentic**。

代码里只有 **部署级** 切换（`generation.py` + `search_engine`）：

| `search_engine` | 工具 | 路径 |
|-----------------|------|------|
| `rag` | Wiki 检索（`TOOLS_FOR_WIKI`） | 普通 RAG：单轮检索 + 生成 |
| `online_search` | `web_search` + `browse_webpage` | Agentic Search：多轮搜 + 深读 |

训练脚本用 `online_search`；默认 yaml 为 `rag`。**完整 routing = query 级 `route()` 决定进哪条 pipeline。**

### 什么场景需要 routing

Routing 解决 **成本/延迟**，不是「模型会不会搜」（那是训练的事）。

| 路径 | 典型场景 | 例子 |
|------|----------|------|
| **不搜** | 静态常识、定义、简单推理、闲聊 | 「Python 里 list 和 tuple 区别」 |
| **普通 RAG** | 单跳事实、企业 KB 可覆盖 | 「我司 XX 产品退款政策」（内部文档） |
| **Agentic Search** | multi-hop、time-sensitive、开放研究 | HotpotQA 类；「2025 年 X 公司 CEO」；对比分析报告 |

**和我们数据的关系**：HotpotQA/DeepResearch-9K → Agentic；NQ 里不少单跳 → 可 RAG/不搜。

### 怎么实现：三层 cascade

```
Query → [L1 规则/小 classifier] → direct | rag | agentic（不确定）
              ↓ 低置信
         [L2 闭卷置信度 / 小 LLM]
              ↓ 仍不确定
         [L3 Agentic Search]（训好的 policy，max_turns 限制）
```

| 层 | 手段 | 延迟 |
|----|------|------|
| L1 | 规则（「最新/2025/对比」→ agentic；短问非时效 → direct）；或 BERT 三分类 | ms 级 |
| L2 | 闭卷生成 + 置信度；KB 检索 top1 分数 > 阈值 → rag | ~100ms |
| L3 | `search_engine=online_search`，Dual-Agent 可选 | 秒级 |

**和项目衔接**：`route(query)` 返回 `direct/rag/agentic` → `direct` 无工具生成；`rag` 设 `search_engine=rag`；`agentic` 设 `online_search` 调 `MultiAgentGenerationManager`。

### 和训练的关系

- **训练**：RL 学 Agentic policy 能力；无 router 标注
- **线上**：router 决定 **何时调用** 这套能力；二者解耦

### 面试建议表述

> "Routing 用在成本和延迟：静态题不搜、单跳 RAG、multi-hop/时效走 Agentic。我们训练用 GRPO 学 search trigger，no-tool penalty 不能照搬线上。代码里已有 rag vs online_search 两条 pipeline，缺的是 query 级 router。实现上三层 cascade：规则或小 classifier 分流，不确定再闭卷看置信度，最后才上 Agentic。Routing 和训练解耦——训的是能力，router 决定什么时候调用。"

---

## 11. latency 和 cost 怎么优化？

### 我们真实训练里的两个瓶颈

**瓶颈 1：Rollout 按 turn 串行，GPU 等工具**

`run_llm_loop` 每轮固定顺序：

```
for turn in range(max_turns):
    vLLM generate（GPU）→ parse → execute_predictions（网络/IO）→ 拼 observation → 下一轮
```

工具执行期间 **vLLM 空闲**；一轮里所有样本都搜完才进入下一轮 generate（`executor_train_flow.md` 亦写明同步调用）。

**已有缓解（同 turn 跨样本并行，非跨 turn）：** `execute_predictions` 用 `ThreadPoolExecutor`——`web_search` 最多 5 并发、`browse_webpage` 最多 4（`TOOL_WEB_SEARCH_MAX_WORKERS` / `TOOL_BROWSE_MAX_WORKERS`）。**turn 之间仍串行。**

**瓶颈 2：生成与训练串联**

`ray_trainer.py` 每 step：

```
gen（含多轮 tool）→ compute_log_prob → reward/adv → update_actor（FSDP）
```

生成和训练 **不能 overlap**；FSDP↔vLLM 还需 `FSDPVLLMShardingManager` 权重同步/切换模式，进一步拉长 step 时间。

### 项目里已有的 cost/latency 手段

| 手段 | 实现 | 作用 |
|------|------|------|
| **Retrieval budget** | `max_turns=5`；`web_search` 每 call 最多 3 个 query | 限制搜索深度 |
| **Early stopping** | 输出 `<answer>` 即从 `activate_list` 移除 | 已答样本不再 generate |
| **Cache** | `search.py` 的 `api_result_dict`，同 query 7 天内复用 | 减 Serper API 调用 |
| **Truncate** | `TOOL_CONTENT_MAX_CHARS=3000` | 减 context 膨胀、下轮 gen 更快 |
| **并行搜索** | 同 turn 多样本 ThreadPool | 减 wall-clock，但 GPU 仍等整批 tool 完成 |
| **Routing（未做）** | Q10 | 简单题不进 Agentic，从根上减 tool 次数 |

### 针对两大瓶颈的可做优化

| 方向 | 做法 |
|------|------|
| **减 tool 等待** | 降 `max_turns` / `agent_grpo.n`；少 browse（browse 最慢）；训练期用小 batch |
| **异步 tool** | 工具放独立 worker 进程/线程池，generate 与 tool IO overlap（需改 rollout 调度，当前未做） |
| **Pipeline 训练** | step N 训练时异步启动 step N+1 rollout（需双缓冲 + 额外 GPU 或 offload infer） |
| **训推分离 GPU** | 部分卡专职 vLLM rollout，部分卡 FSDP 训练，避免模式切换（成本高） |
| **Query reuse** | 同 session 相似 query 走 cache（已有）；Planner 子任务间复用 finding 减重复搜 |
| **Early stopping 加强** | 训练期也可考虑「连续无效搜索 turn 提前停」（rule，未实现） |

### 面试建议表述

> "我们训练 latency 主要卡在两点：一是 rollout 按 turn 串行——vLLM generate 完要等同步 tool，GPU 空等 Serper/browse；同 turn 内已用线程池并行多样本，但 turn 之间仍串行。二是生成和训练串联，每 step 先完整 rollout 再 FSDP update，还有 FSDP-vLLM 权重切换开销。已有手段是 max_turns、early stop、搜索 cache、截断 tool 内容、同 turn 并行搜。进一步优化我会：routing 减少不必要的 Agentic；降 browse 频率；探索训推 pipeline 或 tool worker 异步化。线上还有 retrieval budget 和 query cache；训练侧 cost 靠小 batch 和少轮 GRPO 采样控制。"
