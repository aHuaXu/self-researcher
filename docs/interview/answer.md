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
