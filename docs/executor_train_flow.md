# DeepResearcher GRPO 训练流程

## 整体架构

```
train_grpo.sh
  → verl.trainer.main_ppo (Hydra 入口)
    → RayPPOTrainer.fit() (训练主循环)
      ┌─────────────────────────────────────────────────┐
      │  每个 step:                                      │
      │  1. Rollout: vLLM 生成 + 多轮工具调用            │
      │  2. Reward: F1 打分                              │
      │  3. GRPO Advantage: 组内归一化                    │
      │  4. Actor Update: PPO clip loss                  │
      │  5. Critic Update: value loss                    │
      │  6. KL Penalty: 防止偏离太远                      │
      └─────────────────────────────────────────────────┘
```

## 1. 启动流程

`train_grpo.sh` 调用 `verl.trainer.main_ppo`，通过 Hydra 加载 `ppo_trainer.yaml` 配置。

关键参数：

| 参数 | 值 | 说明 |
|---|---|---|
| `agent_grpo.n` | 16 | 每个 prompt 生成 16 条 rollout |
| `data.train_batch_size` | 256 | 每步训练 batch |
| `actor_rollout_ref.actor.optim.lr` | 1e-6 | Actor 学习率 |
| `algorithm.kl_ctrl.kl_coef` | 0.001 | KL 惩罚系数 |
| `max_turns` | 10 | 每条 rollout 最多 10 轮工具调用 |

前置条件：

```bash
export PET_NODE_RANK=0
export VLLM_ATTENTION_BACKEND=XFORMERS
ray start --head

# 如果用 online_search 模式，另外需要启动搜索服务
python -m scrl.handler.server_handler   # Flask :5000
python -m scrl.handler.handler           # 轮询协调器
```

## 2. Rollout 阶段 — 多轮 Agent 循环

核心在 `scrl/llm_agent/generation.py` 的 `LLMGenerationManager`：

```
对每个 prompt（问题）:
  重复 n=16 次:
    messages = [system_prompt, user_question]
    for turn in range(max_turns=10):
      1. vLLM 生成 → 模型输出文本
      2. 解析输出:
         - 如果包含 <answer>...</answer> → 结束，记录答案
         - 如果包含 <tool_call>{"name":"web_search", ...}</tool_call>
           → 提取工具调用
      3. 执行工具调用（通过文件 IPC）
      4. 将工具结果作为 observation 追加到 messages
      5. 继续下一轮生成
```

模型输出格式：

```xml
<think>我需要搜索关于...</think>
<tool_call>{"name": "web_search", "arguments": {"query": ["..."]}} </tool_call>

<!-- 或者最终回答 -->
<think>根据搜索结果...</think>
<answer>最终答案</answer>
```

## 3. 文件 IPC — 训练节点 ↔ 搜索服务

训练和搜索分别运行在不同进程（可以跨机器），通过两个 JSON 文件通信：

```
训练进程                          搜索协调器 (handler.py)
─────────────                    ──────────────────────
生成 tool_call                   轮询 signal/signal.json
写 signal/data.json (查询)       ↓
写 signal/signal.json = {1}  ──► 读 signal/data.json
等待...                          分发到 server_handler.py
                                 Flask → WebSearchAgent / ReadingAgent
                                 写回 signal/data.json (结果)
读 signal/data.json ◄────────── 写 signal/signal.json = {0}
继续生成
```

`signal/data.json` 格式示例（训练写入）：

```json
[
  {"query": ["A股行情"], "action_type": "web_search"},
  {"url_list": ["https://..."], "action_type": "browse_webpage"}
]
```

搜索服务处理完后写回同一文件，内容替换为搜索/浏览结果。

## 4. Reward 计算

实现在 `verl/utils/reward_score/format_and_f1.py`：

```python
def compute_score(solution_str, ground_truth):
    # 1. 从模型输出中提取 <answer>...</answer>
    # 2. 如果格式不对（没有 answer 标签）→ return -1.0
    # 3. 计算 token-level F1:
    #    prediction_tokens = set(answer.split())
    #    ground_truth_tokens = set(truth.split())
    #    precision = |common| / |prediction|
    #    recall    = |common| / |ground_truth|
    #    f1 = 2 * precision * recall / (precision + recall)
    # 4. 返回 f1 分数 (0.0 ~ 1.0)
```

关键点：输出格式错误直接拿 -1.0，这迫使模型学会正确使用 `<answer>` 标签。

## 5. GRPO Advantage 计算

实现在 `verl/trainer/ppo/core_algos.py`。

GRPO 的核心思路 — 不需要 Critic 网络，直接在同一 prompt 的 16 条 rollout 内做归一化：

```python
# 对同一个 prompt 的 n=16 条 rollout:
rewards = [r1, r2, ..., r16]          # 每条 rollout 的 F1 分数
mean = mean(rewards)
std  = std(rewards)
advantages = [(r - mean) / (std + eps) for r in rewards]

# 如果某个 prompt 的 16 条 rollout 全拿 0 分 → advantage 全为 0
# 如果有高有低 → 高分的 advantage 为正，低分的为负
# 模型学会：做出好的搜索决策 → 更高的 F1 → 正 advantage → 被强化
```

这就是 **Group Relative Policy Optimization** — 相对于同组其他 rollout 的表现来计算优势。

## 6. Actor/Critic 更新

```
Actor Update (PPO clip):
  ratio = π_new(a|s) / π_old(a|s)
  loss = -min(ratio * advantage, clip(ratio, 1-ε, 1+ε) * advantage)
  + KL penalty: kl_coef * KL(π_new || π_ref)

Critic Update:
  loss = MSE(V(s), target_value)
```

KL 惩罚防止模型偏离初始 Qwen2.5-7B 太远，保持语言生成质量。

## 7. 数据格式

训练数据是 `.parquet` 文件，包含 6 个 QA 数据集：

| 数据集 | 类型 | 说明 |
|---|---|---|
| NaturalQuestions (NQ) | 单跳 | Google 搜索问题 |
| HotpotQA | 多跳 | 需要多步推理 |
| 2WikiMultihopQA | 多跳 | 跨维基百科推理 |
| MuSiQue | 多跳 | 组合式问题 |
| TriviaQA | 单跳 | 知识问答 |
| PopQA | 单跳 | 流行文化问答 |

每条数据包含 `question` 和 `ground_truth`（用于 F1 计算）。

## 8. Checkpoint 结构

```
./ckpts/{project_name}/{experiment_name}/
  ├── global_step_100/
  │   ├── actor/          # FSDP sharded weights
  │   ├── critic/
  │   └── optimizer/
  ├── global_step_200/
  └── ...
```

## 9. 关键文件索引

| 文件 | 作用 |
|---|---|
| `train_grpo.sh` | 训练入口脚本 |
| `verl/trainer/main_ppo.py` | Hydra 入口，组装 Ray workers |
| `verl/trainer/ppo/ray_trainer.py` | 训练主循环：rollout → reward → update |
| `verl/trainer/ppo/core_algos.py` | GRPO/GAE advantage 计算，KL 控制 |
| `verl/trainer/config/ppo_trainer.yaml` | 全部超参数 (Hydra 配置) |
| `verl/workers/fsdp_workers.py` | FSDP Actor/Critic/RewardModel workers |
| `verl/workers/rollout/vllm_rollout/` | vLLM 推理引擎 |
| `verl/utils/reward_score/format_and_f1.py` | Token-level F1 reward |
| `verl/protocol.py` | `DataProto` — 核心批量 tensor 数据结构 |
| `scrl/llm_agent/generation.py` | `LLMGenerationManager` — 多轮 agent 循环，tool call 解析 |
| `scrl/handler/handler.py` | 训练↔搜索桥接（文件 IPC） |
| `scrl/handler/server_handler.py` | Flask 搜索服务 |

## 总结

```
数据加载 (parquet) → 对每个 batch:
  ├─ 每个 prompt 生成 16 条多轮 rollout (vLLM + 工具调用)
  ├─ 提取 <answer>，算 F1 分数
  ├─ 组内归一化得到 GRPO advantage
  ├─ PPO clip loss 更新 Actor
  ├─ Value loss 更新 Critic
  └─ KL penalty 防止偏离
→ 保存 checkpoint → 下一个 step
```

核心 insight：模型通过**试错**学会搜索策略 — 16 条 rollout 中搜索效果好的被强化，差的被抑制。不需要人工标注搜索过程，只需要最终答案的 F1 分数作为信号。
