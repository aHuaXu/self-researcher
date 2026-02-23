# Multi-Agent LoRA GRPO Training — Design Spec

## 概述

在 verl 框架内扩展，实现三 Agent（Planner / Executor / Writer）的 GRPO 联合训练。共享 DeepResearcher-7B 基座模型，每个 Agent 一个独立的 LoRA adapter，通过配置开关与原有单 agent 训练完全隔离。

---

## 1. 架构总览

```
DeepResearcher-7B (冻结基座)
  ├── LoRA_planner   (r=64)
  ├── LoRA_executor  (r=64)
  └── LoRA_writer    (r=64)

一个 vLLM 实例，通过 LoRARequest 动态切换 adapter
```

### 训练 Step 流程

```
┌─────────────────────────────────────────────────────────┐
│                 一个 Training Step                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. 数据加载                                             │
│     RLHFDataset 从 multi-research.parquet 读取         │
│     一个 batch = N 条研究问题                             │
│                                                         │
│  2. Rollout (每条 query × 16 条)                        │
│     对每条 rollout:                                      │
│       LoRA_p → Planner 生成 TODO list                   │
│       LoRA_e → Executor 多轮搜索 → findings             │
│       LoRA_w → Writer 生成报告                           │
│     记录三段 token 序列 + log_probs                      │
│                                                         │
│  3. Reward                                              │
│     外部 LLM Judge 异步并发给报告打分 → final_reward     │
│     规则打分 → rule_p, rule_e                            │
│     加权组合 → reward_p, reward_e, reward_w              │
│                                                         │
│  4. Advantage (三次独立 GRPO)                           │
│     对每个 agent:                                       │
│       同一 query 的 16 条 rollout 组内归一化              │
│       → planner_advantage                               │
│       → executor_advantage                              │
│       → writer_advantage                                │
│                                                         │
│  5. Update (三次独立 PPO clip)                          │
│     冻结基座，每次只开启一个 LoRA 的梯度                   │
│     update LoRA_p with planner_advantage                │
│     update LoRA_e with executor_advantage               │
│     update LoRA_w with writer_advantage                 │
│                                                         │
│  6. Checkpoint                                          │
│     只保存三个 LoRA adapter（不保存基座）                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 配置

### 入口脚本

```bash
# 原有单 agent 训练（不变）
bash train_grpo.sh

# 新增多 agent 训练（独立脚本）
bash train_multi_agent.sh
```

共享 `verl.trainer.main_ppo` 入口，通过 Hydra config 区分。

### 新增配置段

```yaml
multi_agent:
  enable: true                           # false 时走原有单 agent 逻辑
  base_model: GAIR/DeepResearcher-7b

  lora:
    rank: 64
    alpha: 128
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
    dropout: 0.05

  agents:                                 # system_prompt 复用 research_agent/prompts/
    planner:
      max_tokens: 1024                   # plan 不需要太长
    executor:
      max_turns: 10                      # 多轮工具调用
    writer:
      max_tokens: 4096                   # 报告需要较长输出

  reward:
    judge_model: "claude-opus-4-7"       # 外部 LLM Judge
    judge_base_url: "https://..."
    judge_api_key: "sk-xxx"
    judge_max_concurrent: 50             # 异步并发数
    alpha: 0.2                           # planner 规则 reward 权重
    beta: 0.3                            # executor 规则 reward 权重
```

`multi_agent.enable: false`（默认）时，所有新增代码路径不走，原有训练完全不受影响。

---

## 3. 数据

### 格式

复用 `RLHFDataset`，不改加载代码。只是 parquet 内容不同：

```python
# 研究问题数据（无 ground_truth）
{
    "data_source": "research_query",
    "prompt": [{"role": "user", "content": "分析2024年全球AI芯片市场格局"}],
    "reward_model": {},
    "extra_info": {"domain": "tech", "index": "rq_00001"}
}
```

### 数据来源

- LLM 批量生成不同领域的研究问题
- 知乎/Reddit 等平台的研究类问题
- 学术论文的 research question
- 金融/行业研究报告标题

### 文件位置

```
data/multi-research.parquet           # 训练集
data/multi-research_dev.parquet       # 验证集
```

`train_multi_agent.sh` 里指定：`data.train_files=./data/multi-research.parquet`

---

## 4. Rollout — 三阶段 LoRA 切换

### 实现方式

新增 `MultiAgentGenerationManager`，继承 `LLMGenerationManager`：

```python
class MultiAgentGenerationManager(LLMGenerationManager):
    """三阶段串行 rollout，复用父类的底层生成和工具调用逻辑"""

    def run_multi_agent_loop(self, prompts):

        # Stage 1 — Planner（单轮生成）
        plan_outputs = self._generate_with_gpu_padding(
            prompts, lora_request=self.lora_planner
        )
        # 解析 TODO list → 每条 prompt 得到 N 个子任务

        # Stage 2 — Executor（每个 TODO 独立执行，组成 batch 并行）
        # 例: 2 条 prompt × 各 5 个 TODO = 10 条 executor prompt
        exec_prompts = build_executor_prompts(prompts, plan_outputs)  # 展开为子任务 batch
        exec_outputs = self.run_llm_loop(
            exec_prompts, lora_request=self.lora_executor
        )
        # 按原始 prompt 聚合各子任务的 findings
        grouped_findings = group_findings_by_prompt(exec_outputs, plan_outputs)

        # Stage 3 — Writer（单轮生成，汇总所有 findings）
        writer_prompts = build_writer_prompts(prompts, plan_outputs, grouped_findings)
        writer_outputs = self._generate_with_gpu_padding(
            writer_prompts, lora_request=self.lora_writer
        )

        return plan_outputs, exec_outputs, writer_outputs
```

### vLLM LoRA 切换

vLLM 原生支持 `LoRARequest`，初始化时启用：

```python
llm = LLM(
    model="GAIR/DeepResearcher-7b",
    enable_lora=True,
    max_lora_rank=64,
)

# 生成时指定用哪个 LoRA
output = llm.generate(prompts, sampling_params, lora_request=LoRARequest("planner", 1, lora_path))
```

切换只替换小权重（~200MB），不需要重载基座。

### verl 封装层改动

`_generate_with_gpu_padding()` 和 `run_llm_loop()` 加一个可选的 `lora_request` 参数，透传给 vLLM：

```python
# 改前
def _generate_with_gpu_padding(self, prompts, ...):
    return self.vllm_engine.generate(prompts, sampling_params)

# 改后
def _generate_with_gpu_padding(self, prompts, ..., lora_request=None):
    return self.vllm_engine.generate(prompts, sampling_params, lora_request=lora_request)
```

原有单 agent 调用不传 `lora_request`（默认 None），行为不变。

### Executor 工具调用

直接调用 `research_agent/tools/` → `scrl/handler/`，和研究助手一致：

```
Executor 生成 <tool_call>
  → 解析出 web_search / browse_webpage
  → research_agent/tools/search.py 或 browse.py
  → scrl/handler/WebSearchAgent / ReadingAgent
  → 结果作为 observation 追加到 messages
```

### Rollout 产出

每条 rollout 产出三段独立的 token 序列 + log_probs（vLLM 生成时自带）：

```python
{
    "planner":  {"output_ids": [...], "log_probs": [...]},
    "executor": {"output_ids": [...], "log_probs": [...]},
    "writer":   {"output_ids": [...], "log_probs": [...]},
    "final_report": "..."   # writer 输出文本，给 Judge 打分
}
```

---

## 5. Reward

### 结构

```
MultiAgentRewardManager:
  1. LLM Judge → final_reward (0-1)
  2. 规则打分 → rule_p, rule_e (0-1)
  3. 加权组合:
     reward_w = final_reward
     reward_e = β · rule_e + (1-β) · final_reward
     reward_p = α · rule_p + (1-α) · final_reward
```

### LLM Judge

外部 API + 异步并发：

```python
class LLMJudge:
    def __init__(self, model, base_url, api_key, max_concurrent=50):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def score_batch(self, queries, reports) -> list[float]:
        tasks = [self._score_one(q, r) for q, r in zip(queries, reports)]
        return await asyncio.gather(*tasks)

    async def _score_one(self, query, report):
        async with self.semaphore:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(...)}],
                max_tokens=16,
            )
            return parse_score(response) / 10.0
```

Judge prompt 要求输出 1-10 分，评估准确性、完整性、结构性、可读性。

一个 batch（256 query × 16 rollout = 4096 次调用），50 并发约 4-7 分钟。

### Planner 规则 reward (rule_p)

| 规则 | 分值 | 说明 |
|------|------|------|
| 子任务数量在 3-7 之间 | +1 | 覆盖度 |
| 子任务无明显重复（关键词重合率 < 阈值） | +1 | 避免浪费 |
| 优先级分布合理（不全是同一级别） | +0.5 | 有主次 |

归一化：rule_p = 得分 / 2.5

### Executor 规则 reward (rule_e)

| 规则 | 分值 | 说明 |
|------|------|------|
| 成功调用搜索且返回非空 | +1 | 基本搜索能力 |
| 调用了 browse（深入阅读） | +1 | 不只看摘要 |
| 返回内容非空且长度合理 | +1 | 有效信息 |
| 没有触发 max_turns 限制 | +1 | 效率 |

归一化：rule_e = 得分 / 4

### α/β 退火

初始 α=0.2, β=0.3。可随训练进度 anneal 到 0，让模型后期更多优化最终报告质量。

---

## 6. Advantage — 三次独立 GRPO

复用现有 `compute_grpo_outcome_advantage()` 函数，调三次：

```python
# 同一个函数，传不同 agent 的 reward
adv_p = compute_grpo_outcome_advantage(planner_rewards, prompt_indices)
adv_e = compute_grpo_outcome_advantage(executor_rewards, prompt_indices)
adv_w = compute_grpo_outcome_advantage(writer_rewards, prompt_indices)
```

每次调用内部逻辑不变：同一 prompt 的 16 条 rollout 组内归一化。函数本身不改。

---

## 7. Update — 分别更新三个 LoRA

### PEFT 初始化

在 `fsdp_workers.py` 的 `_build_model_optimizer()` 中：

```python
if config.multi_agent.enable:
    from peft import get_peft_model, LoraConfig

    lora_config = LoraConfig(
        r=config.multi_agent.lora.rank,
        lora_alpha=config.multi_agent.lora.alpha,
        target_modules=config.multi_agent.lora.target_modules,
        lora_dropout=config.multi_agent.lora.dropout,
    )
    # 基座冻结，三个 LoRA adapter
    model = get_peft_model(model, lora_config, adapter_name="planner")
    model.add_adapter("executor", lora_config)
    model.add_adapter("writer", lora_config)

    # FSDP wrap with is_lora=True（已有支持）
    wrap_policy = get_fsdp_wrap_policy(model, is_lora=True)
```

### 参数隔离更新

每次 update 只开启目标 LoRA 的梯度：

```python
def update_policy(self, trajectory, advantage, old_log_probs, lora_name):
    # 1. 切换到目标 adapter
    self.model.set_adapter(lora_name)

    # 2. 只开启目标 LoRA 的梯度（基座始终冻结）
    for name, param in self.model.named_parameters():
        if "lora_" in name:
            param.requires_grad = (lora_name in name)
        # 基座参数始终 requires_grad=False，不动

    # 3. PPO clip loss → backward → step
    loss = ppo_clip_loss(trajectory, advantage, old_log_probs)
    loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()
```

### 训练主循环

`ray_trainer.py` 的 `fit()` 中新增分支：

```python
if config.multi_agent.enable:
    for agent in ["planner", "executor", "writer"]:
        advantage = compute_grpo_outcome_advantage(rewards[agent], indices)
        actor.update_policy(
            trajectories[agent], advantage,
            old_log_probs[agent], lora_name=agent
        )
else:
    # 原有单 agent 逻辑（不变）
    advantage = compute_grpo_outcome_advantage(rewards, indices)
    actor.update_policy(trajectories, advantage, old_log_probs)
```

### Checkpoint

只保存 LoRA 权重：

```
ckpts/{experiment_name}/global_step_N/
  ├── lora_planner/
  │   ├── adapter_config.json
  │   └── adapter_model.bin      # ~200MB
  ├── lora_executor/
  │   └── ...
  └── lora_writer/
      └── ...
```

---

## 8. 文件改动清单

### 新增文件

| 文件 | 内容 |
|------|------|
| `train_multi_agent.sh` | 多 agent 训练入口脚本 |
| `scrl/llm_agent/multi_agent_generation.py` | `MultiAgentGenerationManager` |
| `verl/workers/reward_manager/multi_agent.py` | `MultiAgentRewardManager` |
| `verl/utils/reward_score/llm_judge.py` | `LLMJudge`（异步外部 API） |
| `verl/utils/reward_score/rule_reward.py` | `planner_rules()` + `executor_rules()` |
| `data/multi-research.parquet` | 训练数据（研究问题） |

### 改动文件（小改，加配置开关或参数透传）

| 文件 | 改动 |
|------|------|
| `verl/trainer/config/ppo_trainer.yaml` | 新增 `multi_agent` 配置段 |
| `verl/workers/rollout/vllm_rollout/vllm_rollout.py` | `__init__` 加 `enable_lora`；`generate_sequences` 加 `lora_request` 透传 |
| `scrl/llm_agent/generation.py` | `_generate_with_gpu_padding()` 和 `run_llm_loop()` 加 `lora_request` 参数 |
| `verl/workers/fsdp_workers.py` | `_build_model_optimizer()` 加 PEFT LoRA 初始化分支 |
| `verl/workers/actor/dp_actor.py` | `update_policy()` 加 `lora_name` 参数做参数隔离 |
| `verl/trainer/ppo/ray_trainer.py` | `fit()` 加 multi_agent 分支（三次 advantage + update） |

### 不改的文件

| 文件 | 原因 |
|------|------|
| `verl/trainer/ppo/core_algos.py` | `compute_grpo_outcome_advantage()` 不变，调用方调三次 |
| `verl/utils/dataset/rl_dataset.py` | `RLHFDataset` 不变，数据格式兼容 |
| `verl/workers/reward_manager/naive.py` | `NaiveRewardManager` 不变，单 agent 训练继续用 |
| `research_agent/` | 研究助手代码不变，工具层已共享 |

---

## 9. 资源估算

### GPU 分配（8×A100 80GB）

| 用途 | GPU | 显存 |
|------|-----|------|
| 基座模型 (FSDP sharded) | 8 卡共享 | ~14GB/卡 |
| 3 个 LoRA adapter | 8 卡共享 | ~600MB 总计 |
| vLLM rollout（KV cache） | 8 卡共享 | ~14GB/卡 |
| Optimizer states | 8 卡共享 | ~4GB/卡 |
| LLM Judge | 外部 API | 0 |
| **总计** | | ~32GB/卡，可行 |

### 训练时间估算（单 step）

| 阶段 | 耗时 |
|------|------|
| Rollout（三阶段 × 16 rollout × 工具调用） | ~15-25 分钟 |
| Reward（4096 次 Judge 调用，50 并发） | ~4-7 分钟 |
| Advantage + Update | ~2-3 分钟 |
| **总计** | ~20-35 分钟/step |

---

## 10. 验证方式

1. **单 agent 不受影响**：`multi_agent.enable=false` 时 `bash train_grpo.sh` 正常运行
2. **LoRA 切换正确**：三个阶段生成的 token 分布不同（不是纯基座输出）
3. **Reward 合理**：Judge 分数和人工评估一致
4. **三个 LoRA 独立更新**：检查 gradient norm，只有目标 LoRA 有梯度
5. **端到端**：训练后把三个 LoRA 加载到研究助手，报告质量提升
