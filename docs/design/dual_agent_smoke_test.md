# Dual-Agent GRPO Smoke Test Plan

## 目标

跑通 `scripts/train/grpo_dual_agent.sh` 多 Agent 训练流程，完成 **2 个 training step** 后自动停止。

## 验证标准

1. 训练无 crash，完成 `global_step` 1 和 2
2. Planner rollout：输出含 `<plan>`，3–5 个子问题，格式可读
3. Executor rollout：正常 tool call / `<answer>`，无乱码
4. Reward / loss 数值合理（非 NaN）

## Smoke Test 配置变更

| 参数 | 值 | 说明 |
|------|-----|------|
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` | 使用完全空闲的 GPU |
| `trainer.total_training_steps` | `2` | 现成 Hydra 参数，跑 2 step 后自动退出 |
| `data.train_batch_size` | `4` | 降低 batch，避免 Planner 阶段 OOM |
| `agent_grpo.n` | `2` | 降低 GRPO 组数（4×2=8 条 rollout） |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | `0.4` | 给 FSDP actor 留更多显存 |

完整训练时：移除 `trainer.total_training_steps=2`（或设为 null），恢复 batch / GRPO 规模。

## 环境依赖

### cachetools 版本

vLLM 0.8.x + LoRA 时若报 `LoRALRUCache ... _LRUCache__update`：

```bash
pip install cachetools==5.5.2
```

原因：`cachetools` 6.x/7.x 移除了 vLLM LRUCache 依赖的内部方法。

### 服务器前置条件

- CUDA 可用（`/dev/nvidia-uvm` 不能 I/O error）
- 磁盘空间充足（曾清理旧 FSDP ckpt 释放 ~94GB）
- GPU 0–3 空闲

## 代码修改记录

### 1. `verl/trainer/ppo/ray_trainer.py` — golden answer 读取

**现象**：Planner + Executor rollout 完成后 reward 阶段 crash  
`KeyError: 'reward_model'`

**原因**：`gen_batch = batch.pop(...)` 只保留 `raw_prompt_ids`，`reward_model` 留在 `batch` 里；multi-agent 分支误从 `gen_batch` 读 golden answer。

**修复**：从 `batch.non_tensor_batch['reward_model']` 读取，并按 `grpo_n` 用 `np.repeat` 对齐 rollout 数量。

---

### 2. `verl/workers/sharding_manager/fsdp_vllm.py` — LoRA hybrid 显存

**现象 A**：Planner 第一次 generate OOM，GPU 已占 ~31GB  
**原因**：FSDP 全量 `state_dict()` + vLLM 权重同时在 GPU。

**修复 A**：
- LoRA 场景 `FullStateDictConfig(offload_to_cpu=True, rank0_only=True)`
- `state_dict()` 后立即 `offload_fsdp_model_to_cpu()`，再 sync 到 vLLM

**现象 B**：Planner 输出全是 `!` 乱码  
**原因**：vLLM init 时用 `torch.empty_like(..., device='cpu')` offload 权重，未初始化内存；若 FSDP sync 不完整则 vLLM 用垃圾权重推理。

**修复 B**：init offload 和 `__exit__` CPU backup 均改为 `param.data.detach().cpu().clone()`。

**现象 C**：Executor 第二次 generate 报 `Expects tensor on cuda:0, was on cpu`  
**原因**：`__exit__` 后 vLLM 权重在 CPU；再次 `__enter__` 时部分 param 仍在 CPU。

**修复 C**：`load_dtensor_weights` 前把 vLLM model 参数 `.cuda()`。

**现象 D**：Executor 阶段 `state_dict()` 报 CPU tensor 错误（曾尝试跳过 load）  
**原因**：上一轮 rollout 后 FSDP 已 offload 到 CPU，不 load 回 GPU 则 `state_dict()` 失败。

**结论**：`fsdp_workers.generate_sequences` **仍需**在 rollout 前 `load_fsdp_model_to_gpu`；OOM 靠 `__enter__` 内 state_dict 后立即 offload FSDP 解决。

**附加**：LoRA `adapter_config.json` 增加 `base_model_name_or_path`；`lora_config` 传入 `base_model` 路径。

---

### 3. `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` — 权重 offload

**修改**：init 时 vLLM 权重 offload 到 CPU 用 `clone()` 替代 `empty_like`（同修复 B）。

---

### 4. `verl/workers/fsdp_workers.py` — LoRA config

**修改**：构建 `FSDPVLLMShardingManager` 时 `lora_config` 增加 `base_model` 字段。

---

### 5. `scripts/train/grpo_dual_agent.sh` — smoke test 参数

**修改**：
- `CUDA_VISIBLE_DEVICES=0,1,2,3`
- `trainer.total_training_steps=2`
- `train_batch_size=4`, `agent_grpo.n=2`, `gpu_memory_utilization=0.4`

---

### 6. `verl/models/registry.py` — 服务器代码同步

**现象**：`ImportError: cannot import name 'check_model_support_rmpad'`  
**原因**：服务器 `verl/models/registry.py` 版本落后于本地。  
**修复**：rsync 同步完整代码（含 `check_model_support_rmpad` 与 `qwen3` 支持）。

---

### 7. `scrl/llm_agent/multi_agent_generation.py` — Executor 多 wave concat 序列长度不一致

**现象**：global step 1 Executor DAG 最后一轮结束后 crash  
`RuntimeError: Sizes of tensors must match except in dimension 0. Expected size 647 but got size 800`

**堆栈**：`_run_executor_dag` → `DataProto.concat(all_exec_output_parts)`

**原因**：Executor 按 DAG wave 分批 rollout，每个 wave 的 prompt/response 长度不同（独立 sub-question、不同 tool turn 数）。`run_llm_loop` 返回的 `input_ids`/`prompts`/`responses` 等 tensor 在 dim=1 上长度各异，直接 `torch.cat(dim=0)` 失败。

**修复**：`DataProto.concat` 前，对所有 wave 的 batch tensor 按 key 分别 pad 到该 key 的最大 seq len（`input_ids`/`prompts`/`responses` 用 `pad_token_id`，`attention_mask`/`position_ids` 用 0）。

---

### 8. SearXNG 未启动导致 Executor 搜索失败

**现象**：Executor web search 大量 `SearXNG localhost:8888 Connection refused`

**原因**：
- `.env` 中 `SEARCH_ENGINE=searxng`、`SEARXNG_URL=http://localhost:8888`（项目原本就用 SearXNG）
- 训练脚本里的 `search_engine=online_search` 只控制 tool/prompt 模式，**不**决定实际搜索后端
- SearXNG 进程未运行（8888 未监听），上次日志停留在 2026-05-16

**修复**：启动本地 SearXNG
```bash
bash /home/zjx/ahua_llm/searxng/start.sh
curl -s -o /dev/null -w '%{http_code}\n' 'http://127.0.0.1:8888/search?q=test&format=json'  # 期望 200
```
- 安装目录：`/home/zjx/ahua_llm/searxng/`
- 绑定：`127.0.0.1:8888`
- 2026-05-31 已启动，PID 2071025，搜索 API 验证通过

---

### 9. `verl/trainer/ppo/ray_trainer.py` — Executor batch 无法被 world_size 整除

**现象**：step 1 rollout + reward 完成后 update 阶段 crash  
`AssertionError: only support equal chunk. Got size of DataProto 31 and chunk 4.`

**堆栈**：`compute_log_prob(agent_output)` → `DataProto.chunk(chunks=world_size)`

**原因**：Executor 按 DAG TODO 数产出 batch（本 run 31 条），4 GPU 数据并行要求 batch 能被 `world_size` 整除。

**修复**：multi-agent update 前对 `agent_output` 调用已有工具 `pad_dataproto_to_divisor(agent_output, world_size)`（与 validation generate 路径一致），再 `compute_log_prob` / `update_actor`。

## 执行步骤

1. 本地更新代码 + 本文档 → rsync 到服务器
2. 训练前检查 GPU 0–3 空闲、CUDA 可用
3. 确认 `cachetools==5.5.2`
4. **启动 SearXNG**：`bash /home/zjx/ahua_llm/searxng/start.sh`（`.env` 使用 `SEARCH_ENGINE=searxng` 时必需）
5. 启动 `bash scripts/train/grpo_dual_agent.sh`
5. 每 30s 监控 `dual_agent_smoke.log`，异常即停
6. 检查 `outputs/deepresearcher/qwen3_4b_dual_agent_ws4/rollout/` 下 planner/executor JSON

## 日志位置

- 训练 stdout：`/home/zjx/ahua_llm/self-researcher/deepresearcher_qwen3_4b_dual_agent_ws4.log`
- smoke 监控：`/home/zjx/ahua_llm/self-researcher/dual_agent_smoke.log`
- Rollout JSON：`outputs/deepresearcher/qwen3_4b_dual_agent_ws4/rollout/`
