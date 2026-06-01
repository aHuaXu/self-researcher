# DeepResearcher 训练调试日志

训练目标：在 4 × V100 32GB 上跑通 `scripts/train/grpo_qwen2.5_7b.sh`（Qwen2.5-7B-Instruct GRPO 单模型 stage 1 训练）。

---

## 问题 1：Ray Worker 占用错误 GPU

**现象**  
脚本中设置了 `CUDA_VISIBLE_DEVICES=1,2,6,7`，但 Ray worker 仍占用到其他用户的 GPU。

**根因**  
`CUDA_VISIBLE_DEVICES` 在脚本里设置，但 `ray start --head` 是提前（在脚本外）手动运行的，worker 进程没有继承该变量。

**修复**  
把 `ray stop --force` 和 `ray start --head` 直接写入训练脚本，并在 `ray start` 之前 `export` 所有需要继承的环境变量：

```bash
export CUDA_VISIBLE_DEVICES=1,2,6,7
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
ray stop --force 2>/dev/null || true
ray start --head
```

---

## 问题 2：训练在 NCCL 初始化后卡死

**现象**  
打印 `NCCL version 2.20.5+cuda12.4` 后 hang 住，GPU 利用率极低。

**根因**  
`NCCL_IB_DISABLE=1` / `NCCL_P2P_DISABLE=1` 在脚本里设置，但在 `ray start` 之后，worker 没有继承。NCCL 尝试用 InfiniBand 通信（V100 机器只有 PCIe），导致死锁。

**修复**  
同问题 1，把这两个变量的 `export` 移到 `ray start` 之前（见上方脚本片段）。

---

## 问题 3：vLLM 抛出 `AssertionError: assert "factor" in rope_scaling`

**现象**  
vLLM 0.6.3 初始化 Qwen2.5-7B 时报错。

**根因**  
Qwen2.5 的 `config.json` 里 `rope_scaling = {"rope_type": "default", "rope_theta": ...}`，没有 `factor` 字段，但 vLLM 0.6.3 的断言要求非 mrope/yarn 类型必须有 `factor`。

**修复（两处）**  
1. 修改 `verl/third_party/vllm/vllm_v_0_6_3/config.py`，在 `super().__init__()` 前把 `rope_type == "default"` 的 `rope_scaling` 置 `None`，让 vLLM 回退到标准 `RotaryEmbedding`：
   ```python
   rope_scaling = getattr(hf_config, "rope_scaling", None)
   if rope_scaling is not None and rope_scaling.get("rope_type", "") == "default":
       hf_config.rope_scaling = None
   ```
2. 同时 patch 服务器上已安装的 vLLM `config.py`，在 `rope_type` 检查白名单中加入 `"default"`。

---

## 问题 4：`ValueError: Unknown RoPE scaling type default`

**现象**  
接问题 3，assert 修掉后 vLLM model builder 仍抛 `ValueError`。

**根因**  
`rotary_embedding.py` 不识别 `"default"` 类型。

**修复**  
问题 3 中直接把 `rope_scaling` 置 `None`，vLLM 不会再看到 `"default"` 字符串，问题自然消失。

---

## 问题 5：`ValueError: max seq len (32768) > KV cache capacity (16064)`

**现象**  
vLLM 初始化时报序列长度超出 KV cache 容量。

**根因**  
vLLM 默认 `max_model_len=32768`（Qwen2.5 的 RoPE 上限），但用 `gpu_memory_utilization=0.5` 在 V100 32GB 上只能分到 ~16064 个 token 的 KV cache。

**修复**  
在脚本中显式限制：
```bash
actor_rollout_ref.rollout.max_model_len=12240 \
actor_rollout_ref.rollout.max_num_batched_tokens=12240 \
```

---

## 问题 6：`AttributeError: 'int' object has no attribute 'item'`

**现象**  
`compute_grpo_outcome_advantage` 中 `index[i].item()` 报错。

**根因**  
`data.non_tensor_batch['agent_grpo_idx']` 是 numpy int 数组而非 torch Tensor，`.item()` 方法不存在。

**修复**  
`verl/trainer/ppo/core_algos.py` 中把所有 `index[i].item()` 改为 `int(index[i])`，兼容 tensor 和非-tensor 整数类型。

---

## 问题 7：optimizer step 时 CUDA OOM（74 MiB）

**现象**  
`_multi_tensor_adamw` 中 `torch._foreach_sqrt(device_exp_avg_sqs)` OOM，申请 74 MiB。

**背景**  
GPU 30.86 GB 已用，物理空闲 27 MB。`_foreach_sqrt` 一次性把**所有参数**的 `exp_avg_sq_sqrt` 打包成一个大块操作，需要一块 74 MiB 连续内存，分配器找不到。

**修复**  
`verl/workers/fsdp_workers.py` 中创建 AdamW 时加 `foreach=False`：
```python
actor_optimizer = optim.AdamW(..., foreach=False)
```
`foreach=False` 每次只处理一个参数的 `sqrt`（单个 shard ~16 MB），刚好能从 PyTorch 82 MB 预留池中找到，不需要向 CUDA 申请新物理内存。

---

## 问题 8：Ray OOM killer 杀死 Worker

**现象**  
batch 0 成功完成后，Ray 日志报 `1 Workers killed due to memory pressure (OOM)`，训练崩溃。

**根因**  
服务器 CPU 内存 252 GB 充裕（只用了 20 GB），但 Ray 的默认内存监控过于激进，误判为 OOM。

**修复**  
在 `ray start` 之前设置：
```bash
export RAY_memory_monitor_refresh_ms=0
```
禁用 Ray 的自动 worker 杀死机制。

---

## 问题 9：forward pass 中 `entropy_from_logits` OOM（1.15 GiB）

**现象**  
`entropy_from_logits` 的 `torch.sum(pd * logits, dim=-1)` 需分配 1.15 GB（`pd * logits` shape 为 `(16, 253, 151936)` bf16）。

**根因**  
Qwen2.5 词表 151,936，`pd * logits` 是 `(bsz, seq_len, vocab)` 全量中间 tensor。

**修复**  
`verl/utils/torch_functional.py` 中 `entropy_from_logits` 改为按 vocab 分 chunk 累加：
```python
def entropy_from_logits(logits, chunk_size=4096):
    log_z = torch.logsumexp(logits, dim=-1)
    entropy = log_z.clone()
    for i in range(0, logits.shape[-1], chunk_size):
        chunk = logits[..., i:i + chunk_size]
        p_chunk = torch.exp(chunk - log_z.unsqueeze(-1))
        entropy -= torch.sum(p_chunk * chunk, dim=-1)
    return entropy
```
每个 chunk 只需 `(bsz, seq, 4096)` ≈ 32 MB，peak 从 1.15 GB 降到 32 MB。

---

## 问题 10：backward pass 时 `torch.autograd.backward` OOM（588 MiB）

**现象**  
step 2 的 actor update backward 阶段 OOM，申请 588 MiB。

**根因**  
`logprobs_from_logits_v2` bf16 分支用 `F.log_softmax(row_logits)` 计算每行 log_prob，autograd 会保留**每一行**的 `(seq_len, vocab_size)` log_softmax tensor 直到 backward 结束。N 行 × 261 MB 同时占用显存，第 3 行开始分配时 OOM。

**修复**  
把 `F.log_softmax` 替换为 `logsumexp` 方式，不产生额外大 tensor：
```python
for row_logits, row_labels in zip(logits, labels):
    log_z = torch.logsumexp(row_logits, dim=-1)      # (seq_len,) — tiny
    label_logit = row_logits.gather(-1, row_labels.unsqueeze(-1)).squeeze(-1)
    logprobs_labels.append(label_logit - log_z)
```
`logsumexp` backward 对每行只临时申请一行的 softmax（261 MB），用完即释放，peak 从 N×261 MB 降到 1×261 MB。

---

## 问题 11：backward pass OOM（588 MiB）持续发生（step 2+）

**现象**  
Step 2 及之后每步 actor update backward 时 OOM，`torch.autograd.backward` 报 588 MiB 申请失败，GPU 0 只剩 ~535 MiB 可用。

**根因**  
Backward 需要计算 `d(loss)/d(logits)` 梯度 tensor，shape = `(micro_bsz, seq_len, vocab_size)` in bf16。每条序列约 261 MB，3 条序列 = 783 MB > 617 MB（可用上限）。  
`ppo_max_token_len_per_gpu=8192` 下每个 micro-batch 约有 8–9 条序列，backward 时梯度 tensor 超出显存。

**修复**  
将 `ppo_max_token_len_per_gpu` 从 8192 降到 900，强制每个 micro-batch 只有 1 条序列（~858 token）：
- `grad_logits` = 1 × 858 × 151,936 × 2 bytes = 261 MB < 617 MB ✓
- 代价：每个 actor update 多做 28 次 backward，训练速度有所下降，但不影响正确性（梯度累积等效）

---

## 问题 12：去掉 ref model 后 ref 推理仍在执行

**现象**  
设置 `use_kl_loss=false` 后，step metrics 里 `timing_s/ref` 仍为 123 s，说明 ref forward pass 还在跑。

**根因**  
`main_ppo.py` 中 `Role.RefPolicy` 无条件写死在 `role_worker_mapping` 里，`ray_trainer.py` 通过 `use_reference_policy = Role.RefPolicy in role_worker_mapping` 判断是否运行 ref，与 `use_kl_loss` 无关。

**修复**  
在 `verl/trainer/main_ppo.py` 中根据 `use_kl_loss` 条件性地注册 RefPolicy：
```python
use_ref_policy = config.actor_rollout_ref.actor.get('use_kl_loss', True)
role_worker_mapping = {Role.ActorRollout: ..., Role.Critic: ...}
if use_ref_policy:
    role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
    mapping[Role.RefPolicy] = global_pool_id
```
同时在脚本中同步去掉 ref 相关参数（`actor_rollout_ref.ref.*`、`algorithm.kl_ctrl.kl_coef`）。

**效果**  
每 step 从 ~14 min 降到 ~11 min（节省 20%），显存压力同步减轻。

---

## 问题 13：step 2+ 持续 CUDA OOM（446 MiB），gradient checkpointing 无效

**现象**  
step 1 正常完成，step 2 的 `loss.backward()` 必现 OOM：
```
Tried to allocate 446.00 MiB. GPU 0 has 31.73 GiB total,
30.45 GiB allocated by PyTorch, 371 MiB free.
```
开启 `enable_gradient_checkpointing=true` 后 30.45 GB 未减少，问题依旧。

**定位过程**  
在 `update_policy` 开头加内存快照 log，对比两步差异：

| 时间点 | Step 1 | Step 2 |
|---|---|---|
| `update_policy START`（`empty_cache()` 后） | 7.63 GB | **22.87 GB** |
| `before backward` | 17.85 GB | 25.46 GB |

Step 1 的 `update_policy START` 只有 7.63 GB——optimizer 刚初始化，`optimizer.state` 为空，`load_fsdp_optimizer()` 直接 `return`（是 no-op）。

Step 2 开始时是 22.87 GB，多出了 15.24 GB = fp32 AdamW 的 momentum + velocity（7B × 2 × 4 bytes / 4 GPUs ≈ 14 GB），因为 `fsdp_workers.py` 的 `update_actor()` 在**最开始**就调用了 `load_fsdp_optimizer()`，把全部优化器状态加载到 GPU，然后整个 forward+backward 过程中这 14 GB 一直占着。

**根因**  
`fsdp_workers.py update_actor()` 的加载顺序错误：
```python
# ❌ 加载放在最开始，forward+backward 全程占 14 GB
load_fsdp_model_to_gpu(...)
load_fsdp_optimizer(...)     # 14 GB，但 forward/backward 根本不需要它
...
update_policy(...)           # OOM
```
`optimizer.state` 只在 `optimizer.step()` 时才被使用，在 forward+backward 期间完全不需要在 GPU 上。

**修复：JIT optimizer load/offload**  
把 optimizer 的 load/offload 从 `update_actor()` 前置移到 `_optimizer_step()` 内部，仅在 `optimizer.step()` 的前后加载/卸载：

`verl/workers/fsdp_workers.py`：
```python
# 不再提前 load optimizer，改为注入回调
if self._is_offload_optimizer:
    self.actor._optimizer_load_fn = lambda: load_fsdp_optimizer(self.actor_optimizer, device_id)
    self.actor._optimizer_offload_fn = lambda: offload_fsdp_optimizer(self.actor_optimizer)
```

`verl/workers/actor/dp_actor.py`：
```python
def _optimizer_step(self):
    if getattr(self, '_optimizer_load_fn', None) is not None:
        self._optimizer_load_fn()          # 临时加载 14 GB optimizer
    ... clip_grad_norm_ ...
    self.actor_optimizer.step()
    if getattr(self, '_optimizer_offload_fn', None) is not None:
        self._optimizer_offload_fn()       # 立即卸载回 CPU
    return grad_norm
```

**效果**  
- Step 2 `update_policy START` 从 22.87 GB 降回 7.63 GB（与 step 1 相同）
- forward+backward 期间峰值内存控制在 ~17 GB，不再 OOM
- `optimizer.step()` 期间短暂有 14 GB optimizer 在 GPU（peak ~21 GB），远低于 31.73 GB 上限

---

## 环境变量汇总（需在 `ray start` 前 export）

```bash
export CUDA_VISIBLE_DEVICES=1,2,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export RAY_memory_monitor_refresh_ms=0
```

## 代码修改汇总

| 文件 | 修改内容 |
|---|---|
| `verl/third_party/vllm/vllm_v_0_6_3/config.py` | strip `rope_scaling` when `rope_type == "default"` |
| `verl/trainer/ppo/core_algos.py` | `index[i].item()` → `int(index[i])` |
| `verl/workers/fsdp_workers.py` | AdamW 加 `foreach=False`；`update_actor` 不再提前加载 optimizer，改为注入 JIT 回调 |
| `verl/utils/torch_functional.py` | `entropy_from_logits` 改为 chunked vocab 累加；`logprobs_from_logits_v2` bf16 分支改为 logsumexp 方式 |
| `verl/workers/actor/dp_actor.py` | `update_policy` 开头加 `empty_cache()`；`_optimizer_step` 支持 JIT load/offload 回调 |
| `verl/trainer/main_ppo.py` | `Role.RefPolicy` 按 `use_kl_loss` 条件注册，`false` 时彻底跳过 ref worker |
| `scripts/train/grpo_qwen2.5_7b.sh` | 集成 ray stop/start；加各环境变量；限制 `max_model_len`；`gpu_memory_utilization=0.10`；`enable_gradient_checkpointing=true` |

---

## Dual-Agent Smoke Test（2026-05-30）

详见 `docs/design/dual_agent_smoke_test.md`。摘要如下。

### 问题 14：CUDA 驱动 / nvidia-uvm 损坏

**现象**：`torch.cuda.is_available()` 为 False，`/dev/nvidia-uvm` I/O error  
**修复**：需管理员 `sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm`

### 问题 15：cachetools 与 vLLM LoRA 不兼容

**现象**：`AttributeError: 'LoRALRUCache' object has no attribute '_LRUCache__update'`  
**修复**：`pip install cachetools==5.5.2`

### 问题 16：Multi-agent reward KeyError

**现象**：rollout 完成后 `KeyError: 'reward_model'`  
**修复**：`ray_trainer.py` 从 `batch` 而非 `gen_batch` 读 golden answer

### 问题 17：LoRA hybrid engine Planner OOM

**现象**：64 条 Planner prompt 同时 generate，GPU 31GB 占满  
**修复**：smoke test 降 batch；`fsdp_vllm.py` state_dict offload + FSDP 及时 offload

### 问题 18：Planner 输出全 `!` 乱码

**现象**：rollout JSON 中 planner output 为重复 `!`  
**原因**：vLLM 权重 offload 用 `empty_like` 未初始化  
**修复**：`vllm_rollout_spmd.py` / `fsdp_vllm.py` 改用 `clone()`

### 问题 19：Executor 多轮 generate CPU tensor 错误

**现象**：Planner 成功后 Executor 第一次 tool turn 报 `was on cpu`  
**修复**：`__enter__` 中 load_dtensor 前将 vLLM params `.cuda()`；rollout 前仍 `load_fsdp_model_to_gpu`

### 问题 20：Executor 多 wave DataProto.concat 序列长度不一致

**现象**：global step 1 Executor DAG 完成后 crash  
`RuntimeError: Expected size 647 but got size 800 for tensor number 1 in the list`  
**堆栈**：`multi_agent_generation._run_executor_dag` → `DataProto.concat(all_exec_output_parts)`

**原因**：各 DAG wave 独立 `run_llm_loop`，prompt/response 长度不同，batch tensor dim=1 不一致无法 concat。

**修复**：concat 前按 key 将各 wave 输出 pad 到统一 max seq len（见 `multi_agent_generation._pad_dataproto_for_concat`）。

### 问题 21：SearXNG 未启动

**现象**：Executor search `Connection refused` on `localhost:8888`  
**原因**：`.env` 配置 `SEARCH_ENGINE=searxng`，但 SearXNG 进程未运行（与 Hydra `search_engine=online_search` 无关）  
**修复**：`bash /home/zjx/self_llm/searxng/start.sh`，验证 HTTP 200

### 问题 22：Executor batch 无法被 4 GPU 整除

**现象**：`executor_step_1.json` + `DualAgentReward` 后 crash  
`AssertionError: only support equal chunk. Got size of DataProto 31 and chunk 4`  
**修复**：`ray_trainer.py` multi-agent 路径 update 前 `pad_dataproto_to_divisor`

## Dual-Agent 代码修改汇总

| 文件 | 修改内容 |
|---|---|
| `verl/trainer/ppo/ray_trainer.py` | multi-agent golden answer 从 `batch` 读取并 repeat |
| `verl/workers/sharding_manager/fsdp_vllm.py` | LoRA state_dict offload；FSDP offload；权重 clone；vLLM params cuda |
| `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` | init offload 用 clone |
| `verl/workers/fsdp_workers.py` | lora_config 增加 base_model |
| `scrl/llm_agent/multi_agent_generation.py` | Executor 多 wave concat 前 pad 序列长度 |
| `scripts/train/grpo_dual_agent.sh` | GPU 0-3；total_training_steps=2；降 batch |
| `docs/design/dual_agent_smoke_test.md` | smoke test 设计与变更记录 |
