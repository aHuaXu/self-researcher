# IGPO Port Notes — 优势/Belief 签名与张量表示

> 作者: Task 0 自动生成  
> 日期: 2026-06-20  
> IGPO 来源: git clone --depth 1 https://github.com/GuoqingWang1/IGPO /tmp/IGPO_ref (clone 成功)

---

## 1. IGPO `compute_grpo_outcome_advantage` — 完整签名

文件: `/tmp/IGPO_ref/verl/trainer/ppo/core_algos.py` 第 189 行

```python
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,   # (bs, response_length) 每 token 的即时奖励
    response_mask: torch.Tensor,          # (bs, response_length) 响应序列掩码
    index: np.ndarray,                    # prompt 分组索引，用于 group normalize
    epsilon: float = 1e-6,               # 防止除零的小常数
    norm_adv_by_std_in_grpo: bool = True,# 是否除以标准差
    gamma: float = 1.0,                  # turn-level 折扣因子，默认 1.0
    info_gain_norm_mode: str = "joint",  # 归一化模式，取值: "joint" | "separate"
    curriculum_f1_weight: float = 1.0,   # F1 reward 的课程权重，默认 1.0
    curriculum_ig_weight: float = 1.0,   # InfoGain reward 的课程权重，默认 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:  # (advantages, returns)，shape 均为 (bs, response_length)
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `token_level_rewards` | `torch.Tensor (bs, L)` | — | 多 turn token 级奖励；最后一个 token 位置放 F1；中间 turn end 位置放 IG 奖励 |
| `response_mask` | `torch.Tensor (bs, L)` | — | 注意：IGPO 用 `response_mask`，本仓库用 `eos_mask`，**名字不同但语义相同** |
| `index` | `np.ndarray` | — | prompt 分组 id，用于 group statistics |
| `epsilon` | `float` | `1e-6` | 防除零 |
| `norm_adv_by_std_in_grpo` | `bool` | `True` | 是否用 std 归一化 |
| `gamma` | `float` | `1.0` | turn 折扣因子（与本仓库 vanilla GRPO 不同，本仓库无此参数） |
| `info_gain_norm_mode` | `str` | `"joint"` | `"joint"`: F1+IG 联合归一化；`"separate"`: 分别归一化 |
| `curriculum_f1_weight` | `float` | `1.0` | 课程学习时调节 F1 奖励比重 |
| `curriculum_ig_weight` | `float` | `1.0` | 课程学习时调节 IG 奖励比重 |

### 内部计算流程（5步）

```
Step 1: 构建 mask
  - last_valid_pos = 每行 response_mask 最后一个有效 token 位置 → f1_mask
  - ig_mask = response_mask==1 & ~f1_mask & token_level_rewards!=0

Step 1.5: 应用 curriculum 权重（若非 1.0）
  - weighted_rewards: f1 位置 * curriculum_f1_weight, ig 位置 * curriculum_ig_weight

Step 2: 构建 group mapping（np.unique → group_ids）

Step 3: 向量化计算 group statistics（mean/std）

Step 4: 归一化
  - "joint": F1+IG 联合 pool 归一化
  - "separate": F1 和 IG 分别 pool 归一化

Step 5: turn-level 折扣累积 + broadcast → 调用 _compute_turn_level_advantage
  - 传入 turn_boundary_mask = f1_mask | ig_mask
```

---

## 2. IGPO `_compute_turn_level_advantage` — 完整签名与输入表示

文件: `/tmp/IGPO_ref/verl/trainer/ppo/core_algos.py` 第 28 行

```python
def _compute_turn_level_advantage(
    normalized_rewards: torch.Tensor,        # (bsz, seq_len) 已归一化的 token 级奖励
    response_mask: torch.Tensor,             # (bsz, seq_len) 响应掩码
    gamma: float,                            # 折扣因子
    bsz: int,                                # batch size
    seq_len: int,                            # 序列长度
    device: torch.device,                    # 设备
    turn_boundary_mask: torch.Tensor = None, # (bsz, seq_len) 可选，预计算的 turn 边界掩码
) -> torch.Tensor:  # (bsz, seq_len) turn-level advantage broadcast 到每个 token
```

### 关键输入表示细节

**输入是 token 级 (bs, L) tensor，不是 per-turn 列表：**

- `normalized_rewards`: shape `(bsz, seq_len)`，已在上游 `compute_grpo_outcome_advantage` 完成归一化
- **turn 边界检测**：优先用 `turn_boundary_mask`（`f1_mask | ig_mask`）；若未提供则用 `normalized_rewards != 0` 启发式方法
  - 注意：必须传 `turn_boundary_mask` 以避免归一化后为 0 的 turn 被漏掉
- `response_mask` 用于 broadcast 时过滤 padding

### 分组键（归一化 pool）

归一化发生在 `_compute_turn_level_advantage` **之前**，在 `compute_grpo_outcome_advantage` 里：
- 分组键是 `index`（prompt id），即 **按 prompt 全局 pool**（global）
- `np.unique(index, return_inverse=True)` → `group_ids`

### normalize 用不用 std

受 `norm_adv_by_std_in_grpo` 控制（默认 `True`，即**用 std**）：
```python
norm_val = (token_level_rewards - mean_map)
if norm_adv_by_std_in_grpo:
    norm_val = norm_val / (std_map + epsilon)
```

### normalize → accumulate 顺序

1. **先 normalize**（在 `compute_grpo_outcome_advantage` 里）
2. **再 accumulate**（在 `_compute_turn_level_advantage` 里）

### scatter 回 token 的具体写法

```python
# 以 Python 循环实现（非向量化）
for sample_idx in range(bsz):
    # 找 turn 边界（reward 非零位置）
    reward_positions = turn_boundary_mask[sample_idx].nonzero(as_tuple=True)[0].tolist()
    
    # 从后往前折扣累积
    next_turn_adv = 0.0
    turn_data = []
    for pos in reversed(reward_positions):
        turn_reward = normalized_rewards[sample_idx, pos].item()
        turn_adv = turn_reward + gamma * next_turn_adv
        turn_data.append((pos, turn_adv))
        next_turn_adv = turn_adv
    turn_data.reverse()
    
    # broadcast：turn i 的范围 [prev_reward_pos+1, current_reward_pos]
    prev_end = 0
    for reward_pos, adv in turn_data:
        for t in range(prev_end, reward_pos + 1):
            if response_mask[sample_idx, t] == 1:
                discounted_returns[sample_idx, t] = adv
        prev_end = reward_pos + 1
```

---

## 3. `compute_all_turns_vectorized` 签名 + Belief 公式

文件: `/tmp/IGPO_ref/scrl/llm_agent/vectorized_gt_logprob.py` 第 377 行

```python
def compute_all_turns_vectorized(
    self,
    model,                                            # 语言模型
    original_input_ids: torch.Tensor,                 # (seq_len,) 原始序列 token IDs
    original_attention_mask: torch.Tensor,            # (seq_len,) 原始 1D attention mask
    original_position_ids: torch.Tensor,              # (seq_len,) 原始位置 IDs
    ground_truth_text: str,                           # GT 答案文本
    turn_end_positions: List[int],                    # 每个 turn 结束的 token 位置列表
    temperature: float = 1.0                          # logit 缩放温度
) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
    # 返回: (gt_log_probs_per_turn, gt_answer_ranges)
    # gt_log_probs_per_turn[t]: (gt_len,) 第 t turn 的 GT log probs
    # gt_answer_ranges[t]: (start, end) 答案部分在 GT 中的 token 范围
```

### Belief 公式

IGPO 中 Belief（置信度/信息增益基础值）计算方式：

```python
# 从 gt_log_probs 中取 answer 部分
answer_log_probs = log_probs[ans_start:ans_end]  # 只取 answer tokens
mean_log_prob = answer_log_probs.mean().item()   # mean(log P)

# 两种模式（由 info_gain_type 控制）：
if info_gain_type == "log_prob_diff":
    cur_value = mean_log_prob                    # log-belief = mean(log P)
else:  # "prob_diff" (默认)
    cur_value = math.exp(mean_log_prob)          # belief = exp(mean(log P))

# 信息增益 = 当前 belief - 上一 turn belief
info_gain = cur_value - prev_value
```

**结论**：
- 默认模式 (`prob_diff`): `Bel_t = exp(mean(log P_t(GT|context_t)))` = 几何平均概率
- 注意：**不是** `exp(mean(answer_log_probs))` 直接等于 `∏P_i^(1/N)`（几何均值）
- 全程 **`torch.no_grad()`**（在 `compute_all_turns_vectorized` 和 `compute_all_turns_sequential` 的 `with torch.no_grad()` 块内）

### 核心优化原理

扩展序列 `[original | GT_0 | GT_1 | ... | GT_{T-1}]`，通过 4D attention mask 让 GT_t 只能看到 context_t 以内的 token，从而一次 forward pass 算所有 turn 的 GT log probs。若检测到 FlashAttention2，自动 fallback 到 sequential 模式。

---

## 4. 本仓库 `compute_grpo_outcome_advantage` 现状 + 命名差异

文件: `/Users/jiahua.xu/dl_learn/self-researcher/verl/trainer/ppo/core_algos.py` 第 111 行

### 当前签名（vanilla GRPO）

```python
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,   # (bs, response_length)
    eos_mask: torch.Tensor,              # (bs, response_length)  ← 命名不同
    index: torch.Tensor,                 # prompt 分组索引
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
```

### 当前逻辑（无 gamma/turn/IG）

```python
scores = token_level_rewards.sum(dim=-1)  # ← sum 压缩到 scalar
# group-normalize（按 index pool）
scores[i] = (scores[i] - mean[group]) / (std[group] + epsilon)
# broadcast 回 token
scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
```

### 关键差异对照表

| 维度 | 本仓库（vanilla） | IGPO |
|------|-------------------|------|
| mask 参数名 | `eos_mask` | `response_mask` |
| `gamma` | 无 | 有，默认 `1.0` |
| `info_gain_norm_mode` | 无 | `"joint"` / `"separate"` |
| `curriculum_*_weight` | 无 | `curriculum_f1_weight`, `curriculum_ig_weight` |
| `norm_adv_by_std_in_grpo` | 无此参数（隐式 True） | 有此参数 |
| 奖励处理 | `sum(-1)` 后 scalar normalize | token 级，区分 F1/IG 位置后归一化 |
| turn 边界 | 无 | `f1_mask \| ig_mask` |
| 折扣累积 | 无 | `_compute_turn_level_advantage` |
| 输出 broadcast | `tile([1, L]) * eos_mask` | `_compute_turn_level_advantage` 结果（同样是 (bs,L)） |

### `eos_mask` ↔ `response_mask` 命名对应关系

- 本仓库 `ray_trainer.py` 第 162–163 行：调用时传 `eos_mask=response_mask`
- 本仓库内 `compute_gae_advantage_return` 等函数参数名也是 `eos_mask`
- IGPO 统一用 `response_mask`
- **两者语义完全相同**：`attention_mask[:, -response_length:]`，即响应部分的掩码（EOS 之后为 0）

---

## 5. IG 钩子在 IGPO Generation 侧的位置

文件: `/tmp/IGPO_ref/scrl/llm_agent/generation.py`

### IG 计算在 `run_llm_loop` 内，发生在每个 turn 的 rollout 之前

**核心流程**（非向量化路径，每 turn 执行一次）：

```python
# step 0（第一个 turn，只初始化 baseline value）:
pseudo_gen_output = pseudo_generate_sequences(info_gain_rollings_active, pseudo_resps_with_gt)
pseudo_gen_output_log_probs = actor_rollout_wg.compute_log_prob(pseudo_gen_output)
gt_values[i] = exp(mean(log_probs[i, gt_idx[i][0]:gt_idx[i][1]]))  # belief 基准

# step > 0（后续 turns，计算 IG）:
log_probs = pseudo_gen_output_log_probs[i, gt_idx[i][0]:gt_idx[i][1]]
cur_value = exp(mean(log_probs))  # 当前 belief
info_gain = cur_value - gt_values[i]   # IG = 当前 - 上一 turn
info_gain_rewards[i].append(info_gain)
gt_values[i] = cur_value  # 滑动更新
```

**钩子位置**：`pseudo_generate_sequences` 调用之后，`_generate_with_gpu_padding`（真实 rollout）调用之前。

即：**先用 GT 续写的伪序列计算 log prob，再生成真实 response**。

### 向量化路径（延迟计算）

当 `use_vectorized_gt_logprob=True` 时：
1. 循环内只收集 `pseudo_gen_output`（存入 `vectorized_data_collector`）
2. 循环结束后，调用 `compute_vectorized_gt_logprob`（来自 `prealigned_vectorized.py`）批量计算

### `gt_idx` 来源

`gt_idx[i]` = sample i 中 GT answer token 的 `(start, end)` 范围，在 `run_llm_loop` 入口处通过 tokenizer 定位。

---

## 6. 本仓库 rollout 入口文件

本仓库 **没有** 单 agent 的 `generation.py`（IGPO 才有）。本仓库 rollout 入口是：
- `scrl/llm_agent/multi_agent_generation.py` — 多 agent rollout（本仓库主要使用）
- `scrl/llm_agent/generation.py` — 存在，但内容为空或无 IG 逻辑（grep 无相关结果）

---

## 7. Task 1/2/3 Port 指引

### Task 1 (port `_compute_turn_level_advantage` + 新签名)

1. **复制** IGPO `_compute_turn_level_advantage` 全函数体到本仓库 `verl/trainer/ppo/core_algos.py`
2. **替换** 本仓库 `compute_grpo_outcome_advantage` 签名，增加参数：
   - `gamma: float = 1.0`
   - `info_gain_norm_mode: str = "joint"`
   - `curriculum_f1_weight: float = 1.0`
   - `curriculum_ig_weight: float = 1.0`
   - `norm_adv_by_std_in_grpo: bool = True`
3. **参数名统一**：把 `eos_mask` → `response_mask`（或在新函数内做别名）
4. **测试张量构造**：
   - `token_level_rewards (bs=2, L=20)`：在某些中间位置放 IG 奖励，最后位置放 F1 奖励
   - `response_mask`: 全 1 或带 padding 的 0/1
   - `index`: `[0, 0]`（同 prompt，测试 normalize）

### Task 2 (port `compute_all_turns_vectorized` / belief)

1. 复制 `VectorizedGTLogProbComputer` 类（`vectorized_gt_logprob.py`）或直接引用
2. 接口：`(model, input_ids(seq,), attn_mask(seq,), pos_ids(seq,), gt_text, turn_end_positions, temp)` → `(List[Tensor], List[Tuple])`
3. Belief 公式确认：`exp(mean(log_probs[ans_start:ans_end]))`，全程 `no_grad`

### Task 3 (port IG generation 钩子)

1. 在 `multi_agent_generation.py` 的 rollout 循环中，在每 turn 真实生成 **之前** 插入 IG 计算
2. 第 0 turn 只初始化 baseline（`gt_values[i]`）
3. 第 1+ turn 计算 `info_gain = cur_belief - prev_belief`，append 到 `info_gain_rewards[i]`
4. 更新 `token_level_rewards`：在各 turn end token 位置写入 info_gain 值

---

## 附：`.venv` 状态

- Python: 3.14.0（`/Users/jiahua.xu/dl_learn/self-researcher/.venv/bin/python`）
- `torch`: **未安装**（`ModuleNotFoundError: No module named 'torch'`）
- `numpy`: 未验证（torch 导入失败中断）
- `pytest`: 未验证

**结论**：`.venv` 中 torch 缺失，测试用例目录已建（`tests/hi_igpo/__init__.py`），但需要在服务器或安装了 torch 的环境中运行 pytest。
