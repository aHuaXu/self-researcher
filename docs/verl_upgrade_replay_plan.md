# VERL 升级与本地改动重放计划

## 目标

- 将项目从当前内置 `verl` 迁移到上游最新实现，降低长期兼容性成本（`transformers` / `vllm` / Qwen3 / Ulysses）。
- 保留本项目关键能力：
  - 单 agent GRPO 训练链路
  - 多 agent（planner/executor/writer）训练链路
  - 线上搜索与工具调用链路
  - V100 资源约束下的可运行性

## 现状快照（已核对）

- 上游参考版本：`verl-project/verl@af4901c`
- 本地与上游 `verl` 差异规模（目录级对比）：
  - `local_only`: 141 文件
  - `up_only`: 332 文件
  - `changed_common`: 82 文件
  - 其中训练核心相关差异（`workers/models/trainer/third_party/utils`）约 72 文件
- 结论：**不适合一次性“整体替换 + 全量手工修复”**，必须分阶段迁移并设置门禁。

## 已执行的迁移锚点（2026-05-17）

- 当前迁移分支：`migration/verl-upstream-af4901c`
- 代码锚点 tag：`anchor/pre-verl-upgrade-be918d2`
- 旧版 `verl` 快照目录：`.tmp/verl-local-legacy`
- 上游 `verl` 快照目录：`.tmp/verl-upstream-main/verl`

### 如何对照旧实现并重放

- 看锚点版本文件：
  - `git show anchor/pre-verl-upgrade-be918d2:verl/workers/fsdp_workers.py`
  - `git show anchor/pre-verl-upgrade-be918d2:verl/models/transformers/monkey_patch.py`
- 看本地旧快照：
  - `.tmp/verl-local-legacy/...`
- 看当前迁移后实现：
  - `verl/...`

推荐固定“三方对照”：
1. 锚点旧实现（项目特有逻辑来源）
2. 上游最新实现（目标结构）
3. 当前迁移实现（重放落点）

## 迁移原则

1. **先跑通再优化**：优先恢复可训练闭环，不在迁移早期追求全部特性。
2. **分层重放**：核心训练路径 > 模型补丁 > rollout/vllm > 多 agent。
3. **每层有验收门槛**：必须通过既定门禁后再进入下一层。
4. **保留回滚点**：每个阶段独立分支，避免大爆炸合并。

## 需要重放的本地关键改动（高优先级）

### A. 训练核心链路

- `verl/workers/fsdp_workers.py`
  - Actor/Critic 初始化与配置注入
  - Ulysses + remove_padding 兼容路径
  - 多 agent LoRA adapter 接入（planner/executor/writer）
- `verl/workers/actor/dp_actor.py`
  - 多 agent adapter 路由
  - PPO update 相关逻辑和调试埋点
- `verl/trainer/ppo/ray_trainer.py`
  - 训练流程里与本项目数据/多 agent相关的扩展

### B. Qwen3 / Ulysses / 注意力补丁

- `verl/models/transformers/monkey_patch.py`
- `verl/models/transformers/qwen2.py`
- `verl/models/transformers/qwen3.py`（本地新增）
- `verl/models/registry.py`

### C. vLLM SPMD 与显存管理

- `verl/workers/sharding_manager/fsdp_vllm.py`
- `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`
- `verl/third_party/vllm/vllm_spmd/dtensor_weight_loaders.py`

### D. 训练配置与入口

- `verl/trainer/config/ppo_trainer.yaml`
- `scripts/train/grpo_qwen3_4b.sh`（项目主脚本）
- `.env`（工具调用、搜索链路、超时等）

### E. 多 agent 业务层（`verl` 外）

- `scrl/llm_agent/multi_agent_generation.py`
- `research_agent/agents/*`
- `research_agent/prompts/*`
- `verl/workers/reward_manager/multi_agent.py`

## 可延后/可下沉处理（中低优先级）

- Megatron 相关本地扩展（若当前生产链路不依赖，可后置）
- 历史多版本 vLLM 兼容目录（`v_0_3_1`/`v_0_4_2`/`v_0_5_4` 等）可在后续统一收敛
- 非关键 debug/诊断日志先最小化迁移

## 分阶段执行计划

## Phase 0：基线固化（当前分支）

- 记录当前可复现实验配置（脚本 + 环境 + 数据）
- 保存关键失败样例（当前已知：Qwen3 Ulysses 进入 compute_log_prob 后的稳定性问题）
- 产出最小验收脚本（单步训练 + 日志关键字检查）

**门禁**
- 能稳定复现当前基线行为（成功或失败都可复现）

## Phase 1：引入上游最新 `verl`（不迁移业务改动）

- 在迁移分支替换 `verl` 代码为上游 `af4901c`
- 修复基础依赖/导入/配置兼容（仅做编译与启动级别修复）

**门禁**
- 训练入口可启动到 worker 初始化，不因导入/配置直接崩溃

## Phase 2：单 agent 最小闭环重放

- 重放 A/B/C/D 中“单 agent必需”的改动（先不引入多 agent）
- 优先恢复：
  1. rollout
  2. compute_log_prob
  3. update_actor
  4. global_metrics

**门禁**
- 连续通过：`step1 -> compute_log_prob -> update_actor -> global_metrics`
- 无 shape mismatch 类错误
- 无不可恢复 OOM（允许通过参数回退规避）

## Phase 3：多 agent 能力重放

- 重放 E 类改动（planner/executor/writer）
- 接通多 agent reward manager 与 LoRA adapter 流

**门禁**
- 多 agent 训练可启动并完成至少 1 个完整 step
- agent 维度的日志与 reward 统计正常

## Phase 4：收敛与清理

- 删除已不再需要的旧补丁与冗余兼容层
- 补齐回归测试与运维文档

**门禁**
- 关键训练脚本通过
- 核心回归测试通过

## 风险与对策

- 风险：上游接口变化导致重放成本高  
  对策：分层迁移 + 每层门禁 + 小步提交

- 风险：V100 与新注意力后端不匹配  
  对策：在配置层显式按 GPU 能力分流注意力实现，避免隐式选择

- 风险：多 agent 与主训练框架耦合过深  
  对策：先抽象接口（batch 构造、adapter 选择、reward 聚合）再移植

## 下一步（建议立即执行）

1. 建立迁移工作分支（仅用于上游同步，不混入业务迭代）
2. 先完成 Phase 1（上游替换 + 启动级修复）
3. 我来给出 Phase 2 的“最小必迁文件清单 + 逐文件重放顺序”

