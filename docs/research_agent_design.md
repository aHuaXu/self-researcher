# 多 Agent 研究助手项目结构设计

## Context

将 hello-agents-ch14 的多 Agent 研究助手架构与 DeepResearcher 项目结合，形成一个可进行整体 RL 训练的研究助手系统。

**目标**:
1. 统一工具层：搜索/浏览工具由 `scrl/handler/` 提供，训练流程和研究助手共享同一套实现
2. 保持现有 DeepResearcher 训练流程不变
3. 新增 Planner → Executor → Writer 三 Agent 研究助手

---

## 项目结构

```
self-researcher/
├── scrl/                              # 搜索处理 + 推理（重构为正规 Python 包）
│   ├── __init__.py                    # [新增]
│   ├── llm_agent/                     # LLM 推理（不变）
│   │   ├── __init__.py
│   │   ├── generation.py
│   │   └── tensor_helper.py
│   └── handler/                       # 搜索/浏览工具核心实现
│       ├── __init__.py                # [新增]
│       ├── handler.py                 # Handler 编排（文件 IPC / Flask API）
│       ├── server_handler.py          # Flask 搜索服务
│       ├── webpage.py                 # 数据类: WebPageInfo, SearchResultInfo, PageReadInfo
│       ├── agent_action.py            # 数据类: ActionInfo, SubActionInfo
│       ├── utils.py                   # LLM 调用 + tag 解析工具函数
│       ├── config.yaml                # 搜索配置
│       ├── reading_agent/             # ReadingAgent — 逐页 LLM 提取
│       │   ├── __init__.py            # [新增]
│       │   ├── reading_agent.py
│       │   └── prompts.py
│       └── web_search_agent/          # WebSearchAgent — Serper 搜索 + 浏览器
│           ├── __init__.py            # [新增]
│           ├── web_search_agent.py
│           ├── prompts.py
│           └── search/                # SimpleTextBrowser + MarkdownConverter
│               ├── __init__.py
│               ├── text_web_browser.py
│               ├── search_api.py
│               ├── mdconvert.py
│               └── cookies.py
├── research_agent/                    # 多 Agent 研究助手（新增模块）
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── llm_client.py             # LLM 客户端封装
│   │   ├── planner.py                # TODO Planner（小模型）
│   │   ├── executor.py               # Task Executor（DeepResearcher 模型）
│   │   └── writer.py                 # Report Writer（小模型）
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── _state.py                 # [新增] 共享状态管理（search↔browse 耦合）
│   │   ├── search.py                 # web_search — 委托给 scrl.handler
│   │   └── browse.py                 # browse_webpage — 委托给 scrl.handler
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── planner.py
│   │   ├── executor.py
│   │   └── writer.py
│   ├── config.py                     # 配置
│   ├── graph.py                      # LangGraph 多 Agent 编排
│   └── run_example.py                # 运行示例
└── verl/                              # 训练框架（不变）
```

---

## 核心设计：统一工具层

### 原则

工具处理逻辑只在 `scrl/handler/` 中实现一次，`research_agent/tools/` 只做 LangChain @tool 封装 + 状态管理。

### 数据流

**训练流程**（不变）：
```
vLLM → tool_call JSON → signal/data.json → handler.py → WebSearchAgent/ReadingAgent → 结果
```

**研究助手**（新增）：
```
Executor Agent → LangChain @tool → _state.py → WebSearchAgent/ReadingAgent → 结果
```

两条路径调用的是**同一个** `WebSearchAgent` 和 `ReadingAgent`，保证工具行为完全一致。

### 共享状态管理 (_state.py)

`browse_webpage` 依赖 `web_search` 的结果（WebPageInfo 中的 browser 对象）。在原 handler 中由 `Handler.id_to_context` 管理。

研究助手中由 `ToolState` 单例管理：
```python
class ToolState:
    web_search_agent: WebSearchAgent
    reading_agent: ReadingAgent
    current_question: str
    action_info: ActionInfo | None    # 最近一次 search 结果，供 browse 使用
    api_result_dict: dict             # 搜索缓存
```

### 工具参数与返回格式

必须与 `scrl/llm_agent/generation.py` 中的 TOOLS 定义完全一致：

**web_search**: `query: list[str]`
```json
[{"search_query": "...", "web_page_info_list": [{"title": "...", "url": "...", "quick_summary": "..."}, ...]}]
```

**browse_webpage**: `url_list: list[str]`
```json
[{"url": "...", "information": [{"page_number": 0, "page_summary": "..."}, ...]}]
```

---

## scrl/handler/ 重构要点

1. 新增 4 个 `__init__.py`（空文件）
2. 14 个裸 import 改为绝对 import（如 `from utils import` → `from scrl.handler.utils import`）
3. 已有的相对 import（`from .prompts import *`）不需改
4. `handler.py` 和 `server_handler.py` 改用 `python -m scrl.handler.handler` 方式运行

---

## 实施计划

### Phase 1: scrl 包化
1. 创建 4 个 `__init__.py`
2. 修复 6 个文件的裸 import
3. 安装依赖（smolagents, pathvalidate, mammoth 等）
4. 验证 `from scrl.handler.handler import Handler` 可导入

### Phase 2: 工具封装
1. 创建 `research_agent/tools/_state.py`
2. 重写 `research_agent/tools/search.py` — 委托给 `scrl.handler`
3. 重写 `research_agent/tools/browse.py` — 委托给 `scrl.handler`

### Phase 3: 编排 + 验证
1. 更新 `research_agent/graph.py` 初始化 ToolState
2. 运行 `run_example.py` 端到端验证

### Phase 4: 统一工具调用（训练和研究助手）
1. 修改 `scrl/llm_agent/generation.py` 的 `execute_predictions` → 直接调用 ToolState
2. 训练前初始化 ToolState（无需启动 server_handler/handler 进程）
3. 工具行为完全一致

---

## 训练工具调用：从文件 IPC 改为直接调用

### 旧流程（文件 IPC）
```
python -m scrl.handler.server_handler  # Flask :5000
python -m scrl.handler.handler     # polling

generation.py:
  write data.json → signal.json=1 → sleep 10s 轮询 → 读结果
```

### 新流程（直接调用）
```
训练前只需初始化:
  tool_state.initialize(config, client)

generation.py:
  web_search.invoke({"query": [...]})  → 毫秒级返回
```

### 优势
- 去掉文件 IPC（无 signal/data.json）
- 去掉 10 秒 sleep 轮询
- 无需手动启动 server_handler/handler 进程
- 多 GPU 训练时工具在主进程直接调用，无跨进程开销

---

## 与原始 DeepResearcher 的关系

| 组件 | 原始 DeepResearcher | research_agent |
|------|-------------------|---------------|
| 搜索工具 | scrl/handler/web_search_agent/ | 同一份代码，LangChain @tool 封装 |
| 浏览工具 | scrl/handler/reading_agent/ | 同一份代码，LangChain @tool 封装 |
| 工具参数 | generation.py:70-114 的 TOOLS | 完全一致 |
| LLM 调用 | scrl/llm_agent/generation.py | 独立封装，复用相同的 LLM |
| 编排 | handler.py (文件 IPC) | LangGraph (Planner→Executor→Writer) |

---

## Multi-Agent RL 训练方案

### 整体思路

共享基座模型（DeepResearcher-7B）+ 3 个 LoRA adapter，串行生成完整 rollout，分别计算 reward，独立更新各自的 LoRA 参数。

```
DeepResearcher-7B (冻结基座)
  ├── LoRA_planner
  ├── LoRA_executor
  └── LoRA_writer

Query → Planner(LoRA_p) → Executor(LoRA_e) → Writer(LoRA_w) → Report
              ↓                  ↓                  ↓
          reward_p           reward_e           reward_w
              ↓                  ↓                  ↓
        update LoRA_p      update LoRA_e      update LoRA_w
```

### 模型架构

- **基座模型**：DeepResearcher-7B（GRPO 训练后的 Qwen2.5-7B），冻结不更新
- **LoRA adapter**：每个 Agent 一个独立的 LoRA 模块（r=16~64），只训练 LoRA 参数
- **推理**：一个 vLLM 实例，通过 `--enable-lora` 支持多 LoRA 切换，基座内存只占一份

### Rollout 流程

```
┌─────────────────────────────────────────────────────────────┐
│                     Multi-Agent Rollout                      │
├─────────────────────────────────────────────────────────────┤
│  Query (研究问题)                                            │
│    │                                                        │
│    ▼  切换 LoRA_planner                                     │
│  ┌─────────────────┐                                       │
│  │    Planner      │ ──→ TODO list (子主题 + 优先级)        │
│  │ (base + LoRA_p) │                                       │
│  └─────────────────┘                                       │
│    │                                                        │
│    ▼  切换 LoRA_executor                                    │
│  ┌─────────────────┐                                       │
│  │   Executor      │ ──→ 多轮 web_search + browse          │
│  │ (base + LoRA_e) │     → findings (研究发现)              │
│  └─────────────────┘                                       │
│    │                                                        │
│    ▼  切换 LoRA_writer                                      │
│  ┌─────────────────┐                                       │
│  │    Writer       │ ──→ 结构化研究报告                     │
│  │ (base + LoRA_w) │                                       │
│  └─────────────────┘                                       │
│    │                                                        │
│    ▼                                                        │
│  LLM Judge → final_reward (1-10 分)                        │
└─────────────────────────────────────────────────────────────┘
```

### Reward 设计

#### final_reward

由冻结的 LLM Judge（如 Qwen2.5-72B）对最终报告打分（1-10），评估维度：
- 准确性：信息是否正确
- 完整性：是否覆盖了问题的核心方面
- 结构性：报告是否有清晰的逻辑结构
- 可读性：语言是否流畅

#### 各 Agent 的 reward 计算

```
reward_writer   = final_reward_normalized
reward_executor = β · rule_e + (1 - β) · final_reward_normalized
reward_planner  = α · rule_p + (1 - α) · final_reward_normalized
```

- α = 0.2, β = 0.3（初始值，可随训练 anneal 到 0）
- final_reward_normalized = final_reward / 10（归一化到 0-1）

#### Planner 规则 reward (rule_p)

| 规则 | 分值 | 说明 |
|------|------|------|
| 子任务数量在 3-7 之间 | +1 / 0 | 太少覆盖不全，太多发散 |
| 子任务之间无明显重复（关键词重合率 < 阈值） | +1 / 0 | 避免浪费 Executor 轮次 |
| 优先级分布合理（不全是同一个级别） | +0.5 / 0 | 有主次区分 |

归一化：rule_p = 得分 / 2.5

#### Executor 规则 reward (rule_e)

| 规则 | 分值 | 说明 |
|------|------|------|
| 成功调用了搜索工具（返回非空结果） | +1 / 0 | 基本搜索能力 |
| 调用了 browse（深入阅读网页） | +1 / 0 | 不只看摘要 |
| 返回内容非空且长度合理 | +1 / 0 | 确实找到了有效信息 |
| 没有触发 max_turns 限制 | +1 / 0 | 能在限制内完成任务 |

归一化：rule_e = 得分 / 4

### 训练数据

不复用 DeepResearcher 的 QA 数据（只有问-答对，无法训练报告生成）。

需要构造**研究类问题**数据集，不需要标准答案（reward 来自 LLM Judge）：

```json
{"query": "分析2024年全球AI芯片市场格局", "domain": "tech"}
{"query": "比较中美新能源汽车产业链差异", "domain": "industry"}
{"query": "总结最近一周加密货币市场走势", "domain": "finance"}
```

数据来源：
- LLM 批量生成不同领域的研究问题
- 知乎/Reddit 等平台的研究类问题
- 学术论文的 research question
- 金融/行业研究报告的标题

### GRPO 训练细节

```python
# 对同一个 query 的 n=16 条 rollout:
# 每条 rollout 包含完整的 Planner → Executor → Writer 过程
# 分别对三个 LoRA 做 GRPO 更新

for query in batch:
    rollouts = [run_full_pipeline(query) for _ in range(16)]
    
    # 每条 rollout 得到三个 reward
    rewards_p = [r.reward_planner for r in rollouts]
    rewards_e = [r.reward_executor for r in rollouts]
    rewards_w = [r.reward_writer for r in rollouts]
    
    # GRPO 组内归一化
    adv_p = normalize(rewards_p)  # 好的 plan → 正 advantage → 强化 LoRA_p
    adv_e = normalize(rewards_e)
    adv_w = normalize(rewards_w)
    
    # 分别更新三个 LoRA
    update_lora(planner_lora, trajectories_p, adv_p)
    update_lora(executor_lora, trajectories_e, adv_e)
    update_lora(writer_lora, trajectories_w, adv_w)
```

### 对 verl 框架的改动

| 模块 | 改动 |
|------|------|
| `LLMGenerationManager` | 三阶段串行 rollout，每阶段切换 LoRA |
| `core_algos.py` | 分别对三个 Agent 的 trajectory 计算 GRPO advantage |
| `fsdp_workers.py` | Actor update 只更新对应的 LoRA 参数 |
| `vllm_rollout/` | 启用 `--enable-lora`，支持多 LoRA 切换 |
| `reward_score/` | 新增 LLM Judge reward + 规则 reward |
| `RLHFDataset` | 适配新的训练数据格式（研究问题，无 ground_truth） |

### 资源估算

- 基座 7B 模型：~14GB 显存（FP16）
- 3 个 LoRA（r=64）：每个 ~200MB，共 ~600MB
- vLLM rollout 引擎：~14GB（KV cache）
- LLM Judge（本地 72B）：需要额外 GPU
- 总计：8×A100 80GB 应该足够（基座 + LoRA 训练 + Judge 推理）
