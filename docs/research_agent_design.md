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
