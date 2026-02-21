# 多 Agent 研究助手项目结构设计

## Context

将 hello-agents-ch14 的多 Agent 研究助手架构与 DeepResearcher 项目结合，形成一个可进行整体 RL 训练的研究助手系统。

> **注意**: 本设计基于 DeepResearcher 的实际实现（单 Executor Agent + 2 工具），而非直接复用 hello-agents-ch14 的 3 Agent 架构

**目标**:
1. 设计清晰的项目结构
2. 保持现有 DeepResearcher 流程不变
3. 支持后续多 Agent RL 训练

---

## 项目结构

```
DeepResearcher/
├── research_agent/                    # 新建研究助手模块
│   ├── __init__.py
│   ├── agents/                        # Agent 定义
│   │   ├── __init__.py
│   │   ├── llm_client.py          # LLM 客户端封装
│   │   ├── planner.py              # TODO Planner (3B 小模型)
│   │   ├── executor.py            # Task Executor (DeepResearcher 模型)
│   │   └── writer.py             # Report Writer (3B 小模型)
│   ├── tools/                     # 工具定义 (LangChain @tool)
│   │   ├── __init__.py
│   │   ├── search.py             # web_search 工具
│   │   └── browse.py             # browse_webpage 工具
│   ├── prompts/                   # Agent 提示词
│   │   ├── __init__.py
│   │   ├── planner.py
│   │   ├── executor.py
│   │   └── writer.py
│   ├── config.py                # 配置
│   └── graph.py                 # LangGraph 多 Agent 编排
├── scr/                          # 现有推理代码 (不变)
└── verl/                         # 训练框架 (不变)
```

---

## 技术选型

- **Agent 框架**: LangChain / LangGraph
- **工具定义**: @tool 装饰器
- **多 Agent 编排**: LangGraph
- **LLM 调用**: LangChain LLM 接口

> **注意**: 工具参数必须与 scrl/llm_agent/generation.py 中的 TOOLS 定义完全一致:
> - web_search: `query: list[str]`
> - browse_webpage: `url_list: list[str]`

---

## 实施计划

### Phase 1: 项目框架搭建
1. 安装依赖: langchain, langchain-core, langgraph, python-dotenv
2. 创建 research_agent/ 目录结构
3. 创建配置类 config.py

### Phase 2: 工具封装 (LangChain @tool)
> **重要**: 工具封装只新增 LangChain 层的包装，不修改原有实现，保证原 DeepResearcher 流程不变
>
> **关键**: 工具实现必须与 handler.py:280-339 的逻辑完全一致

1. 创建 tools/search.py - 复用 scrl/handler/web_search_agent/
   - 调用 web_search_agent.search_web_batch()
   - 参数: `query: list[str]`
   - 返回格式: `[{search_query, web_page_info_list: [{title, url, quick_summary}, ...]}, ...]`

2. 创建 tools/browse.py - 复用 scrl/handler/reading_agent/
   - 调用 reading_agent.read_batch()
   - 参数: `url_list: list[str]`
   - 必须先调用过 web_search（依赖 context）
   - 返回格式: `[{url, information: [{page_number, page_summary}, ...]}, ...]`

### Phase 3: Agent 实现 (LangChain)
1. 实现 agents/planner.py - TODO Planner (3B 小模型)
2. 实现 agents/executor.py - Task Executor
   - 绑定 DeepResearcher 模型 (待训练)
   - 绑定 search/browse 工具
3. 实现 agents/writer.py - Report Writer (3B 小模型)
4. 实现 agents/llm_client.py - LLM 客户端封装

### Phase 4: LangGraph 多 Agent 编排
1. 创建 graph.py
2. 定义节点: planner -> executor -> writer
3. 定义边和条件判断
4. 创建 run_example.py 运行示例验证流程

### Phase 5: 验证与测试
1. 运行 run_example.py 验证流程
2. 修复发现的问题

### 配置文件
- research_agent/.env.example - 环境变量示例
- research_agent/run_example.py - 运行示例验证流程

---

## 关键文件

### 新建文件
- research_agent/config.py - 配置 (LLM 名称、API keys 等)
- research_agent/tools/search.py - 搜索工具 (LangChain @tool, 参数: query: list[str])
- research_agent/tools/browse.py - 浏览工具 (LangChain @tool, 参数: url_list: list[str])
- research_agent/agents/planner.py
- research_agent/agents/executor.py
- research_agent/agents/writer.py
- research_agent/agents/llm_client.py
- research_agent/graph.py - LangGraph 编排 (planner -> executor -> writer)
- research_agent/run_example.py - 运行示例

### 复用文件
- scrl/handler/web_search_agent/search/search_api.py - 搜索 API (仅用配置)
- scrl/handler/web_search_agent/web_search_agent.py - WebSearchAgent 类
- scrl/handler/reading_agent/reading_agent.py - ReadingAgent 类
- scrl/llm_agent/generation.py - LLM 调用 (保留，现流程不变)

### 关键: 工具返回格式必须与 handler.py:280-339 一致

**web_search 返回格式**:
```json
[{
  "search_query": "...",
  "web_page_info_list": [
    {"title": "...", "url": "...", "quick_summary": "..."},
    ...
  ]
}, ...]
```

**browse_webpage 返回格式**:
```json
[{
  "url": "...",
  "information": [
    {"page_number": 0, "page_summary": "..."},
    ...
  ]
}, ...]
```

---

## 验证方式

1. 单独测试每个 LangChain Tool
2. 单独测试每个 LangChain Agent
3. 测试 LangGraph 多 Agent 流程
4. 对比原有 DeepResearcher 流程输出

---

## 与原始 DeepResearcher 的关系

> 本模块是 LangChain/LangGraph 层的封装，不修改原始 scrl/ 流程

| 组件 | 原始 DeepResearcher | research_agent |
|------|-------------------|---------------|
| LLM 调用 | scrl/llm_agent/generation.py | 仅外层封装，复用相同的 LLM |
| 工具 | scrl/handler/web_search_agent/ | 仅包装，原实现不变 |
| 工具参数 | generation.py:70-114 的 TOOLS | 必须完全一致 |

**集成方式**:
- 独立使用: `from research_agent import research`
- RL 训练: 复用 generation.py 的工具定义和执行逻辑