"""Example: 运行多 Agent 深度研究任务"""

from research_agent.config import set_config, AgentConfig

# 1. 设置配置（替换为你的实际配置）
config = AgentConfig()
config.search.serper_api_key = "your-serper-api-key"
config.llm.executor_model = "qwen-plus"  # 用可用的模型测试
config.llm.executor_base_url = "https://api.qwen.com/v1"
config.llm.executor_api_key = "your-api-key"
config.llm.small_model = "qwen2.5-3b"
config.llm.small_base_url = "https://api.qwen.com/v1"
config.llm.small_api_key = "your-api-key"
config.llm.summary_model = "qwen-plus"
config.llm.summary_base_url = "https://api.qwen.com/v1"
config.llm.summary_api_key = "your-api-key"
set_config(config)

# 2. 导入并运行
from research_agent.graph import ResearchAssistant

assistant = ResearchAssistant()

# 3. 运行研究任务
question = "What is the capital of France?"

print(f"Research Question: {question}")
print("=" * 50)

result = assistant.research(question)

# 4. 打印结果
print(f"\nStatus: {result.get('status')}")
print(f"\nTODOs ({len(result.get('todos', []))}):")
for i, todo in enumerate(result.get('todos', [])):
    print(f"  {i+1}. [{todo.get('priority')}] {todo.get('sub_topic')}")
    print(f"     Search: {todo.get('search_query')}")

print(f"\nFindings ({len(result.get('findings', []))}):")
for i, finding in enumerate(result.get('findings', [])):
    print(f"  {i+1}. {finding.get('tool')}: {finding.get(' args', {})}")

print(f"\nReport:\n{result.get('report', 'No report generated')}")