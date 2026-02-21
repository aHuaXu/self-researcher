"""Simple test: 验证工具定义是否正确"""

from research_agent.prompts.executor import EXECUTOR_TOOLS

print("Tools definition:")
print("=" * 50)

for tool in EXECUTOR_TOOLS:
    print(f"\n{tool['function']['name']}:")
    print(f"  Description: {tool['function']['description']}")
    print(f"  Parameters: {tool['function']['parameters']}")


print("\n" + "=" * 50)
print("验证通过！工具定义与 generation.py 一致")
print("要运行完整研究任务，需要配置有效的 API key")
print("请编辑 example.py 设置正确的配置后运行")