"""简单测试：验证三个 Agent 串联流程"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(os.path.join(os.path.dirname(__file__), '.env.example'))

from research_agent.config import set_config, get_config
from research_agent import research

# 测试简单问题
question = "What is the capital of France?"

print(f"问题: {question}")
print("=" * 50)
print("开始研究...")

try:
    result = research(question)
    print(f"\n状态: {result.get('status')}")
    print(f"\nTODO 数量: {len(result.get('todos', []))}")
    print(f"Findings 数量: {len(result.get('findings', []))}")
    print(f"\n报告:\n{result.get('report', 'No report')[:500]}...")
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()