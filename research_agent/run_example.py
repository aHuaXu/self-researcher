"""验证 Planner → Executor → Writer 三 Agent 串联流程"""

import os
import sys
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 关掉 httpx/openai 的 INFO 日志，只保留 WARNING 以上
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from research_agent.config import load_config, set_config

config = load_config()
set_config(config)

print(f"Model: {config.llm.model}")
print(f"Base URL: {config.llm.base_url}")
print(f"Serper key: {'configured' if config.search.serper_api_key else 'MISSING'}")
print()

from research_agent.graph import research

question = "总结下今日A股行情？"

print(f"研究问题: {question}")
print("=" * 60)

try:
    result = research(question)
    print(f"\n状态: {result.get('status')}")
    print(f"\nTODO 列表 ({len(result.get('todos', []))}):")
    for todo in result.get("todos", []):
        print(f"  - [{todo.get('priority', '?')}] {todo.get('sub_topic', '?')}")

    print(f"\nFindings ({len(result.get('findings', []))}):")
    for f in result.get("findings", []):
        answer = f.get("answer", "")
        print(f"  - {f.get('sub_topic', '?')}: {answer}")

    print(f"\n{'=' * 60}")
    print("报告:")
    print(result.get("report", "No report"))
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
