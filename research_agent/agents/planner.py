"""TODO Planner agent - breaks down research questions into actionable items."""

import re
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import BaseTool

from research_agent.agents.llm_client import LLMClient, create_llm_client
from research_agent.config import get_config
from research_agent.prompts.planner import get_planner_prompt


class TodoPlanner:
    """Planner agent that breaks down research questions into TODO items."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        if llm_client is None:
            config = get_config()
            llm_client = create_llm_client(config, "small")
        self.llm = llm_client

    def plan(self, question: str) -> List[Dict[str, Any]]:
        """
        Break down a research question into TODO items.

        Args:
            question: The research question to break down.

        Returns:
            List of TODO items with sub-topic, search_query, and priority.
        """
        messages = get_planner_prompt(question)
        response = self.llm.chat(messages)

        if "error" in response:
            return [{"error": response["error"]}]

        content = response.get("content", "")
        todos = self._parse_todos(content)
        if not todos:
            print(f"  [Planner] 解析失败，LLM 原始输出:\n{content[:500]}")

        if not todos:
            todos = [{
                "sub_topic": question,
                "priority": "high"
            }]

        return todos

    def _parse_todos(self, content: str) -> List[Dict[str, Any]]:
        """Parse TODO items from LLM response.

        Aligned with training-time parsing in multi_agent_generation.py.
        Output: {index, priority, sub_topic} — no search_query.
        """
        todos = []

        pattern = r'(\d+)\.\s*\[(HIGH|MEDIUM|LOW)\]\s*(.+?)(?=\n\d+\.\s*\[|</todos>|$)'
        matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
        for match in matches:
            sub_topic = match[2].strip()
            sub_topic = re.sub(
                r'^(?:Sub-topic|子主题|主题)[：:]\s*',
                '',
                sub_topic,
                flags=re.IGNORECASE,
            )
            sub_topic = sub_topic.rstrip('</todos>').strip()
            todos.append({
                "index": int(match[0]),
                "priority": match[1].lower(),
                "sub_topic": sub_topic,
            })

        if todos:
            return todos

        clean_text = content.strip()
        if clean_text:
            todos.append({
                "index": 1,
                "priority": "high",
                "sub_topic": clean_text[:200],
            })

        return todos


def create_planner_agent(llm_client: Optional[LLMClient] = None) -> TodoPlanner:
    """Factory function to create a planner agent."""
    return TodoPlanner(llm_client)