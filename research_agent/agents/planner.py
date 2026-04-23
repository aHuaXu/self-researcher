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
            # Fallback: create a simple todo from the question
            todos = [{
                "sub_topic": question,
                "search_query": question,
                "priority": "high"
            }]

        return todos

    def _parse_todos(self, content: str) -> List[Dict[str, Any]]:
        """Parse TODO items from LLM response."""
        todos = []

        # Try strict format first: 1. [HIGH] Sub-topic: XXX\n   Search Query: XXX
        pattern = r'(\d+)\.\s*\[(HIGH|MEDIUM|LOW)\]\s*Sub-topic:\s*(.+?)\n\s*Search Query:\s*(.+?)(?=\n\d+\.|\n\n|</todos>|$)'
        matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
        for match in matches:
            search_query = match[3].strip().rstrip('</todos>')
            todos.append({
                "index": int(match[0]),
                "priority": match[1].lower(),
                "sub_topic": match[2].strip(),
                "search_query": search_query,
            })

        if todos:
            return todos

        # Fallback: match numbered items with brackets like "1. [HIGH] ..."
        pattern2 = r'(\d+)\.\s*\[(HIGH|MEDIUM|LOW)\]\s*(.+?)(?=\n\d+\.\s*\[|$)'
        matches2 = re.findall(pattern2, content, re.IGNORECASE | re.DOTALL)
        for match in matches2:
            text = match[2].strip()
            # Try to split sub_topic and search_query
            sq_match = re.search(r'(?:Search\s*Query|搜索|查询)[：:]\s*(.+)', text, re.IGNORECASE)
            if sq_match:
                sub_topic = text[:sq_match.start()].strip().rstrip('\n')
                search_query = sq_match.group(1).strip()
            else:
                sub_topic = text.split('\n')[0].strip()
                search_query = sub_topic
            sub_topic = re.sub(r'^(?:Sub-topic|子主题|主题)[：:]\s*', '', sub_topic, flags=re.IGNORECASE)
            todos.append({
                "index": int(match[0]),
                "priority": match[1].lower(),
                "sub_topic": sub_topic,
                "search_query": search_query,
            })

        return todos


def create_planner_agent(llm_client: Optional[LLMClient] = None) -> TodoPlanner:
    """Factory function to create a planner agent."""
    return TodoPlanner(llm_client)