"""Task Executor agent - performs web research with tool calling."""

import json
import re
from typing import List, Dict, Any, Optional, Tuple

from research_agent.agents.llm_client import LLMClient, create_llm_client
from research_agent.config import get_config
from research_agent.prompts.executor import get_executor_prompt, EXECUTOR_TOOLS
from research_agent.tools import web_search, browse_webpage


class TaskExecutor:
    """Executor agent that performs research with tool calling."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        max_turns: int = 10,
    ):
        config = get_config()
        if llm_client is None:
            llm_client = create_llm_client(config, "executor")
        self.llm = llm_client
        self.max_turns = max_turns
        self.search_history: List[str] = []
        self.browse_history: List[Dict] = []

    def execute(
        self,
        question: str,
        context: str = "",
        todo_list: Optional[List[Dict]] = None,
    ) -> Tuple[str, List[Dict]]:
        """
        Execute research for a question with optional TODO list.

        Args:
            question: The research question.
            context: Initial context/research so far.
            todo_list: Optional TODO list from planner.

        Returns:
            Tuple of (final_answer, research_trajectory)
        """
        messages = get_executor_prompt(question, context)

        if todo_list:
            todo_context = self._format_todo_context(todo_list)
            messages[1]["content"] += f"\n\nTODO List:\n{todo_context}"

        trajectory = []
        turns = 0

        while turns < self.max_turns:
            turns += 1
            response = self.llm.chat(messages, tools=EXECUTOR_TOOLS)

            if "error" in response:
                return f"Error: {response['error']}", trajectory

            content = response.get("content", "")
            tool_calls = response.get("tool_calls", [])

            if not tool_calls:
                answer = self._extract_answer(content)
                return answer, trajectory

            # Process tool calls
            tool_results = []
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                args_str = tc["function"]["arguments"]

                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except:
                    args = {}

                if func_name == "web_search":
                    print(f"    [{func_name}] query={args.get('query', [])}")
                elif func_name == "browse_webpage":
                    print(f"    [{func_name}] urls={args.get('url_list', [])}")
                else:
                    print(f"    [{func_name}] {args}")

                tool_result = self._execute_tool(func_name, args)
                trajectory.append({
                    "turn": turns,
                    "tool": func_name,
                    "args": args,
                    "result": tool_result,
                })
                tool_results.append((tc, tool_result))

            # Append assistant message with original tool_calls (preserves id)
            messages.append({
                "role": "assistant",
                "content": content or "",
                "tool_calls": tool_calls,
            })
            # Append each tool result
            for tc, result in tool_results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", f"call_{turns}"),
                    "content": result,
                })

        return "Max turns reached without answer", trajectory

    def _execute_tool(self, func_name: str, args: Dict) -> str:
        """Execute a tool and return the result."""
        try:
            if func_name == "web_search":
                query = args.get("query", [])
                return web_search.invoke({"query": query})

            elif func_name == "browse_webpage":
                url_list = args.get("url_list", [])
                return browse_webpage.invoke({"url_list": url_list})

            return f"Unknown tool: {func_name}"
        except Exception as e:
            return f"Tool execution error: {str(e)}"

    def _extract_answer(self, content: str) -> str:
        """Extract answer from LLM response."""
        pattern = r'<answer>(.*?)</answer>'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return content.strip()

    def _format_todo_context(self, todo_list: List[Dict]) -> str:
        """Format TODO list as context string."""
        lines = []
        for i, todo in enumerate(todo_list):
            priority = todo.get("priority", "medium").upper()
            sub_topic = todo.get("sub_topic", "")
            lines.append(f"{i+1}. [{priority}] {sub_topic}")
        return "\n".join(lines)


def create_executor_agent(
    llm_client: Optional[LLMClient] = None,
    max_turns: int = 10,
) -> TaskExecutor:
    """Factory function to create an executor agent."""
    return TaskExecutor(llm_client, max_turns)