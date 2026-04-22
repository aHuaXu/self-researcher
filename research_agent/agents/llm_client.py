"""Base LLM client wrapper for Research Agent."""

from typing import List, Dict, Any, Optional
from openai import OpenAI


class LLMClient:
    """Wrapper for LLM API calls."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Call the LLM with messages."""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }

        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}

        try:
            response = self.client.chat.completions.create(**kwargs)

            if stream:
                return response

            message = response.choices[0].message
            result = {
                "content": message.content or "",
                "finish_reason": response.choices[0].finish_reason,
            }

            if hasattr(message, "tool_calls") and message.tool_calls:
                result["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in message.tool_calls
                ]

            return result
        except Exception as e:
            return {"content": "", "error": str(e)}


def create_llm_client(config: "AgentConfig", model_type: str = "small") -> LLMClient:
    """Create an LLM client. All agents share the same model for now."""
    from research_agent.config import AgentConfig

    return LLMClient(
        model=config.llm.model,
        base_url=config.llm.base_url,
        api_key=config.llm.api_key,
    )