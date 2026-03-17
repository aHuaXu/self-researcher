"""Research Agent package.

Heavy graph dependencies are lazy-loaded so ``import research_agent.tools`` does not
require langgraph or the full assistant stack.
"""

from typing import Any, List

__all__: List[str] = ["ResearchAssistant", "research"]


def __getattr__(name: str) -> Any:
    if name == "ResearchAssistant":
        from research_agent.graph import ResearchAssistant

        return ResearchAssistant
    if name == "research":
        from research_agent.graph import research

        return research
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return list(__all__)
