"""Tools package — lazy imports so ``import research_agent.tools._state`` stays lightweight."""

from typing import Any, List

__all__ = [
    "web_search",
    "browse_webpage",
    "get_tool_state",
    "build_handler_config",
]


def __getattr__(name: str) -> Any:
    if name == "web_search":
        from research_agent.tools.search import web_search

        return web_search
    if name == "browse_webpage":
        from research_agent.tools.browse import browse_webpage

        return browse_webpage
    if name == "get_tool_state":
        from research_agent.tools._state import get_tool_state

        return get_tool_state
    if name == "build_handler_config":
        from research_agent.tools._state import build_handler_config

        return build_handler_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return list(__all__)
