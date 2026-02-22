"""Shared state between web_search and browse_webpage tools."""

from typing import Dict, Optional

from openai import OpenAI

from scrl.handler.web_search_agent.web_search_agent import WebSearchAgent
from scrl.handler.reading_agent.reading_agent import ReadingAgent
from scrl.handler.agent_action import ActionInfo


class ToolState:
    """Mutable state shared between search and browse tools.

    browse_webpage needs WebPageInfo objects created during web_search,
    mirroring Handler.id_to_context in scrl/handler/handler.py.
    """

    def __init__(self):
        self.web_search_agent: Optional[WebSearchAgent] = None
        self.reading_agent: Optional[ReadingAgent] = None
        self.current_question: str = ""
        self.action_info: Optional[ActionInfo] = None
        self.api_result_dict: Dict = {}
        self._initialized = False

    def initialize(self, config: dict, client: OpenAI):
        if self._initialized:
            return
        self.web_search_agent = WebSearchAgent(config=config, client=client)
        self.reading_agent = ReadingAgent(config=config, client=client)
        self._initialized = True

    def reset_for_question(self, question: str):
        self.current_question = question
        self.action_info = None


_state = ToolState()


def get_tool_state() -> ToolState:
    return _state


def build_handler_config(agent_config) -> dict:
    """Convert research_agent AgentConfig to the dict format scrl/handler expects."""
    return {
        "search_engine": agent_config.search.engine,
        "serper_api_key": agent_config.search.serper_api_key,
        "search_top_k": agent_config.search.top_k,
        "search_region": agent_config.search.region,
        "search_lang": agent_config.search.lang,
        "azure_bing_search_subscription_key": agent_config.search.azure_subscription_key,
        "azure_bing_search_mkt": agent_config.search.azure_mkt,
        "quick_summary_model": agent_config.llm.model,
        "reading_agent_model": agent_config.llm.model,
        "query_save_path": agent_config.query_save_path,
        "page_view_port_size": agent_config.search.viewport_size,
    }
