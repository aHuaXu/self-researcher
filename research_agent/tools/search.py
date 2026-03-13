"""Web search tool — delegates to scrl.handler for identical behavior with training."""

import json
import time
from typing import List

from langchain_core.tools import tool

from scrl.handler.web_search_agent.search.search_api import web_search as serper_search
from scrl.handler.webpage import WebPageInfo, SearchResultInfo
from scrl.handler.agent_action import ActionInfo
from research_agent.tools._state import get_tool_state


@tool
def web_search(query: List[str]) -> str:
    """Search the web for relevant information from google.

    Args:
        query: The queries to search (list of strings).

    Returns:
        JSON string with search results.
    """
    if not query or not isinstance(query, list):
        return json.dumps({"error": "query must be a non-empty list"})

    state = get_tool_state()
    if state.web_search_agent is None:
        return json.dumps({"error": "ToolState not initialized. Call state.initialize() first."})

    search_query_list = query[:3]

    # Step 1: Call Serper API for each query, with caching
    for sq in search_query_list:
        if sq in state.api_result_dict and len(state.api_result_dict[sq].get('organic', [])) > 0 \
                and (time.time() - state.api_result_dict[sq]['timestamp'] <= 60 * 60 * 24 * 7):
            continue
        config = state.web_search_agent.config
        organic = serper_search(sq, config)
        state.api_result_dict[sq] = {
            "timestamp": time.time(),
            "organic": organic,
        }

    # Step 2: Build WebPageInfo lists
    web_page_info_list_batch = state.web_search_agent.search_web_batch(
        user_query=state.current_question,
        search_query_list=search_query_list,
        api_result_dict=state.api_result_dict,
    )

    # Step 3: Wrap into SearchResultInfo + ActionInfo
    search_result_info_list = [
        SearchResultInfo(
            search_query=search_query_list[j],
            web_page_info_list=web_page_info_list,
        )
        for j, web_page_info_list in enumerate(web_page_info_list_batch)
    ]

    state.action_info = ActionInfo(
        user_query=state.current_question,
        search_thinking="",
        search_query_list=search_query_list,
        search_result_info_list=search_result_info_list,
    )

    # Step 4: Format output
    content = []
    for search_result_info in search_result_info_list:
        ret_web_page_info_list = []
        for wpi in search_result_info.web_page_info_list:
            ret_web_page_info_list.append({
                "title": wpi.title,
                "url": wpi.url,
                "quick_summary": wpi.quick_summary,
            })
        content.append({
            "search_query": search_result_info.search_query,
            "web_page_info_list": ret_web_page_info_list,
        })

    return json.dumps(content, indent=2, ensure_ascii=False)
