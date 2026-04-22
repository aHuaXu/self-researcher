"""Webpage browsing tool — delegates to scrl.handler ReadingAgent for identical behavior with training."""

import json
from typing import List

from langchain_core.tools import tool

from scrl.handler.webpage import WebPageInfo
from research_agent.tools._state import get_tool_state


@tool
def browse_webpage(url_list: List[str]) -> str:
    """Browse webpages and return extracted content.

    Args:
        url_list: The URLs to browse (should be from previous search results).

    Returns:
        JSON string with browsing results.
    """
    if not url_list or not isinstance(url_list, list):
        return json.dumps({"error": "url_list must be a non-empty list"})

    state = get_tool_state()
    if state.reading_agent is None or state.web_search_agent is None:
        return json.dumps({"error": "ToolState not initialized."})
    if state.action_info is None:
        return json.dumps({"error": "No previous search results. Call web_search first."})

    # Mirrors handler.py:316-339
    read_webpage_list: List[WebPageInfo] = state.reading_agent.read_batch(
        user_query=state.current_question,
        search_result_info_list=state.action_info.search_result_info_list,
        url_list=url_list,
        web_search_agent=state.web_search_agent,
    )

    content = []
    for read_webpage in read_webpage_list:
        information = []
        for page_read_info in read_webpage.page_read_info_list:
            if page_read_info.used:
                continue
            information.append({
                "page_number": page_read_info.page_number,
                "page_summary": page_read_info.page_summary,
            })
            page_read_info.used = True
        content.append({
            "url": read_webpage.url,
            "information": information,
        })

    return json.dumps(content, indent=2, ensure_ascii=False)
