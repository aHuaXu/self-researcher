"""Webpage browsing tool — delegates to scrl.handler ReadingAgent for identical behavior with training."""

import json
from typing import List

from langchain_core.tools import tool

from scrl.handler.webpage import SearchResultInfo, WebPageInfo
from research_agent.tools._state import get_tool_state
from research_agent.tools.context import tool_rollout_message_idx, tool_rollout_user_query


@tool
def browse_webpage(url_list: List[str], goal: str = "") -> str:
    """Browse webpages and return extracted content.

    Args:
        url_list: The URLs to browse (should be from previous search results).
        goal: Optional information goal for this visit (DR-Venus `visit.goal`). When set,
            the reading agent reads the full page once and does a single goal-directed summary.

    Returns:
        JSON string with browsing results.
    """
    if not url_list or not isinstance(url_list, list):
        return json.dumps({"error": "url_list must be a non-empty list"})

    state = get_tool_state()
    if state.reading_agent is None or state.web_search_agent is None:
        return json.dumps({"error": "ToolState not initialized."})

    msg_idx = tool_rollout_message_idx.get()
    if msg_idx is not None and msg_idx in state.per_message_action_info:
        action_info = state.per_message_action_info[msg_idx]
    elif state.action_info is not None:
        action_info = state.action_info
    else:
        action_info = None

    user_query = tool_rollout_user_query.get() or state.current_question
    search_result_info_list = []
    if action_info is not None:
        search_result_info_list.extend(action_info.search_result_info_list)

    cached_urls = {
        webpage.url
        for search_result_info in search_result_info_list
        for webpage in search_result_info.web_page_info_list
    }
    missing_urls = [url for url in url_list if url not in cached_urls]
    if missing_urls:
        # DR-Venus visit(url, goal) directly fetches arbitrary URLs. Keep search-cache reuse as
        # the fast path, but fall back to direct fetch when the model supplies a valid URL that
        # was not present in the latest search result object.
        search_result_info_list.append(
            SearchResultInfo(
                search_query=goal or user_query or "direct_visit",
                web_page_info_list=[
                    WebPageInfo(
                        title=url,
                        url=url,
                        quick_summary="",
                        browser=None,
                        sub_question=goal or user_query or "direct_visit",
                    )
                    for url in missing_urls
                ],
            )
        )

    print(
        f"browse_webpage start: msg_idx={msg_idx}, {len(url_list)} url(s), "
        f"cache_hit={len(url_list) - len(missing_urls)}, direct_fetch={len(missing_urls)}",
        flush=True,
    )
    t0 = __import__("time").time()
    read_webpage_list: List[WebPageInfo] = state.reading_agent.read_batch(
        user_query=user_query,
        search_result_info_list=search_result_info_list,
        url_list=url_list,
        web_search_agent=state.web_search_agent,
        goal=goal,
    )
    elapsed = __import__("time").time() - t0
    fetch_fail = sum(1 for webpage in read_webpage_list if webpage.browser == "error")
    extract_empty = sum(1 for webpage in read_webpage_list if not webpage.page_read_info_list)
    print(
        f"browse_webpage done: msg_idx={msg_idx}, {len(url_list)} url(s), "
        f"{elapsed:.1f}s, fetch_fail={fetch_fail}, extract_empty={extract_empty}",
        flush=True,
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
