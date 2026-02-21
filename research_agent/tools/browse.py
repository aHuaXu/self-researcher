"""Webpage browsing tool for Research Agent - 完全复用 handler.py:316-339 的逻辑"""

import json
from typing import List
from langchain_core.tools import tool

# 复用现有的 Agent 类
from scrl.handler.reading_agent.reading_agent import ReadingAgent
from scrl.handler.web_search_agent.web_search_agent import WebSearchAgent

# 复用 search.py 的全局实例
from research_agent.tools import search as search_module


@tool
def browse_webpage(url_list: List[str]) -> str:
    """
    Browse the webpage and return its content.
    必须与 handler.py:316-339 的实现逻辑完全一致。

    Args:
        url_list: The URLs to browse (must be from previous web_search results).

    Returns:
        JSON string containing browse results with format:
        [{
            "url": "...",
            "information": [
                {"page_number": 0, "page_summary": "..."},
                ...
            ]
        }, ...]
    """
    if not url_list or not isinstance(url_list, list):
        return json.dumps({"error": "url_list must be a list with at least 1 URL"})

    if len(url_list) < 1:
        return json.dumps({"error": "url_list must have at least 1 URL"})

    # 检查是否有 search 结果（与 handler.py:320 一致）
    if not hasattr(search_module, '_api_result_dict') or not search_module._api_result_dict:
        return json.dumps({"error": "No search result found. Please call web_search first."})

    # 获取 search_agent 实例
    web_search_agent = search_module.get_search_agent()

    # 创建 ReadingAgent
    from research_agent.config import get_config
    config = get_config()
    from openai import OpenAI
    client = OpenAI(
        base_url=config.llm.executor_base_url,
        api_key=config.llm.executor_api_key
    )
    reading_agent = ReadingAgent(config={
        'reading_agent_model': config.llm.summary_model,
    }, client=client)

    # 构建 search_result_info_list（从 _api_result_dict 重建）
    from scrl.handler.webpage import SearchResultInfo, WebPageInfo
    search_result_info_list = []
    for query, result in search_module._api_result_dict.items():
        organic = result.get('organic', [])
        web_page_info_list = []
        for web_info in organic:
            web_page_info_list.append(WebPageInfo(
                title=web_info.get('title', ''),
                url=web_info.get('link', ''),
                quick_summary=web_info.get('snippet', ''),
                browser=None,
                sub_question=query
            ))
        search_result_info_list.append(SearchResultInfo(
            search_query=query,
            web_page_info_list=web_page_info_list
        ))

    # 调用 read_batch，与 handler.py:323 一致
    user_query = ""  # 需要传入
    read_webpage_list = reading_agent.read_batch(
        user_query=user_query,
        search_result_info_list=search_result_info_list,
        url_list=url_list,
        web_search_agent=web_search_agent
    )

    # 格式化输出，与 handler.py:324-338 一致
    content = []
    for read_webpage in read_webpage_list:
        information = []
        for page_read_info in read_webpage.page_read_info_list:
            if page_read_info.used:
                continue
            information.append({
                "page_number": page_read_info.page_number,
                "page_summary": page_read_info.page_summary
            })
            page_read_info.used = True
        content.append({
            "url": read_webpage.url,
            "information": information
        })

    return json.dumps(content, indent=2, ensure_ascii=False)