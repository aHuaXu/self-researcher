"""Web search tool for Research Agent - 完全复用 handler.py:281-315 的逻辑"""

import json
from typing import List
from langchain_core.tools import tool

# 复用现有的 Agent 类
from scrl.handler.web_search_agent.web_search_agent import WebSearchAgent
from scrl.handler.web_search_agent.search.search_api import web_search as _web_search


# 全局实例缓存
_search_agent = None
_api_result_dict = {}


def get_search_agent():
    """获取或创建 WebSearchAgent 实例"""
    global _search_agent
    if _search_agent is None:
        from research_agent.config import get_config
        config = get_config()
        from openai import OpenAI
        client = OpenAI(
            base_url=config.llm.executor_base_url,
            api_key=config.llm.executor_api_key
        )
        agent_config = {
            'search_engine': config.search.engine,
            'serper_api_key': config.search.serper_api_key,
            'search_top_k': config.search.top_k,
            'search_region': config.search.region,
            'search_lang': config.search.lang,
            'quick_summary_model': config.llm.summary_model,
            'query_save_path': config.query_save_path,
        }
        _search_agent = WebSearchAgent(client=client, config=agent_config)
    return _search_agent


@tool
def web_search(query: List[str]) -> str:
    """
    Search the web for relevant information.
    必须与 handler.py:281-315 的实现逻辑完全一致。

    Args:
        query: The queries to search (list of strings).

    Returns:
        JSON string containing search results with format:
        [{
            "search_query": "...",
            "web_page_info_list": [
                {"title": "...", "url": "...", "quick_summary": "..."},
                ...
            ]
        }, ...]
    """
    global _api_result_dict

    if not query or not isinstance(query, list):
        return json.dumps({"error": "query must be a list"})

    search_query_list = query[:3]  # 只执行前 3 个搜索，与 handler.py:285 一致

    # 先调用搜索 API 填充 api_result_dict
    web_search_agent = get_search_agent()
    for sq in search_query_list:
        if sq not in _api_result_dict:
            organic = _web_search(sq, web_search_agent.config)
            _api_result_dict[sq] = {'organic': organic}

    # 调用 search_web_batch，与 handler.py:286 一致
    user_query = ""  # 需要传入 question
    web_page_info_list_batch = web_search_agent.search_web_batch(
        user_query=user_query,
        search_query_list=search_query_list,
        api_result_dict=_api_result_dict
    )

    # 格式化输出，与 handler.py:300-313 一致
    content = []
    for search_result_info in web_page_info_list_batch:
        search_query = search_result_info.search_query
        ret_web_page_info_list = []
        for web_page_info in search_result_info.web_page_info_list:
            ret_web_page_info_list.append({
                "title": web_page_info.title,
                "url": web_page_info.url,
                "quick_summary": web_page_info.quick_summary
            })
        content.append({
            "search_query": search_query,
            "web_page_info_list": ret_web_page_info_list
        })

    return json.dumps(content, indent=2, ensure_ascii=False)