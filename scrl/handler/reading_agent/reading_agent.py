import os
import json
from typing import List, Dict, Any
from scrl.handler.utils import (
    get_content_from_tag,
    get_response_from_llm
)
from .prompts import *
from scrl.handler.webpage import *
import time
import random
import html2text
import concurrent.futures


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to max_tokens (tiktoken cl100k_base, like DR-Venus visit). Falls back to a
    char approximation (~4 chars/token) if tiktoken is unavailable."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        toks = enc.encode(text)
        if len(toks) <= max_tokens:
            return text
        return enc.decode(toks[:max_tokens])
    except Exception:
        max_chars = max_tokens * 4
        return text if len(text) <= max_chars else text[:max_chars]


def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract the first valid JSON object from a possibly wrapped LLM response."""
    if not text:
        return {}
    cleaned = text.replace("```json", "```").replace("```JSON", "```").strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned[3:-3].strip()

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(cleaned):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return {}


def format_drvenus_visit_summary(url: str, goal: str, evidence: str, summary: str) -> str:
    return (
        f"The useful information in {url} for user goal {goal} as follows: \n\n"
        f"Evidence in page: \n{evidence.strip()}\n\n"
        f"Summary: \n{summary.strip()}\n\n"
    )


def invalid_visit_extraction(evidence: str, summary: str) -> bool:
    text = f"{evidence}\n{summary}".strip()
    if len(text) < 20:
        return True
    lower = text.lower()
    bad_fragments = (
        "should contain the new information",
        "string",
        "not enough information",
    )
    return any(fragment in lower for fragment in bad_fragments)


class ReadingAgent:
    def __init__(self,
                 config,
                 client):
        self.config = config
        self.client = client
        
    def read(
            self,
            main_question,
            sub_question,
            selected_result_idx: int,
            cur_webpage: WebPageInfo,
            context: List[WebPageInfo] = [],
            web_search_agent = None,
            goal: str = "",
    ):
        if cur_webpage.browser == "error":
            return cur_webpage
        if cur_webpage.browser is None:
            cur_webpage.browser = web_search_agent.scrape_and_check_valid_api(cur_webpage.url)
            if cur_webpage.browser is None:
                cur_webpage.browser = "error"
                print(f"read_page scrape_fail: {cur_webpage.url[:80]}", flush=True)
                return cur_webpage
        context_so_far_prefix = ""
        for webpage in context:
            useful_info = ""
            for page_read_info in webpage.page_read_info_list:
                useful_info += page_read_info.page_summary + "\n\n"
            if len(useful_info):
                context_so_far_prefix += f"<sub_question>{webpage.sub_question}</sub_question>\n<useful_info>{useful_info}</useful_info>\n"

        # DR-Venus-aligned path: when a per-visit `goal` is given, read the FULL page once
        # (truncated to a token budget) and do a SINGLE goal-directed summary, instead of the
        # viewport page-by-page loop. Backward-compatible: no goal -> original viewport behavior.
        if goal:
            return self._read_full_goal(main_question, goal, selected_result_idx,
                                        cur_webpage, context_so_far_prefix)

        cur_useful_info = ""
        total_pages = len(cur_webpage.browser.viewport_pages)
        max_pages = int(os.getenv("READ_PAGE_MAX_PAGES", "3"))
        _url_short = cur_webpage.url[:60]
        print(f"read_page start: {_url_short} ({total_pages} pages)", flush=True)
        _page_t0 = time.time()
        pages_processed = 0
        while cur_webpage.browser.viewport_current_page < total_pages and pages_processed < max_pages:
            context_so_far = ""
            if cur_useful_info:
                context_so_far = context_so_far_prefix + f"<sub_question>{sub_question}</sub_question>\n<useful_info>{cur_useful_info}</useful_info>"
            else:
                context_so_far = context_so_far_prefix
            cur_web_page_content = cur_webpage.browser._state()[1]
            cur_web_page_content = html2text.html2text(cur_web_page_content)
            page_index = cur_webpage.browser.viewport_current_page + 1
            prompt = EXTRACT_NEW_INFO_PROMPT.format(
                main_question=main_question,
                sub_question=sub_question,
                context_so_far=context_so_far.strip(),
                page_index=page_index,
                total_pages=total_pages,
                page_content=cur_web_page_content
            )

            messages = [{"role": "user", "content": prompt}]
            response = get_response_from_llm(
                messages=messages,
                client=self.client,
                model=self.config["reading_agent_model"],
                stream=False
            )
            
            extracted_info = get_content_from_tag(response["content"], "extracted_info", "").strip()
            page_down = get_content_from_tag(response["content"], "page_down", "").strip()
            short_summary = get_content_from_tag(response["content"], "short_summary", "").strip()

            if "yes" in page_down:
                page_down = True
            else:
                page_down = False

            if extracted_info:
                cur_webpage.page_read_info_list.append(
                    PageReadInfo(
                        search_results_idx=selected_result_idx,
                        url=cur_webpage.url,
                        page_title=cur_webpage.title,
                        fetch_res=cur_web_page_content,
                        page_thinking=response["reasoning_content"] if "reasoning_content" in response else "",
                        page_summary=extracted_info,
                        page_number=cur_webpage.browser.viewport_current_page,
                        need_page_down=page_down,
                        used=False,
                    )
                )
                cur_useful_info += extracted_info + "\n\n"

            pages_processed += 1
            if page_down:
                cur_webpage.browser.page_down()
            else:
                break
        _elapsed = time.time() - _page_t0
        print(f"read_page done: {_url_short} ({len(cur_webpage.page_read_info_list)} extracts, {_elapsed:.1f}s)", flush=True)
        return cur_webpage

    def _read_full_goal(self, main_question, goal, selected_result_idx, cur_webpage, context_so_far_prefix):
        """Full-page goal-directed read using the DR-Venus visit-style JSON extractor."""
        _url_short = cur_webpage.url[:60]
        _t0 = time.time()
        full_content = html2text.html2text(cur_webpage.browser.page_content or "")
        max_tokens = int(os.getenv("WEBCONTENT_MAXLENGTH", "32000"))
        full_content = truncate_to_tokens(full_content, max_tokens)
        print(f"read_page(goal) start: {_url_short} (full {len(full_content)} chars, goal-directed)", flush=True)

        contents = [full_content]
        for ratio in (0.7, 0.45):
            next_len = int(len(full_content) * ratio)
            if next_len > 0 and next_len < len(contents[-1]):
                contents.append(full_content[:next_len])

        extracted_info = ""
        raw_len = 0
        max_retries = max(1, int(os.getenv("DRVENUS_VISIT_EXTRACT_RETRIES", "3")))
        max_output_tokens = max(128, int(os.getenv("DRVENUS_VISIT_EXTRACT_MAX_TOKENS", "768")))

        for attempt, content in enumerate(contents[:max_retries], start=1):
            prompt = DRVENUS_VISIT_EXTRACTOR_PROMPT.format(
                webpage_content=content,
                goal=goal,
            )
            messages = [{"role": "user", "content": prompt}]
            response = get_response_from_llm(
                messages=messages,
                client=self.client,
                model=self.config["reading_agent_model"],
                stream=False,
                temperature=0,
                max_tokens=max_output_tokens,
            )
            raw = response.get("content", "") if isinstance(response, dict) else ""
            raw_len = len(raw)
            parsed = extract_json_object(raw)
            evidence = str(parsed.get("evidence", "")).strip() if parsed else ""
            summary = str(parsed.get("summary", "")).strip() if parsed else ""
            if evidence and summary and not invalid_visit_extraction(evidence, summary):
                extracted_info = format_drvenus_visit_summary(cur_webpage.url, goal, evidence, summary)
                print(
                    f"read_page(goal) extractor_ok: {_url_short} "
                    f"attempt={attempt}, raw={raw_len}, evidence={len(evidence)}, summary={len(summary)}",
                    flush=True,
                )
                break
            print(
                f"read_page(goal) extractor_retry: {_url_short} "
                f"attempt={attempt}, raw={raw_len}, parsed={bool(parsed)}, "
                f"evidence={len(evidence)}, summary={len(summary)}",
                flush=True,
            )

        if not extracted_info:
            extracted_info = format_drvenus_visit_summary(
                cur_webpage.url,
                goal,
                "[visit] Failed to extract reliable goal-directed evidence from this webpage.",
                "The webpage content could not be processed into useful information for the requested goal.",
            )
            print(
                f"read_page(goal) extractor_fail: {_url_short} raw={raw_len}",
                flush=True,
            )

        if extracted_info:
            cur_webpage.page_read_info_list.append(
                PageReadInfo(
                    search_results_idx=selected_result_idx,
                    url=cur_webpage.url,
                    page_title=cur_webpage.title,
                    fetch_res=full_content,
                    page_thinking=response.get("reasoning_content", "") if isinstance(response, dict) else "",
                    page_summary=extracted_info,
                    page_number=0,
                    need_page_down=False,
                    used=False,
                )
            )
        print(f"read_page(goal) done: {_url_short} ({len(extracted_info)} chars, {time.time()-_t0:.1f}s)", flush=True)
        return cur_webpage

    def read_batch(
            self,
            user_query: str,
            search_result_info_list: List[SearchResultInfo],
            url_list: List[str],
            web_search_agent = None,
            goal: str = "",
    ):
        url_dict = {}
        for url in url_list:
            url_dict[url] = []
        future_to_content = []
        # Many parallel LLM calls share one OpenAI client → bursts cause timeouts / connection errors
        # against external APIs (e.g. MiniMax). Tune with READING_AGENT_MAX_WORKERS (default 8).
        max_workers = max(1, int(os.getenv("READING_AGENT_MAX_WORKERS", "8")))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for search_result_info in search_result_info_list:
                search_query = search_result_info.search_query
                web_page_info_list = search_result_info.web_page_info_list
                for selected_result_idx, cur_webpage in enumerate(web_page_info_list):
                    if cur_webpage.url not in url_dict:
                        continue
                    future = executor.submit(self.read,
                                            user_query,
                                            search_query,
                                            selected_result_idx,
                                            cur_webpage,
                                            web_page_info_list,
                                            web_search_agent,
                                            goal)
                    future_to_content.append(future)
        read_webpage_list = []
        for i, future in enumerate(future_to_content):
            cur_webpage: WebPageInfo = future.result()
            read_webpage_list.append(cur_webpage)
        return read_webpage_list

                
                
