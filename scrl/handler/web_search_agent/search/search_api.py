import requests
import json
import http.client
import time
import threading
import re
import hashlib


def web_search(query, config):
    if not query:
        raise ValueError("Search query cannot be empty")
    engine = config['search_engine']
    if engine == 'google':
        return serper_google_search(
            query=query,
            serper_api_key=config['serper_api_key'],
            top_k=config['search_top_k'],
            region=config['search_region'],
            lang=config['search_lang']
        )
    elif engine == 'bing':
        return azure_bing_search(
            query=query,
            subscription_key=config['azure_bing_search_subscription_key'],
            mkt=config['azure_bing_search_mkt'],
            top_k=config['search_top_k']
        )
    elif engine == 'duckduckgo':
        return duckduckgo_search(
            query=query,
            top_k=config['search_top_k'],
            region=config.get('search_region', 'us'),
        )
    elif engine == 'searxng':
        return searxng_search(
            query=query,
            top_k=config['search_top_k'],
            lang=config.get('search_lang', 'en'),
            searxng_url=config.get('searxng_url', 'http://localhost:8888'),
        )
    else:
        raise ValueError(f"Unsupported search engine: {engine}")


def azure_bing_search(query, subscription_key, mkt, top_k, depth=0):
    params = {'q': query, 'mkt': mkt, 'count': top_k}
    headers = {'Ocp-Apim-Subscription-Key': subscription_key}

    results = []

    try:
        response = requests.get("https://api.bing.microsoft.com/v7.0/search", headers=headers, params=params)
        json_response = response.json()
        for e in json_response['webPages']['value']:
            results.append({
                "title": e['name'],
                "link": e['url'],
                "snippet": e['snippet']
            })
    except Exception as e:
        print(f"Bing search API error: {e}")
        if depth < 1024:
            time.sleep(1)
            return azure_bing_search(query, subscription_key, mkt, top_k, depth+1)
    return results


def serper_google_search(
        query, 
        serper_api_key,
        top_k,
        region,
        lang,
        depth=0
    ):
    try:
        conn = http.client.HTTPSConnection("google.serper.dev")
        payload = json.dumps({
                "q": query,
                "num": top_k,
                "gl": region,
                "hl": lang,
            })
        headers = {
            'X-API-KEY': serper_api_key,
            'Content-Type': 'application/json'
        }
        conn.request("POST", "/search", payload, headers)
        res = conn.getresponse()
        data = json.loads(res.read().decode("utf-8"))

        if not data:
            raise Exception("The google search API is temporarily unavailable, please try again later.")

        if "organic" not in data:
            raise Exception(f"No results found for query: '{query}'. Use a less specific query.")
        else:
            results = data["organic"]
            print(f"search success for {query}")
            return results
    except Exception as e:
        print(f"Serper search API error: {e}")
        if depth < 3:
            time.sleep(1)
            return serper_google_search(query, serper_api_key, top_k, region, lang, depth=depth+1)
    print(f"search failed for {query}")
    return []


_ddg_semaphore = threading.Semaphore(int(__import__("os").getenv("DDG_MAX_CONCURRENT", "2")))

_REGION_MAP = {
    "us": "us-en", "uk": "uk-en", "cn": "cn-zh", "jp": "jp-jp",
    "kr": "kr-kr", "de": "de-de", "fr": "fr-fr", "ru": "ru-ru",
}


def duckduckgo_search(query, top_k, region="us", depth=0):
    from duckduckgo_search import DDGS
    from duckduckgo_search.exceptions import RatelimitException, TimeoutException

    ddg_region = _REGION_MAP.get(region, "wt-wt")
    qid = _query_fingerprint(query)
    _log_search_event("ddg_start", qid=qid, region=ddg_region, depth=depth, top_k=top_k)
    _ddg_semaphore.acquire()
    try:
        results_raw = DDGS(timeout=20).text(
            query, region=ddg_region, safesearch="off",
            backend="auto", max_results=top_k,
        )
        results = []
        for r in (results_raw or []):
            results.append({
                "title": r.get("title", ""),
                "link": r.get("href", ""),
                "snippet": r.get("body", ""),
            })
        print(f"search success for {query} (duckduckgo, {len(results)} results)")
        _log_search_event("ddg_success", qid=qid, count=len(results), depth=depth)
        return results
    except RatelimitException:
        print(f"DuckDuckGo rate limited for query='{query}', retry after 5s")
        if depth < 3:
            time.sleep(5)
            return duckduckgo_search(query, top_k, region, depth + 1)
    except TimeoutException:
        print(f"DuckDuckGo timeout for query='{query}', retry after 2s")
        if depth < 3:
            time.sleep(2)
            return duckduckgo_search(query, top_k, region, depth + 1)
    except Exception as e:
        print(f"DuckDuckGo search error: {e}")
        _log_search_event("ddg_error", qid=qid, error=repr(e), depth=depth)
        if depth < 3:
            time.sleep(2)
            return duckduckgo_search(query, top_k, region, depth + 1)
    finally:
        _ddg_semaphore.release()
    print(f"search failed for {query} (duckduckgo)")
    return []


import os as _os

_SEARXNG_URL = _os.getenv("SEARXNG_URL", "http://localhost:8888")


def _log_search_event(event: str, **kwargs):
    """Structured log for search path debugging."""
    if _os.getenv("SEARCH_DEBUG_LOG", "1").lower() in ("0", "false", "no"):
        return
    parts = [f"event={event}"]
    for k, v in kwargs.items():
        parts.append(f"{k}={v}")
    print("[search_debug] " + " ".join(parts))


def _query_fingerprint(query: str) -> str:
    q = (query or "").strip()
    return hashlib.md5(q.encode("utf-8")).hexdigest()[:8]


def _get_searxng_engine_priority():
    """
    Parse env var and return strict engine priority list.
    Example:
      SEARXNG_ENGINE_PRIORITY=bing,google
    """
    raw = _os.getenv("SEARXNG_ENGINE_PRIORITY", "").strip()
    if not raw:
        return []
    return [e.strip() for e in raw.split(",") if e.strip()]


def _simplify_query_for_retry(query: str) -> str:
    """Make overly specific query shorter for retry."""
    q = re.sub(r"\s+", " ", (query or "").strip())
    # Keep phrase semantics but drop heavy punctuation that often hurts recall.
    q = re.sub(r"[\"'`()\\[\\]{}]", "", q)
    tokens = q.split()
    if len(tokens) > 10:
        q = " ".join(tokens[:10])
    return q.strip()


def _searxng_request(url, query, top_k, lang, engine=None):
    params = {"q": query, "format": "json", "language": lang}
    if engine:
        params["engines"] = engine
    resp = requests.get(
        f"{url}/search",
        params=params,
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    raw_results = data.get("results", [])
    results = []
    for r in raw_results[:top_k]:
        results.append({
            "title": r.get("title", ""),
            "link": r.get("url", ""),
            "snippet": r.get("content", ""),
        })
    return results, data.get("unresponsive_engines", [])


def searxng_search(query, top_k, lang="en", searxng_url=None, depth=0):
    url = searxng_url or _SEARXNG_URL
    qid = _query_fingerprint(query)
    try:
        engine_priority = _get_searxng_engine_priority()
        # No priority list => use SearXNG default weighting.
        engines_to_try = engine_priority if engine_priority else [None]
        collected_unresponsive = []
        _log_search_event("searxng_start",
                          qid=qid,
                          depth=depth,
                          top_k=top_k,
                          lang=lang,
                          engines="default" if not engine_priority else ",".join(engine_priority))

        for idx, engine in enumerate(engines_to_try):
            _log_search_event("searxng_try", qid=qid, depth=depth, attempt=idx + 1, engine=engine or "default")
            results, unresponsive = _searxng_request(url, query, top_k, lang, engine=engine)
            if unresponsive:
                collected_unresponsive.extend(unresponsive)
            if results:
                if engine:
                    print(f"search success for {query} (searxng:{engine}, {len(results)} results)")
                else:
                    print(f"search success for {query} (searxng, {len(results)} results)")
                _log_search_event("searxng_success",
                                  qid=qid,
                                  depth=depth,
                                  engine=engine or "default",
                                  count=len(results))
                return results

        # Empty-result fallback path:
        # 1) retry with simplified query once
        # 2) fallback to duckduckgo if enabled
        print(f"SearXNG empty results for '{query}', unresponsive_engines={collected_unresponsive}")
        _log_search_event("searxng_empty",
                          qid=qid,
                          depth=depth,
                          unresponsive=repr(collected_unresponsive))

        if depth < 1:
            retry_query = _simplify_query_for_retry(query)
            if retry_query and retry_query != query:
                print(f"SearXNG retry with simplified query: '{retry_query}'")
                _log_search_event("searxng_retry_simplified",
                                  qid=qid,
                                  depth=depth,
                                  retry_qid=_query_fingerprint(retry_query))
                return searxng_search(retry_query, top_k, lang, searxng_url, depth + 1)

        enable_ddg_fallback = _os.getenv("SEARXNG_EMPTY_FALLBACK_DDG", "1").lower() not in ("0", "false", "no")
        if enable_ddg_fallback:
            region = _os.getenv("SEARCH_REGION", "us")
            _log_search_event("fallback_ddg_start", qid=qid, depth=depth, region=region)
            ddg_results = duckduckgo_search(query, top_k, region=region)
            if ddg_results:
                print(f"search success for {query} (searxng->duckduckgo fallback, {len(ddg_results)} results)")
                _log_search_event("fallback_ddg_success", qid=qid, depth=depth, count=len(ddg_results))
                return ddg_results

        print(f"search failed for {query} (searxng empty after fallback)")
        _log_search_event("search_failed_after_fallback", qid=qid, depth=depth)
        return []
    except Exception as e:
        print(f"SearXNG search error for '{query}': {e}")
        _log_search_event("searxng_error", qid=qid, depth=depth, error=repr(e))
        if depth < 3:
            time.sleep(1)
            return searxng_search(query, top_k, lang, searxng_url, depth + 1)
    print(f"search failed for {query} (searxng)")
    return []


if __name__ == "__main__":
    print(searxng_search("test", 5, "en"))