"""Tests for parallel web_search in LLMGenerationManager.execute_predictions."""

import json
import threading
import time
from unittest.mock import MagicMock

import pytest

import research_agent.tools._state as tool_state_mod
from research_agent.tools._state import ToolState
from scrl.llm_agent.generation import LLMGenerationManager


def _handler_config():
    return {
        "search_engine": "google",
        "serper_api_key": "dummy-key",
        "search_top_k": 2,
        "search_region": "us",
        "search_lang": "en",
        "azure_bing_search_subscription_key": "",
        "azure_bing_search_mkt": "en-US",
        "quick_summary_model": "m",
        "reading_agent_model": "m",
        "query_save_path": "/tmp/scrl_tool_query_save_test.json",
        "page_view_port_size": 1024,
    }


@pytest.fixture
def fresh_tool_state():
    """Replace process-wide ToolState with a fresh initialized instance."""
    prev = tool_state_mod._state
    state = ToolState()
    state.initialize(_handler_config(), MagicMock())
    tool_state_mod._state = state
    yield state
    tool_state_mod._state = prev


def test_execute_predictions_parallel_web_search_concurrent_serper(
    monkeypatch, fresh_tool_state
):
    """Multiple web_search in one batch should invoke Serper concurrently (overlap)."""
    import research_agent.tools.search as search_mod

    lock = threading.Lock()
    overlap_max = [0]
    inside = [0]

    def fake_serper(query, config):
        with lock:
            inside[0] += 1
            overlap_max[0] = max(overlap_max[0], inside[0])
        time.sleep(0.08)
        with lock:
            inside[0] -= 1
        return [
            {
                "title": f"t-{query}",
                "link": f"https://example.com/{query}",
                "snippet": "s",
            }
        ]

    monkeypatch.setattr(search_mod, "serper_search", fake_serper)

    tool_call_list = [
        (0, "q0", "think0", {"name": "web_search", "arguments": {"query": ["alpha"]}}),
        (1, "q1", "think1", {"name": "web_search", "arguments": {"query": ["beta"]}}),
        (2, "q2", "think2", {"name": "web_search", "arguments": {"query": ["gamma"]}}),
    ]

    t0 = time.perf_counter()
    out = LLMGenerationManager.execute_predictions(object(), tool_call_list)
    elapsed = time.perf_counter() - t0

    assert len(out) == 3
    for i in range(3):
        assert out[i]["idx"] == tool_call_list[i][0]
        data = json.loads(out[i]["content"])
        assert isinstance(data, list)
        assert data[0]["search_query"] == ["alpha", "beta", "gamma"][i]

    assert overlap_max[0] >= 2, "expected overlapping Serper calls"
    assert elapsed < 0.35, f"expected parallel Serper wall time, got {elapsed:.3f}s"


def test_execute_predictions_order_preserved_with_unknown_tool(monkeypatch, fresh_tool_state):
    """Result list order matches tool_call_list; non-web tools run in index order after web pool."""
    import research_agent.tools.search as search_mod

    def fake_serper(query, config):
        return [{"title": "t", "link": "https://u", "snippet": "s"}]

    monkeypatch.setattr(search_mod, "serper_search", fake_serper)

    tool_call_list = [
        (10, "qA", "t0", {"name": "web_search", "arguments": {"query": ["w0"]}}),
        (11, "qB", "t1", {"name": "web_search", "arguments": {"query": ["w1"]}}),
        (12, "qC", "t2", {"name": "not_a_tool", "arguments": {}}),
    ]

    out = LLMGenerationManager.execute_predictions(object(), tool_call_list)
    assert [r["idx"] for r in out] == [10, 11, 12]
    assert "Unknown tool" in out[2]["content"]


def test_per_message_action_info_isolation(monkeypatch, fresh_tool_state):
    """Each rollout line gets its own ActionInfo.user_query under parallel web_search."""
    import research_agent.tools.search as search_mod

    def fake_serper(query, config):
        return [{"title": "x", "link": "https://x", "snippet": "y"}]

    monkeypatch.setattr(search_mod, "serper_search", fake_serper)

    tool_call_list = [
        (0, "question-zero", "", {"name": "web_search", "arguments": {"query": ["z"]}}),
        (1, "question-one", "", {"name": "web_search", "arguments": {"query": ["z"]}}),
    ]
    LLMGenerationManager.execute_predictions(object(), tool_call_list)

    st = fresh_tool_state
    assert st.per_message_action_info[0].user_query == "question-zero"
    assert st.per_message_action_info[1].user_query == "question-one"


def test_web_then_browse_same_rollout_one_batch(monkeypatch, fresh_tool_state):
    """Same messages_list index: web then browse in one batch resolves per_message_action_info."""
    import research_agent.tools.search as search_mod

    def fake_serper(query, config):
        return [{"title": "t", "link": "https://match.example/x", "snippet": "s"}]

    monkeypatch.setattr(search_mod, "serper_search", fake_serper)

    captured = {}

    def fake_read_batch(self, user_query, search_result_info_list, url_list, web_search_agent=None):
        captured["uq"] = user_query
        captured["n_results"] = len(search_result_info_list)
        return []

    from scrl.handler.reading_agent.reading_agent import ReadingAgent

    monkeypatch.setattr(ReadingAgent, "read_batch", fake_read_batch)

    tool_call_list = [
        (7, "main-q", "", {"name": "web_search", "arguments": {"query": ["qq"]}}),
        (
            7,
            "main-q",
            "",
            {"name": "browse_webpage", "arguments": {"url_list": ["https://match.example/x"]}},
        ),
    ]
    out = LLMGenerationManager.execute_predictions(object(), tool_call_list)
    assert len(out) == 2
    assert captured["uq"] == "main-q"
    assert captured["n_results"] >= 1
