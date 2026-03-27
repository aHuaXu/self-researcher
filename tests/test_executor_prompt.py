"""Unit tests for executor prompt with prior_findings support."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from research_agent.prompts.executor import (
    get_executor_prompt,
    format_prior_findings,
    EXECUTOR_TOOLS,
)


class TestGetExecutorPrompt:
    def test_returns_two_messages(self):
        msgs = get_executor_prompt("What is X?")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_question_in_user_message(self):
        msgs = get_executor_prompt("Who directed Move?")
        assert "Who directed Move?" in msgs[1]["content"]

    def test_system_contains_today_date(self):
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        msgs = get_executor_prompt("Test")
        assert today in msgs[0]["content"]

    def test_no_context_shows_placeholder(self):
        msgs = get_executor_prompt("Test")
        assert "[No research done yet]" in msgs[1]["content"]

    def test_with_context(self):
        msgs = get_executor_prompt("Test", context="Previous search results...")
        assert "Previous search results..." in msgs[1]["content"]
        assert "[No research done yet]" not in msgs[1]["content"]

    def test_without_prior_findings_uses_basic_template(self):
        msgs = get_executor_prompt("Test", prior_findings="")
        assert "Prior Research Findings" not in msgs[1]["content"]

    def test_with_prior_findings_uses_findings_template(self):
        findings_text = "[Finding from sub-question 1]: The director is Stuart Rosenberg"
        msgs = get_executor_prompt("Test", prior_findings=findings_text)
        user_content = msgs[1]["content"]
        assert "Prior Research Findings" in user_content
        assert findings_text in user_content
        assert "Use the prior findings as background knowledge" in user_content

    def test_prior_findings_with_context(self):
        msgs = get_executor_prompt(
            "What is GDP?",
            context="Searched for GDP data...",
            prior_findings="[Finding from sub-question 1]: Country is France",
        )
        user_content = msgs[1]["content"]
        assert "What is GDP?" in user_content
        assert "Searched for GDP data..." in user_content
        assert "Country is France" in user_content

    def test_system_prompt_has_tool_instructions(self):
        msgs = get_executor_prompt("Test")
        system = msgs[0]["content"]
        assert "web search" in system
        assert "webpage browsing" in system

    def test_system_prompt_has_output_format(self):
        msgs = get_executor_prompt("Test")
        system = msgs[0]["content"]
        assert "<think>" in system
        assert "<answer>" in system
        assert "<tool_call>" in system


class TestFormatPriorFindings:
    def test_empty_dict_returns_empty_string(self):
        assert format_prior_findings({}) == ""

    def test_single_finding(self):
        result = format_prior_findings({
            1: {"question": "Who directed Move?", "answer": "Stuart Rosenberg"},
        })
        assert "[Sub-question 1] Who directed Move?" in result
        assert "[Finding]: Stuart Rosenberg" in result

    def test_multiple_findings_sorted(self):
        result = format_prior_findings({
            3: {"question": "What is the GDP?", "answer": "2.7 trillion"},
            1: {"question": "What country?", "answer": "France"},
        })
        blocks = result.split("\n\n")
        assert len(blocks) == 2
        assert "Sub-question 1" in blocks[0]
        assert "France" in blocks[0]
        assert "Sub-question 3" in blocks[1]
        assert "2.7 trillion" in blocks[1]

    def test_includes_both_question_and_answer(self):
        result = format_prior_findings({
            1: {"question": "What is X?", "answer": "It is Y"},
        })
        assert "What is X?" in result
        assert "It is Y" in result

    def test_preserves_finding_content(self):
        result = format_prior_findings({
            1: {"question": "Revenue?", "answer": "$100, 50% growth"},
        })
        assert "$100, 50% growth" in result

    def test_ordering_is_by_index(self):
        findings = {
            5: {"question": "Q5", "answer": "E"},
            2: {"question": "Q2", "answer": "B"},
            4: {"question": "Q4", "answer": "D"},
            1: {"question": "Q1", "answer": "A"},
            3: {"question": "Q3", "answer": "C"},
        }
        result = format_prior_findings(findings)
        blocks = result.split("\n\n")
        assert len(blocks) == 5
        assert "Sub-question 1" in blocks[0]
        assert "Sub-question 2" in blocks[1]
        assert "Sub-question 3" in blocks[2]
        assert "Sub-question 4" in blocks[3]
        assert "Sub-question 5" in blocks[4]


class TestExecutorTools:
    def test_has_two_tools(self):
        assert len(EXECUTOR_TOOLS) == 2

    def test_web_search_tool(self):
        search_tool = EXECUTOR_TOOLS[0]
        assert search_tool["type"] == "function"
        assert search_tool["function"]["name"] == "web_search"
        params = search_tool["function"]["parameters"]
        assert "query" in params["properties"]

    def test_browse_webpage_tool(self):
        browse_tool = EXECUTOR_TOOLS[1]
        assert browse_tool["type"] == "function"
        assert browse_tool["function"]["name"] == "browse_webpage"
        params = browse_tool["function"]["parameters"]
        assert "url_list" in params["properties"]
