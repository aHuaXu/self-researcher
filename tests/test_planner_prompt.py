"""Unit tests for planner prompt and plan parser."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from research_agent.prompts.planner import (
    get_planner_prompt,
    parse_plan,
    DEFAULT_MIN_TASKS,
    DEFAULT_MAX_TASKS,
)


class TestGetPlannerPrompt:
    def test_returns_two_messages(self):
        msgs = get_planner_prompt("What is X?")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_question_in_user_message(self):
        msgs = get_planner_prompt("Who directed Move (1970)?")
        assert "Who directed Move (1970)?" in msgs[1]["content"]

    def test_default_task_range_in_system(self):
        msgs = get_planner_prompt("Test question")
        assert str(DEFAULT_MIN_TASKS) in msgs[0]["content"]
        assert str(DEFAULT_MAX_TASKS) in msgs[0]["content"]

    def test_custom_task_range(self):
        msgs = get_planner_prompt("Test", min_tasks=2, max_tasks=7)
        assert "2" in msgs[0]["content"]
        assert "7" in msgs[0]["content"]
        assert "2-7" in msgs[1]["content"]

    def test_system_contains_format_instructions(self):
        msgs = get_planner_prompt("Test")
        system = msgs[0]["content"]
        assert "<plan>" in system
        assert "[INDEPENDENT]" in system
        assert "[DEPENDS:" in system


class TestParsePlan:
    def test_basic_plan(self):
        text = """<plan>
1. [INDEPENDENT] What country was director X born in?
2. [INDEPENDENT] What year was X's most famous film released?
3. [DEPENDS:1,2] Compare the GDP of the country with film revenue in that year
</plan>"""
        items = parse_plan(text)
        assert len(items) == 3
        assert items[0]["index"] == 1
        assert items[0]["deps"] == []
        assert items[0]["sub_question"] == "What country was director X born in?"
        assert items[0]["is_final"] is False

        assert items[1]["index"] == 2
        assert items[1]["deps"] == []

        assert items[2]["index"] == 3
        assert items[2]["deps"] == [1, 2]
        assert items[2]["is_final"] is True

    def test_single_dependency(self):
        text = """<plan>
1. [INDEPENDENT] Who is the director of Move (1970)?
2. [DEPENDS:1] What nationality is the director?
</plan>"""
        items = parse_plan(text)
        assert len(items) == 2
        assert items[1]["deps"] == [1]

    def test_multiple_dependencies(self):
        text = """<plan>
1. [INDEPENDENT] Q1
2. [INDEPENDENT] Q2
3. [INDEPENDENT] Q3
4. [DEPENDS:1,2,3] Final synthesis
</plan>"""
        items = parse_plan(text)
        assert items[3]["deps"] == [1, 2, 3]

    def test_last_item_is_final(self):
        text = """<plan>
1. [INDEPENDENT] Q1
2. [DEPENDS:1] Q2
3. [DEPENDS:2] Q3
</plan>"""
        items = parse_plan(text)
        assert items[0]["is_final"] is False
        assert items[1]["is_final"] is False
        assert items[2]["is_final"] is True

    def test_no_plan_tags_returns_empty(self):
        text = "Here are some thoughts about the question..."
        items = parse_plan(text)
        assert items == []

    def test_empty_plan_returns_empty(self):
        text = "<plan>\n</plan>"
        items = parse_plan(text)
        assert items == []

    def test_plan_with_surrounding_text(self):
        text = """Let me think about this step by step.

<plan>
1. [INDEPENDENT] First sub-question
2. [DEPENDS:1] Second sub-question
</plan>

That should cover the decomposition."""
        items = parse_plan(text)
        assert len(items) == 2

    def test_case_insensitive_tags(self):
        text = """<plan>
1. [independent] What is X?
2. [depends:1] What is Y based on X?
</plan>"""
        items = parse_plan(text)
        assert len(items) == 2
        assert items[0]["deps"] == []
        assert items[1]["deps"] == [1]

    def test_ignores_malformed_lines(self):
        text = """<plan>
1. [INDEPENDENT] Valid line
This is garbage
2. [DEPENDS:1] Another valid line
- Some bullet point
</plan>"""
        items = parse_plan(text)
        assert len(items) == 2
        assert items[0]["sub_question"] == "Valid line"
        assert items[1]["sub_question"] == "Another valid line"

    def test_realistic_l3_decomposition(self):
        text = """<plan>
1. [INDEPENDENT] What seminal literary work explores psychological turmoil and was initially published by a house that became part of a global conglomerate?
2. [DEPENDS:1] Which illustrator, known for blending whimsy with darkness, was nurtured by this conglomerate during a transformative decade?
3. [DEPENDS:2] Which singer-songwriter, emblematic of a seismic shift in popular music, did this illustrator collaborate with?
4. [DEPENDS:3] What venue, synonymous with intimate performances, shaped the sound of the era associated with this musician?
</plan>"""
        items = parse_plan(text)
        assert len(items) == 4
        assert items[0]["deps"] == []
        assert items[1]["deps"] == [1]
        assert items[2]["deps"] == [2]
        assert items[3]["deps"] == [3]
        assert items[3]["is_final"] is True
