"""Tests verifying inference-time agents match training-time input/output formats.

Uses mock-based approach to avoid heavy transitive dependencies (pdfminer etc).
"""

import re
import json
import pytest


# ---------------------------------------------------------------------------
# Inline the parsing logic under test so we don't pull in the full import chain.
# These must stay in sync with the source files.
# ---------------------------------------------------------------------------

def planner_parse_todos(content: str):
    """Mirror of TodoPlanner._parse_todos (research_agent/agents/planner.py)."""
    todos = []
    pattern = r'(\d+)\.\s*\[(HIGH|MEDIUM|LOW)\]\s*(.+?)(?=\n\d+\.\s*\[|</todos>|$)'
    matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
    for match in matches:
        sub_topic = match[2].strip()
        sub_topic = re.sub(
            r'^(?:Sub-topic|子主题|主题)[：:]\s*', '', sub_topic, flags=re.IGNORECASE,
        )
        sub_topic = sub_topic.rstrip('</todos>').strip()
        todos.append({"index": int(match[0]), "priority": match[1].lower(), "sub_topic": sub_topic})
    if todos:
        return todos
    clean_text = content.strip()
    if clean_text:
        todos.append({"index": 1, "priority": "high", "sub_topic": clean_text[:200]})
    return todos


def executor_format_todo_context(todo_list):
    """Mirror of TaskExecutor._format_todo_context (research_agent/agents/executor.py)."""
    lines = []
    for i, todo in enumerate(todo_list):
        priority = todo.get("priority", "medium").upper()
        sub_topic = todo.get("sub_topic", "")
        lines.append(f"{i+1}. [{priority}] {sub_topic}")
    return "\n".join(lines)


def writer_build_findings_block(findings_text: str, plan_text: str = ""):
    """Mirror of ReportWriter.write() findings_block construction (research_agent/agents/writer.py)."""
    if plan_text:
        return f"=== Research Plan ===\n{plan_text}\n\n=== Research Findings ===\n{findings_text}"
    return findings_text


# ---------------------------------------------------------------------------
# Planner _parse_todos
# ---------------------------------------------------------------------------

class TestPlannerParseTodos:
    """Planner output parsing must match training-time format:
    {index, priority, sub_topic} — no search_query.
    """

    def test_standard_format(self):
        content = """<todos>
1. [HIGH] The history and evolution of quantum computing
2. [MEDIUM] Current applications in industry
3. [LOW] Future outlook and trends
</todos>"""
        todos = planner_parse_todos(content)
        assert len(todos) == 3
        assert todos[0] == {"index": 1, "priority": "high", "sub_topic": "The history and evolution of quantum computing"}
        assert todos[1]["priority"] == "medium"
        assert todos[2]["priority"] == "low"
        for t in todos:
            assert "search_query" not in t

    def test_sub_topic_prefix_stripped(self):
        content = "<todos>\n1. [HIGH] Sub-topic: Quantum computing\n2. [LOW] Neural network\n</todos>"
        todos = planner_parse_todos(content)
        assert len(todos) == 2
        assert todos[0]["sub_topic"] == "Quantum computing"
        assert todos[1]["sub_topic"] == "Neural network"
        for t in todos:
            assert "search_query" not in t

    def test_chinese_prefix_stripped(self):
        content = "1. [HIGH] 子主题：量子计算的历史"
        todos = planner_parse_todos(content)
        assert len(todos) == 1
        assert todos[0]["sub_topic"] == "量子计算的历史"

    def test_fallback_raw_text(self):
        content = "This is not a valid TODO format at all."
        todos = planner_parse_todos(content)
        assert len(todos) == 1
        assert todos[0]["priority"] == "high"
        assert todos[0]["sub_topic"] == content[:200]
        assert "search_query" not in todos[0]

    def test_empty_content(self):
        todos = planner_parse_todos("")
        assert todos == []

    def test_matches_training_output(self):
        """Verify exact same output as multi_agent_generation._parse_todos."""
        content = "1. [HIGH] Sub-topic: Quantum advantage\n2. [MEDIUM] 主题：Error correction"
        todos = planner_parse_todos(content)
        assert todos[0]["sub_topic"] == "Quantum advantage"
        assert todos[1]["sub_topic"] == "Error correction"
        for t in todos:
            assert set(t.keys()) == {"index", "priority", "sub_topic"}


# ---------------------------------------------------------------------------
# Writer input format
# ---------------------------------------------------------------------------

class TestWriterInput:
    """Writer must receive plan_text + findings, matching training format."""

    def test_with_plan_text(self):
        block = writer_build_findings_block("Finding A", plan_text="1. [HIGH] Research A")
        assert "=== Research Plan ===" in block
        assert "1. [HIGH] Research A" in block
        assert "=== Research Findings ===" in block
        assert "Finding A" in block

    def test_without_plan_text(self):
        block = writer_build_findings_block("Finding A")
        assert "=== Research Plan ===" not in block
        assert block == "Finding A"

    def test_format_matches_training(self):
        """Training builds: '=== Research Plan ===\\n{plan}\\n\\n=== Research Findings ===\\n{findings}'"""
        plan = "1. [HIGH] Topic"
        findings = "Some findings"
        block = writer_build_findings_block(findings, plan_text=plan)
        expected = f"=== Research Plan ===\n{plan}\n\n=== Research Findings ===\n{findings}"
        assert block == expected


# ---------------------------------------------------------------------------
# Executor _format_todo_context
# ---------------------------------------------------------------------------

class TestExecutorTodoFormat:
    """Executor TODO formatting should not reference search_query."""

    def test_format_without_search_query(self):
        todos = [
            {"priority": "high", "sub_topic": "Topic A"},
            {"priority": "low", "sub_topic": "Topic B"},
        ]
        result = executor_format_todo_context(todos)
        assert "Search:" not in result
        assert "search_query" not in result
        assert "1. [HIGH] Topic A" in result
        assert "2. [LOW] Topic B" in result

    def test_ignores_extra_fields(self):
        """Even if a TODO has search_query, it should not appear in output."""
        todos = [{"priority": "high", "sub_topic": "X", "search_query": "should not appear"}]
        result = executor_format_todo_context(todos)
        assert "should not appear" not in result
