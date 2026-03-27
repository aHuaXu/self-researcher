import importlib.util
import os
import sys

import pytest

_module_path = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "verl", "utils", "reward_score", "rule_reward.py",
)
_spec = importlib.util.spec_from_file_location("rule_reward", os.path.abspath(_module_path))
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

planner_rules = _mod.planner_rules
executor_rules = _mod.executor_rules


# ── Planner Tests (new <plan> format) ──


class TestPlannerRulesEmpty:

    def test_empty_string(self):
        assert planner_rules("") == 0.0

    def test_none_like_whitespace(self):
        assert planner_rules("   \n\n  ") == 0.0

    def test_unparseable_garbage(self):
        assert planner_rules("hello world no tasks here") == 0.0


class TestPlannerRulesTaskCount:

    def test_too_few_tasks(self):
        plan = """<plan>
1. [INDEPENDENT] What is X?
2. [DEPENDS:1] What is Y?
</plan>"""
        score = planner_rules(plan)
        # 2 tasks < min(3), count score = 0
        assert score <= 0.5

    def test_too_many_tasks(self):
        lines = ["<plan>"]
        for i in range(1, 8):
            lines.append(f"{i}. [INDEPENDENT] Unique topic number {i} about something")
        lines.append("</plan>")
        plan = "\n".join(lines)
        score = planner_rules(plan)
        # 7 tasks > max(5), count score = 0
        assert score <= 0.5

    def test_three_tasks_valid(self):
        plan = """<plan>
1. [INDEPENDENT] What country was director X born in?
2. [INDEPENDENT] What year was the film released?
3. [DEPENDS:1,2] Compare the GDP of the country in that year
</plan>"""
        score = planner_rules(plan)
        assert score == 1.0  # 3 tasks in [3,5], no duplicates

    def test_five_tasks_valid(self):
        plan = """<plan>
1. [INDEPENDENT] Who directed the 1970 film Move?
2. [INDEPENDENT] What year was the film Méditerranée released?
3. [DEPENDS:1] What nationality does this director hold?
4. [DEPENDS:2] Which production company distributed the second film?
5. [DEPENDS:3,4] Compare the cultural backgrounds of both films
</plan>"""
        score = planner_rules(plan)
        assert score == 1.0  # 5 tasks in [3,5], no duplicates

    def test_custom_range(self):
        plan = """<plan>
1. [INDEPENDENT] Q1
2. [DEPENDS:1] Q2
</plan>"""
        # With min=2, max=3 this should pass count check
        score = planner_rules(plan, min_tasks=2, max_tasks=3)
        assert score >= 0.5

    def test_custom_range_too_many(self):
        plan = """<plan>
1. [INDEPENDENT] Q1 about topic A
2. [INDEPENDENT] Q2 about topic B
3. [DEPENDS:1] Q3 about topic C
4. [DEPENDS:2] Q4 about topic D
</plan>"""
        # With max=3, 4 tasks exceeds
        score = planner_rules(plan, min_tasks=2, max_tasks=3)
        assert score <= 0.5


class TestPlannerRulesDuplicates:

    def test_duplicate_subtasks(self):
        plan = """<plan>
1. [INDEPENDENT] What is the AI chip market size?
2. [INDEPENDENT] What is the AI chip market growth?
3. [DEPENDS:1,2] Compare market size and growth
</plan>"""
        score = planner_rules(plan)
        # Tasks 1 and 2 have high keyword overlap
        assert score < 1.0

    def test_no_duplicates(self):
        plan = """<plan>
1. [INDEPENDENT] Who directed the film Move?
2. [INDEPENDENT] What year was Méditerranée released?
3. [DEPENDS:1,2] Are both directors from the same country?
</plan>"""
        score = planner_rules(plan)
        assert score == 1.0


class TestPlannerRulesLegacyFormat:
    """Ensure backward compatibility with old [HIGH/MEDIUM/LOW] format."""

    def test_legacy_format_still_works(self):
        plan = (
            "1. [HIGH] AI chip market size\n"
            "2. [MEDIUM] GPU demand trends\n"
            "3. [LOW] Cloud computing growth\n"
        )
        score = planner_rules(plan)
        assert score >= 0.5  # Should parse 3 tasks

    def test_legacy_without_plan_tags(self):
        plan = (
            "1. [HIGH] Sub-topic: AI chips\n"
            "2. [MEDIUM] Sub-topic: Quantum computing\n"
            "3. [LOW] Sub-topic: Edge AI\n"
            "4. [HIGH] Sub-topic: Cloud platforms\n"
        )
        score = planner_rules(plan)
        assert score >= 0.5


class TestPlannerRulesScoring:

    def test_perfect_score(self):
        plan = """<plan>
1. [INDEPENDENT] What seminal literary work explores psychological turmoil?
2. [DEPENDS:1] Which illustrator was nurtured by the publisher of that work?
3. [DEPENDS:2] Which singer-songwriter did the illustrator collaborate with?
4. [DEPENDS:3] What venue shaped the sound of that musician's era?
</plan>"""
        score = planner_rules(plan)
        assert score == 1.0

    def test_zero_score_empty_plan_tags(self):
        plan = "<plan>\n</plan>"
        score = planner_rules(plan)
        assert score == 0.0


# ── Executor Tests ──


class TestExecutorRulesFullSuccess:

    def test_all_criteria_met(self):
        trajectory = [
            {"tool": "web_search", "result": "Found results about AI chips market growing rapidly in 2024"},
            {"tool": "browse_webpage", "result": "The global AI chip market reached $50B in 2024, with NVIDIA holding 80% share..."},
        ]
        score = executor_rules(trajectory, max_turns=5, actual_turns=3)
        assert score == 1.0


class TestExecutorRulesNoSearch:

    def test_no_search_tool(self):
        trajectory = [
            {"tool": "browse_webpage", "result": "Some content from a webpage that is long enough"},
        ]
        score = executor_rules(trajectory, max_turns=5, actual_turns=3)
        assert score < 1.0


class TestExecutorRulesNoBrowse:

    def test_no_browse_tool(self):
        trajectory = [
            {"tool": "web_search", "result": "Found results about AI chips market"},
        ]
        score = executor_rules(trajectory, max_turns=5, actual_turns=3)
        assert score < 1.0


class TestExecutorRulesMaxTurns:

    def test_hit_max_turns(self):
        trajectory = [
            {"tool": "web_search", "result": "Found results about AI chips market growing rapidly"},
            {"tool": "browse_webpage", "result": "The global AI chip market reached $50B in 2024, with NVIDIA holding 80%..."},
        ]
        score = executor_rules(trajectory, max_turns=5, actual_turns=5)
        assert score < 1.0

    def test_within_max_turns(self):
        trajectory = [
            {"tool": "web_search", "result": "Found results about AI chips market growing rapidly"},
            {"tool": "browse_webpage", "result": "The global AI chip market reached $50B in 2024, with NVIDIA holding 80%..."},
        ]
        score = executor_rules(trajectory, max_turns=5, actual_turns=4)
        assert score == 1.0


class TestExecutorRulesEmptyResults:

    def test_empty_search_result(self):
        trajectory = [
            {"tool": "web_search", "result": ""},
            {"tool": "browse_webpage", "result": "The global AI chip market reached $50B in 2024, with NVIDIA holding 80%..."},
        ]
        score = executor_rules(trajectory, max_turns=5, actual_turns=3)
        assert score < 1.0

    def test_short_browse_result(self):
        trajectory = [
            {"tool": "web_search", "result": "Found results about AI chips market growing rapidly"},
            {"tool": "browse_webpage", "result": "short"},
        ]
        score = executor_rules(trajectory, max_turns=5, actual_turns=3)
        assert score < 1.0


class TestExecutorRulesEmptyTrajectory:

    def test_empty_trajectory_within_turns(self):
        score = executor_rules([], max_turns=5, actual_turns=3)
        assert score == 1.0 / 4.0

    def test_empty_trajectory_at_max_turns(self):
        score = executor_rules([], max_turns=5, actual_turns=5)
        assert score == 0.0


class TestExecutorRulesExactScores:

    def test_only_max_turns_credit(self):
        score = executor_rules([], max_turns=10, actual_turns=5)
        assert score == pytest.approx(0.25)

    def test_search_and_turns_only(self):
        trajectory = [
            {"tool": "web_search", "result": "Found relevant results about the topic here"},
        ]
        score = executor_rules(trajectory, max_turns=5, actual_turns=3)
        assert score == pytest.approx(2.0 / 4.0)

    def test_three_of_four(self):
        trajectory = [
            {"tool": "web_search", "result": "Found relevant results about the topic here"},
            {"tool": "browse_webpage", "result": "x" * 51},
        ]
        score = executor_rules(trajectory, max_turns=5, actual_turns=5)
        assert score == pytest.approx(3.0 / 4.0)
