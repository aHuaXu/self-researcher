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


# ── Planner Tests ──


class TestPlannerRulesEmpty:

    def test_empty_string(self):
        assert planner_rules("") == 0.0

    def test_none_like_whitespace(self):
        assert planner_rules("   \n\n  ") == 0.0

    def test_unparseable_garbage(self):
        assert planner_rules("hello world no tasks here") == 0.0


class TestPlannerRulesTaskCount:

    def test_too_few_tasks(self):
        plan = (
            "1. [HIGH] Sub-topic: AI chips\n"
            "   Search Query: AI chip market\n"
            "2. [LOW] Sub-topic: GPU demand\n"
            "   Search Query: GPU demand 2024\n"
        )
        score = planner_rules(plan)
        assert score < 0.8

    def test_too_many_tasks(self):
        lines = []
        for i in range(1, 9):
            lines.append(f"{i}. [HIGH] Sub-topic: topic {i}\n   Search Query: query {i}")
        plan = "\n".join(lines)
        score = planner_rules(plan)
        assert score < 0.8

    def test_three_tasks_valid(self):
        plan = (
            "1. [HIGH] Sub-topic: AI chips\n"
            "   Search Query: AI chip market\n"
            "2. [MEDIUM] Sub-topic: GPU demand\n"
            "   Search Query: GPU demand 2024\n"
            "3. [LOW] Sub-topic: Cloud computing\n"
            "   Search Query: cloud computing growth\n"
        )
        score = planner_rules(plan)
        assert score >= 0.8

    def test_seven_tasks_valid(self):
        topics = ["AI chips", "GPU market", "Cloud computing", "Edge AI",
                  "Data centers", "Quantum computing", "5G infrastructure"]
        priorities = ["HIGH", "HIGH", "MEDIUM", "MEDIUM", "LOW", "LOW", "MEDIUM"]
        lines = []
        for i, (topic, pri) in enumerate(zip(topics, priorities), 1):
            lines.append(f"{i}. [{pri}] Sub-topic: {topic}\n   Search Query: {topic} market 2024")
        plan = "\n".join(lines)
        score = planner_rules(plan)
        assert score >= 0.8


class TestPlannerRulesPriority:

    def test_all_same_priority(self):
        plan = (
            "1. [HIGH] Sub-topic: AI chips\n"
            "   Search Query: AI chip market\n"
            "2. [HIGH] Sub-topic: GPU demand\n"
            "   Search Query: GPU demand 2024\n"
            "3. [HIGH] Sub-topic: Cloud computing\n"
            "   Search Query: cloud computing growth\n"
        )
        score = planner_rules(plan)
        assert score < 1.0

    def test_two_distinct_priorities(self):
        plan = (
            "1. [HIGH] Sub-topic: AI chips\n"
            "   Search Query: AI chip market\n"
            "2. [MEDIUM] Sub-topic: GPU demand\n"
            "   Search Query: GPU demand 2024\n"
            "3. [HIGH] Sub-topic: Cloud computing\n"
            "   Search Query: cloud computing growth\n"
        )
        score = planner_rules(plan)
        assert score >= 0.8

    def test_three_distinct_priorities(self):
        plan = (
            "1. [HIGH] Sub-topic: AI chips\n"
            "   Search Query: AI chip market\n"
            "2. [MEDIUM] Sub-topic: GPU demand\n"
            "   Search Query: GPU demand 2024\n"
            "3. [LOW] Sub-topic: Cloud computing\n"
            "   Search Query: cloud computing growth\n"
        )
        score = planner_rules(plan)
        assert score >= 0.8


class TestPlannerRulesDuplicates:

    def test_duplicate_subtopics(self):
        plan = (
            "1. [HIGH] Sub-topic: AI chip market size\n"
            "   Search Query: AI chip market size 2024\n"
            "2. [MEDIUM] Sub-topic: AI chip market growth\n"
            "   Search Query: AI chip market growth forecast\n"
            "3. [LOW] Sub-topic: Cloud computing\n"
            "   Search Query: cloud computing growth\n"
        )
        score = planner_rules(plan)
        assert score < 1.0

    def test_no_duplicates(self):
        plan = (
            "1. [HIGH] Sub-topic: AI chip market\n"
            "   Search Query: AI chip market size\n"
            "2. [MEDIUM] Sub-topic: Quantum computing\n"
            "   Search Query: quantum computing progress\n"
            "3. [LOW] Sub-topic: Cloud infrastructure\n"
            "   Search Query: cloud infrastructure growth\n"
        )
        score = planner_rules(plan)
        assert score >= 0.8


class TestPlannerRulesChinese:

    def test_chinese_labels(self):
        plan = (
            "1. [HIGH] 子主题: AI芯片市场规模\n"
            "   搜索查询: AI chip market size 2024\n"
            "2. [MEDIUM] 子主题: 主要玩家竞争格局\n"
            "   搜索查询: NVIDIA AMD Intel AI chip competition\n"
            "3. [LOW] 子主题: 未来发展趋势\n"
            "   搜索查询: AI chip future trend\n"
        )
        score = planner_rules(plan)
        assert score >= 0.8


class TestPlannerRulesGoodPlan:

    def test_perfect_plan(self):
        plan = (
            "1. [HIGH] Sub-topic: AI chip market size\n"
            "   Search Query: AI chip market size 2024\n"
            "2. [HIGH] Sub-topic: NVIDIA competitive advantage\n"
            "   Search Query: NVIDIA AMD Intel AI chip competition\n"
            "3. [MEDIUM] Sub-topic: China domestic chip development\n"
            "   Search Query: China AI chip Huawei Ascend\n"
            "4. [MEDIUM] Sub-topic: Edge AI processor trends\n"
            "   Search Query: edge AI chip mobile deployment\n"
            "5. [LOW] Sub-topic: Quantum computing impact\n"
            "   Search Query: quantum computing AI chip disruption\n"
        )
        score = planner_rules(plan)
        assert 0.8 <= score <= 1.0


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

    def test_short_search_result(self):
        trajectory = [
            {"tool": "web_search", "result": "no data"},
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
