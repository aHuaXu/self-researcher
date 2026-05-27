"""Integration tests verifying data format consistency across the dual-agent pipeline.

These tests verify that the output of one stage matches what the next stage expects:
1. planner_rules() can parse planner output in the new <plan> format
2. executor_rules() expects {tool, result} dicts
3. MultiAgentRewardManager expects specific non_tensor_batch fields
4. MultiAgentResult dataclass fields match what ray_trainer expects
"""

import importlib.util
import os
import re
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Import rule_reward via importlib to avoid collision with mocked verl module
# ---------------------------------------------------------------------------
import importlib.util as _ilu

_rr_path = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "verl", "utils", "reward_score", "rule_reward.py",
)
_rr_spec = _ilu.spec_from_file_location("rule_reward", os.path.abspath(_rr_path))
_rr_mod = _ilu.module_from_spec(_rr_spec)
_rr_spec.loader.exec_module(_rr_mod)
_parse_plan_tasks = _rr_mod._parse_plan_tasks
executor_rules = _rr_mod.executor_rules
planner_rules = _rr_mod.planner_rules

# ---------------------------------------------------------------------------
# Bootstrap: mock heavy imports so we can load multi_agent_generation.py
# ---------------------------------------------------------------------------


class _FakeDataProto:
    def __init__(self):
        self.batch = {}
        self.non_tensor_batch = {}
        self.meta_info = {}

    @classmethod
    def from_dict(cls, tensors, non_tensors=None, meta_info=None, **kw):
        obj = cls()
        obj.batch = tensors
        obj.non_tensor_batch = non_tensors or {}
        obj.meta_info = meta_info or {}
        return obj

    def __len__(self):
        first_key = next(iter(self.batch), None)
        if first_key is not None:
            return self.batch[first_key].shape[0]
        return 0


_verl_mod = sys.modules.get("verl")
_verl_proto_mod = sys.modules.get("verl.protocol")
if _verl_mod is not None:
    _verl_mod.DataProto = _FakeDataProto
if _verl_proto_mod is not None:
    _verl_proto_mod.DataProto = _FakeDataProto
else:
    _fake_verl = ModuleType("verl")
    _fake_verl.DataProto = _FakeDataProto
    _fake_proto = ModuleType("verl.protocol")
    _fake_proto.DataProto = _FakeDataProto
    sys.modules.setdefault("verl", _fake_verl)
    sys.modules.setdefault("verl.protocol", _fake_proto)

_gen_mod = ModuleType("scrl.llm_agent.generation")


class _FakeLLMGenerationManager:
    pass


class _FakeGenerationConfig:
    pass


_gen_mod.LLMGenerationManager = _FakeLLMGenerationManager
_gen_mod.GenerationConfig = _FakeGenerationConfig
sys.modules.setdefault("scrl", ModuleType("scrl"))
sys.modules.setdefault("scrl.llm_agent", ModuleType("scrl.llm_agent"))
sys.modules.setdefault("scrl.llm_agent.generation", _gen_mod)

_ra_mod = ModuleType("research_agent")
_ra_mod.__path__ = [
    os.path.join(os.path.dirname(__file__), os.pardir, "research_agent")
]
sys.modules.setdefault("research_agent", _ra_mod)

_module_path = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    "scrl",
    "llm_agent",
    "multi_agent_generation.py",
)
_spec = importlib.util.spec_from_file_location(
    "multi_agent_generation", os.path.abspath(_module_path)
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

MultiAgentGenerationManager = _mod.MultiAgentGenerationManager
MultiAgentResult = _mod.MultiAgentResult
schedule_waves = _mod.schedule_waves

# Import SubTask from planner prompt
_planner_path = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "research_agent", "prompts", "planner.py",
)
_planner_spec = _ilu.spec_from_file_location("planner_prompt", os.path.abspath(_planner_path))
_planner_mod = _ilu.module_from_spec(_planner_spec)
_planner_spec.loader.exec_module(_planner_mod)
SubTask = _planner_mod.SubTask
parse_plan = _planner_mod.parse_plan


# ===========================================================================
# Issue 1: planner_rules can parse planner's actual output format
# ===========================================================================


class TestPlannerOutputFormatConsistency:
    """Verify planner_rules can parse the planner's actual output format."""

    def test_new_plan_format_parseable(self):
        """The new <plan> format with [INDEPENDENT]/[DEPENDS:N] should be parseable."""
        plan_text = (
            "<plan>\n"
            "1. [INDEPENDENT] What is the capital of France?\n"
            "2. [INDEPENDENT] What is the population of France?\n"
            "3. [DEPENDS:1,2] Summarize the key facts about France.\n"
            "</plan>"
        )
        tasks = _parse_plan_tasks(plan_text)
        assert len(tasks) == 3
        assert tasks[0]["tag"] == "INDEPENDENT"
        assert tasks[2]["tag"] == "DEPENDS:1,2"

    def test_planner_rules_scores_valid_plan(self):
        """planner_rules should return a non-zero score for a valid plan."""
        plan_text = (
            "<plan>\n"
            "1. [INDEPENDENT] What is the capital of France?\n"
            "2. [INDEPENDENT] What is the GDP of Germany?\n"
            "3. [DEPENDS:1,2] Compare the two countries economically.\n"
            "</plan>"
        )
        score = planner_rules(plan_text)
        assert score > 0.0

    def test_parse_plan_matches_parse_plan_tasks(self):
        """parse_plan (generation) and _parse_plan_tasks (reward) should agree."""
        plan_text = (
            "<plan>\n"
            "1. [INDEPENDENT] History of quantum computing\n"
            "2. [DEPENDS:1] Current state of quantum hardware\n"
            "3. [DEPENDS:1,2] Future outlook for quantum computing\n"
            "</plan>"
        )
        parsed_subtasks = parse_plan(plan_text)
        parsed_reward_tasks = _parse_plan_tasks(plan_text)

        assert len(parsed_subtasks) == len(parsed_reward_tasks) == 3


# ===========================================================================
# Issue 2: executor_rules format compatibility
# ===========================================================================


class TestExecutorRulesFormatConsistency:
    """Verify executor_rules with expected trajectory format."""

    def test_executor_rules_with_valid_trajectory(self):
        """executor_rules should score > 0 for a good trajectory."""
        trajectory = [
            {"tool": "web_search", "result": "Found 10 results about AI safety"},
            {"tool": "browse_webpage", "result": "A" * 100},
        ]
        score = executor_rules(trajectory, max_turns=10, actual_turns=3)
        assert score == 1.0

    def test_executor_rules_empty_trajectory(self):
        """Empty trajectory should get turn efficiency bonus only."""
        score = executor_rules([], max_turns=10, actual_turns=3)
        assert score == 0.25  # Only turn efficiency


# ===========================================================================
# Issue 3: MultiAgentResult has all fields the trainer needs
# ===========================================================================


class TestMultiAgentResultFields:
    """Verify MultiAgentResult dataclass has the fields ray_trainer expects."""

    def test_has_required_fields(self):
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(MultiAgentResult)}
        required = {
            "planner_outputs", "executor_outputs", "queries",
            "plan_texts", "parsed_plans", "final_answers",
            "all_findings", "todo_mapping",
        }
        assert required.issubset(field_names)

    def test_no_writer_field(self):
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(MultiAgentResult)}
        assert "writer_outputs" not in field_names


# ===========================================================================
# Issue 4: DAG scheduling matches what executor expects
# ===========================================================================


class TestDAGSchedulingIntegration:
    """Verify schedule_waves output is compatible with executor DAG runner."""

    def test_schedule_waves_independent_tasks(self):
        tasks = [
            SubTask(index=1, sub_question="Q1", deps=[]),
            SubTask(index=2, sub_question="Q2", deps=[]),
            SubTask(index=3, sub_question="Q3", deps=[1, 2], is_final=True),
        ]
        waves = schedule_waves(tasks)
        assert len(waves) == 2
        assert len(waves[0]) == 2  # Q1, Q2 in parallel
        assert len(waves[1]) == 1  # Q3 depends on both

    def test_schedule_waves_fully_sequential(self):
        tasks = [
            SubTask(index=1, sub_question="Q1", deps=[]),
            SubTask(index=2, sub_question="Q2", deps=[1]),
            SubTask(index=3, sub_question="Q3", deps=[2], is_final=True),
        ]
        waves = schedule_waves(tasks)
        assert len(waves) == 3
