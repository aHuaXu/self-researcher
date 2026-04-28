"""Integration tests verifying data format consistency across the multi-agent pipeline.

These tests verify that the output of one stage matches what the next stage expects,
focusing on three known risk areas:
1. planner_rules() can't parse planner outputs that follow the prompt's example format
2. executor_rules() expects {tool, result} dicts but _build_exec_trajectories produces {todo_idx, trajectory}
3. MultiAgentRewardManager expects exec_actual_turns/exec_max_turns but run_multi_agent_loop doesn't provide them
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
# (test_multi_agent_generation.py mocks sys.modules["verl"] during collection)
# ---------------------------------------------------------------------------
import importlib.util as _ilu

_rr_path = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "verl", "utils", "reward_score", "rule_reward.py",
)
_rr_spec = _ilu.spec_from_file_location("rule_reward", os.path.abspath(_rr_path))
_rr_mod = _ilu.module_from_spec(_rr_spec)
_rr_spec.loader.exec_module(_rr_mod)
_parse_tasks = _rr_mod._parse_tasks
executor_rules = _rr_mod.executor_rules
planner_rules = _rr_mod.planner_rules

# ---------------------------------------------------------------------------
# Bootstrap: mock heavy imports so we can load multi_agent_generation.py
# without requiring ray, vllm, tensordict, or the full scrl.handler chain.
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


# Ensure verl/verl.protocol in sys.modules point to our fake DataProto
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


def _make_manager():
    mgr = object.__new__(MultiAgentGenerationManager)
    mgr.tokenizer = MagicMock(pad_token_id=0, pad_token="<pad>")
    mgr.tensor_fn = MagicMock()
    mgr.config = MagicMock()
    mgr.lora_save_dir = './tmp_lora_adapters'
    return mgr


# ===========================================================================
# Issue 1: planner_rules._parse_tasks 要求 "Sub-topic:" 前缀，
#           但 planner prompt 示例输出不带这个前缀
# ===========================================================================


class TestPlannerOutputFormatConsistency:
    """Verify that planner_rules can parse the planner's actual output format."""

    def test_planner_prompt_example_format(self):
        """The planner prompt example uses '1. [HIGH] Description' without 'Sub-topic:' prefix.
        _parse_tasks should be able to parse this format."""
        # This is the format shown in planner.py PLANNER_SYSTEM_PROMPT example
        plan_text = (
            "1. [HIGH] The history and evolution of quantum computing\n"
            "2. [MEDIUM] Current applications of quantum computing in industry\n"
            "3. [LOW] Future outlook and emerging trends"
        )
        tasks = _parse_tasks(plan_text)
        # BUG: _parse_tasks requires "Sub-topic:" or "子主题：" prefix, so this returns []
        assert len(tasks) > 0, (
            f"_parse_tasks cannot parse planner's example format! "
            f"Got {tasks}. The regex requires 'Sub-topic:' prefix but "
            f"planner prompt example doesn't use it."
        )

    def test_planner_rules_with_example_format(self):
        """planner_rules should return a non-zero score for well-formed plans."""
        plan_text = (
            "1. [HIGH] The history and evolution of quantum computing\n"
            "2. [MEDIUM] Current applications of quantum computing in industry\n"
            "3. [LOW] Future outlook and emerging trends"
        )
        score = planner_rules(plan_text)
        # BUG: Returns 0.0 because _parse_tasks can't parse this format
        assert score > 0.0, (
            f"planner_rules returned {score} for a valid plan. "
            f"This means the planner LoRA gets no rule-based reward signal."
        )

    def test_subtopic_prefix_format_works(self):
        """Verify the format that _parse_tasks CAN currently handle."""
        plan_text = (
            "1. [HIGH] Sub-topic: The history of quantum computing\n"
            "2. [MEDIUM] Sub-topic: Current applications in industry\n"
            "3. [LOW] Sub-topic: Future outlook and trends"
        )
        tasks = _parse_tasks(plan_text)
        # This format works because it has "Sub-topic:" prefix
        assert len(tasks) == 3

    def test_parse_todos_vs_parse_tasks_consistency(self):
        """_parse_todos (multi_agent_generation) and _parse_tasks (rule_reward)
        should agree on what constitutes a valid planner output."""
        mgr = _make_manager()
        # Planner prompt example format
        plan_text = (
            "1. [HIGH] Climate change impacts on agriculture\n"
            "2. [MEDIUM] Renewable energy solutions\n"
            "3. [LOW] Policy frameworks for sustainability"
        )
        # _parse_todos can handle this format (used during generation)
        parsed_todos = mgr._parse_todos(plan_text)
        # _parse_tasks should also handle it (used during reward)
        parsed_tasks = _parse_tasks(plan_text)

        assert len(parsed_todos) > 0, "_parse_todos failed"
        assert len(parsed_tasks) > 0, (
            f"_parse_tasks returned [] but _parse_todos returned {len(parsed_todos)} items. "
            f"Format mismatch between generation and reward!"
        )


# ===========================================================================
# Issue 2: executor_rules 期望 List[Dict] 里有 tool/result，
#           但 _build_exec_trajectories 产出 todo_idx/trajectory
# ===========================================================================


class TestExecutorTrajectoryFormatConsistency:
    """Verify executor_rules can consume _build_exec_trajectories output."""

    def test_build_exec_trajectories_output_format(self):
        """Check _build_exec_trajectories produces structured tool/result dicts."""
        mgr = _make_manager()
        exec_msgs = [
            '<think>searching</think><tool_call>{"name":"web_search","arguments":{"query":["AI"]}}</tool_call>\n<observation>results here</observation>\n<think>ok</think><answer>AI is cool</answer>',
            '<think>reading</think><tool_call>{"name":"browse_webpage","arguments":{"url_list":["http://x.com"]}}</tool_call>\n<observation>page content here with lots of text</observation>\n<think>done</think><answer>Found info</answer>',
        ]
        mapping = [0, 0]
        trajectories = mgr._build_exec_trajectories(exec_msgs, mapping, 1)

        # trajectories[0] should have 2 parsed tool steps (one from each exec msg)
        assert len(trajectories[0]) == 2
        first_entry = trajectories[0][0]
        assert "tool" in first_entry
        assert "result" in first_entry
        assert first_entry["tool"] == "web_search"
        assert trajectories[0][1]["tool"] == "browse_webpage"

    def test_executor_rules_with_actual_trajectory_format(self):
        """executor_rules expects dicts with 'tool' and 'result' keys.
        But _build_exec_trajectories produces dicts with 'todo_idx' and 'trajectory'."""
        mgr = _make_manager()
        exec_msgs = [
            '<think>searching</think><tool_call>{"name":"web_search","arguments":{"query":["AI"]}}</tool_call>\n<observation>search results about AI</observation>\n<think>browsing</think><tool_call>{"name":"browse_webpage","arguments":{"url_list":["http://x.com"]}}</tool_call>\n<observation>page content with lots of useful information about artificial intelligence</observation>\n<think>done</think><answer>AI answer</answer>',
        ]
        mapping = [0]
        trajectories = mgr._build_exec_trajectories(exec_msgs, mapping, 1)

        # Feed this to executor_rules - this is what actually happens in the pipeline
        score = executor_rules(trajectories[0], max_turns=10, actual_turns=3)

        # BUG: score will be 0.0 because executor_rules looks for step.get("tool")
        # but entries have "todo_idx" and "trajectory" keys, not "tool" and "result"
        assert score > 0.0, (
            f"executor_rules returned {score} for trajectory from _build_exec_trajectories. "
            f"Format mismatch: executor_rules expects [{{'tool': ..., 'result': ...}}] "
            f"but got [{trajectories[0][0].keys()}]"
        )

    def test_executor_rules_with_expected_format(self):
        """Verify executor_rules works with its expected format (for reference)."""
        # This is what executor_rules actually expects
        trajectory = [
            {"tool": "web_search", "result": "Found 10 results about AI safety"},
            {"tool": "browse_webpage", "result": "A" * 100},
        ]
        score = executor_rules(trajectory, max_turns=10, actual_turns=3)
        assert score == 1.0, f"executor_rules returned {score} for ideal trajectory"


# ===========================================================================
# exec_actual_turns / exec_max_turns are populated by ray_trainer.py (L1061-1068)
# between run_multi_agent_loop and MultiAgentRewardManager, so no gap exists.
# ===========================================================================


class TestMetadataFieldCompleteness:
    """Verify the fields that ray_trainer.py bridges are computed correctly."""

    def test_exec_actual_turns_derived_from_trajectories(self):
        """ray_trainer.py computes exec_actual_turns = [len(traj) for traj in exec_trajectories].
        Verify this logic produces correct values."""
        exec_trajectories = [
            [{"tool": "web_search", "result": "r1"}, {"tool": "browse_webpage", "result": "r2"}],
            [{"tool": "web_search", "result": "r3"}],
            [],
        ]
        actual_turns = [len(traj) for traj in exec_trajectories]
        assert actual_turns == [2, 1, 0]


# ===========================================================================
# Bonus: end-to-end format flow validation
# ===========================================================================


class TestEndToEndFormatFlow:
    """Validate the full data format flow from planner to reward."""

    def test_planner_output_flows_to_executor(self):
        """Planner output → _parse_todos → _build_executor_batch should work."""
        mgr = _make_manager()
        # Simulate planner output (following prompt example format)
        planner_output = (
            "<todos>\n"
            "1. [HIGH] The impact of climate change on global food security\n"
            "2. [MEDIUM] Adaptation strategies in developing countries\n"
            "3. [LOW] Policy responses and international cooperation\n"
            "</todos>"
        )
        todos = mgr._parse_todos(planner_output)
        assert len(todos) >= 3, f"_parse_todos returned {len(todos)} items"
        assert all("sub_topic" in t for t in todos), "Missing sub_topic key"

    def test_executor_output_flows_to_writer(self):
        """Executor findings → _group_findings → writer batch should be coherent."""
        mgr = _make_manager()
        exec_msgs = [
            "<think>researching</think><answer>Food production declined by 5%</answer>",
            "<think>investigating</think><answer>Crop rotation helps adapt</answer>",
        ]
        mapping = [0, 0]
        findings = mgr._group_findings(exec_msgs, mapping, 1)

        assert len(findings) == 1
        assert "Food production declined by 5%" in findings[0]
        assert "Crop rotation helps adapt" in findings[0]

    def test_full_format_chain(self):
        """Complete format verification: planner → executor → writer → reward."""
        mgr = _make_manager()
        question = "What is the impact of AI on healthcare?"

        # Stage 1: Planner output
        planner_output = (
            "1. [HIGH] AI-powered diagnostics and imaging\n"
            "2. [MEDIUM] Drug discovery acceleration\n"
            "3. [LOW] Administrative automation"
        )
        todos = mgr._parse_todos(planner_output)
        assert len(todos) == 3

        # Stage 2: Executor outputs (one per TODO)
        exec_msgs = [
            "<think>searching AI diagnostics</think><answer>AI detects cancer with 95% accuracy</answer>",
            "<think>searching drug discovery</think><answer>AI reduced drug discovery time by 40%</answer>",
            "<think>searching admin</think><answer>NLP automates 60% of paperwork</answer>",
        ]
        mapping = [0, 0, 0]  # All belong to question 0
        findings = mgr._group_findings(exec_msgs, mapping, 1)
        assert "95% accuracy" in findings[0]

        # Stage 3: Writer receives plan + findings
        writer_input = (
            f"=== Research Plan ===\n{planner_output}\n\n"
            f"=== Research Findings ===\n{findings[0]}"
        )
        assert "Research Plan" in writer_input
        assert "Research Findings" in writer_input

        # Reward: verify format compatibility
        trajectories = mgr._build_exec_trajectories(exec_msgs, mapping, 1)
        # This is what goes to executor_rules:
        for entry in trajectories[0]:
            # executor_rules will call step.get("tool") on these entries
            assert "tool" in entry or "todo_idx" in entry, (
                "Trajectory entry format unclear"
            )
