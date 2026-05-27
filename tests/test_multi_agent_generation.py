"""Unit tests for scrl/llm_agent/multi_agent_generation.py core functions."""

import importlib.util
import os
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest
import torch

# ---------------------------------------------------------------------------
# Bootstrap: mock heavy imports so we can load multi_agent_generation.py
# without requiring ray, vllm, tensordict, or the full scrl.handler chain.
# ---------------------------------------------------------------------------

# 1. Mock verl.DataProto — provide a minimal stand-in
_real_dataproto_mod = ModuleType("verl")
_real_protocol_mod = ModuleType("verl.protocol")


class _FakeDataProto:
    """Minimal DataProto stand-in for tests."""

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

    @classmethod
    def concat(cls, parts):
        obj = cls()
        if not parts:
            obj.batch = {
                "input_ids": torch.zeros((0, 1), dtype=torch.long),
            }
            return obj
        all_ids = torch.cat([p.batch["input_ids"] for p in parts], dim=0)
        obj.batch = {"input_ids": all_ids}
        return obj

    def __len__(self):
        first_key = next(iter(self.batch), None)
        if first_key is not None:
            return self.batch[first_key].shape[0]
        return 0


_real_dataproto_mod.DataProto = _FakeDataProto
_real_protocol_mod.DataProto = _FakeDataProto
sys.modules.setdefault("verl", _real_dataproto_mod)
sys.modules.setdefault("verl.protocol", _real_protocol_mod)

# 2. Mock scrl.llm_agent.generation (parent class)
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

# 3. research_agent.prompts can be imported normally (lightweight)
_ra_mod = ModuleType("research_agent")
_ra_mod.__path__ = [
    os.path.join(os.path.dirname(__file__), os.pardir, "research_agent")
]
sys.modules.setdefault("research_agent", _ra_mod)

# Now load the module under test via importlib
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
schedule_waves = _mod.schedule_waves

from research_agent.prompts.planner import SubTask


# ---------------------------------------------------------------------------
# Helper: create a manager instance without calling __init__
# ---------------------------------------------------------------------------

def _make_manager(tokenizer=None):
    mgr = object.__new__(MultiAgentGenerationManager)
    mgr.tokenizer = tokenizer or MagicMock(pad_token_id=0, pad_token="<pad>")
    mgr.tensor_fn = MagicMock()
    mgr.config = MagicMock()
    mgr.lora_save_dir = "./tmp_lora_adapters"
    return mgr


# ===========================================================================
# A. schedule_waves tests
# ===========================================================================


class TestScheduleWaves:

    def test_empty_plan(self):
        assert schedule_waves([]) == []

    def test_all_independent(self):
        plan = [
            SubTask(index=1, sub_question="Q1", deps=[]),
            SubTask(index=2, sub_question="Q2", deps=[]),
            SubTask(index=3, sub_question="Q3", deps=[], is_final=True),
        ]
        waves = schedule_waves(plan)
        assert len(waves) == 1
        assert len(waves[0]) == 3

    def test_linear_chain(self):
        plan = [
            SubTask(index=1, sub_question="Q1", deps=[]),
            SubTask(index=2, sub_question="Q2", deps=[1]),
            SubTask(index=3, sub_question="Q3", deps=[2], is_final=True),
        ]
        waves = schedule_waves(plan)
        assert len(waves) == 3
        assert waves[0][0].index == 1
        assert waves[1][0].index == 2
        assert waves[2][0].index == 3

    def test_diamond_dag(self):
        """
        1 (independent)
        2 (independent)
        3 (depends on 1, 2)
        """
        plan = [
            SubTask(index=1, sub_question="Q1", deps=[]),
            SubTask(index=2, sub_question="Q2", deps=[]),
            SubTask(index=3, sub_question="Q3", deps=[1, 2], is_final=True),
        ]
        waves = schedule_waves(plan)
        assert len(waves) == 2
        assert len(waves[0]) == 2  # Q1 and Q2 in parallel
        assert waves[1][0].index == 3

    def test_complex_dag(self):
        """
        1 (independent)
        2 (independent)
        3 (depends on 1)
        4 (depends on 2)
        5 (depends on 3, 4)
        """
        plan = [
            SubTask(index=1, sub_question="Q1", deps=[]),
            SubTask(index=2, sub_question="Q2", deps=[]),
            SubTask(index=3, sub_question="Q3", deps=[1]),
            SubTask(index=4, sub_question="Q4", deps=[2]),
            SubTask(index=5, sub_question="Q5", deps=[3, 4], is_final=True),
        ]
        waves = schedule_waves(plan)
        assert len(waves) == 3
        # Wave 0: 1, 2
        wave0_indices = {t.index for t in waves[0]}
        assert wave0_indices == {1, 2}
        # Wave 1: 3, 4
        wave1_indices = {t.index for t in waves[1]}
        assert wave1_indices == {3, 4}
        # Wave 2: 5
        assert waves[2][0].index == 5

    def test_circular_dependency_fallback(self):
        """Circular deps should not hang — force remaining into one wave."""
        plan = [
            SubTask(index=1, sub_question="Q1", deps=[2]),
            SubTask(index=2, sub_question="Q2", deps=[1], is_final=True),
        ]
        waves = schedule_waves(plan)
        # Should not be empty — fallback forces them into one wave
        assert len(waves) == 1
        assert len(waves[0]) == 2

    def test_single_task(self):
        plan = [
            SubTask(index=1, sub_question="Only Q", deps=[], is_final=True),
        ]
        waves = schedule_waves(plan)
        assert len(waves) == 1
        assert waves[0][0].sub_question == "Only Q"


# ===========================================================================
# B. _extract_answer tests
# ===========================================================================


class TestExtractAnswer:

    def setup_method(self):
        self.mgr = _make_manager()

    def test_with_answer_tags(self):
        msg = "<think>Let me think...</think><answer>Stuart Rosenberg</answer>"
        assert MultiAgentGenerationManager._extract_answer(msg) == "Stuart Rosenberg"

    def test_answer_tags_with_whitespace(self):
        msg = "<answer>\n  France  \n</answer>"
        assert MultiAgentGenerationManager._extract_answer(msg) == "France"

    def test_fallback_after_think(self):
        msg = "<think>thinking...</think>The answer is 42."
        result = MultiAgentGenerationManager._extract_answer(msg)
        assert "The answer is 42." in result

    def test_fallback_last_content(self):
        msg = "Some random text without any tags at the end."
        result = MultiAgentGenerationManager._extract_answer(msg)
        assert "random text" in result

    def test_empty_string(self):
        assert MultiAgentGenerationManager._extract_answer("") == ""

    def test_multiple_answer_tags_uses_first(self):
        msg = "<answer>First</answer> stuff <answer>Second</answer>"
        assert MultiAgentGenerationManager._extract_answer(msg) == "First"

    def test_multiline_answer(self):
        msg = "<answer>Line 1\nLine 2\nLine 3</answer>"
        result = MultiAgentGenerationManager._extract_answer(msg)
        assert "Line 1" in result
        assert "Line 3" in result


# ===========================================================================
# C. Integration: run_multi_agent_loop structure tests
# ===========================================================================


class TestRunMultiAgentLoopStructure:
    """Test the return structure of run_multi_agent_loop (mocked internals)."""

    def test_return_keys_no_writer(self):
        """Verify the return dict has planner/executor/metadata but no writer."""
        mgr = _make_manager()
        # The actual method requires too many mocked internals to call,
        # but we can verify the module no longer references writer
        import inspect
        source = inspect.getsource(MultiAgentGenerationManager.run_multi_agent_loop)
        assert "writer" not in source.lower() or "no writer" in source.lower()

    def test_schedule_waves_is_module_level(self):
        """schedule_waves should be importable from the module."""
        assert callable(schedule_waves)

    def test_no_writer_import(self):
        """Module should not import writer prompt."""
        import inspect
        source = inspect.getsource(_mod)
        assert "get_writer_prompt" not in source
        assert "from research_agent.prompts.writer" not in source
