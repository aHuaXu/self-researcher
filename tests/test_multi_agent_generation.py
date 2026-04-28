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
#    but research_agent/__init__.py imports graph.py which pulls heavy deps.
#    Pre-populate research_agent as a namespace package so the sub-imports work.
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


# ---------------------------------------------------------------------------
# Helper: create a manager instance without calling __init__
# ---------------------------------------------------------------------------

def _make_manager(tokenizer=None):
    mgr = object.__new__(MultiAgentGenerationManager)
    mgr.tokenizer = tokenizer or MagicMock(pad_token_id=0, pad_token="<pad>")
    mgr.tensor_fn = MagicMock()
    mgr.config = MagicMock()
    mgr.lora_save_dir = './tmp_lora_adapters'
    return mgr


# ===========================================================================
# A. Pure logic tests (no mock needed)
# ===========================================================================


class TestParseTodos:

    def setup_method(self):
        self.mgr = _make_manager()

    def test_basic_format(self):
        text = "1. [HIGH] Climate change impacts on agriculture"
        result = self.mgr._parse_todos(text)
        assert len(result) == 1
        assert result[0]["index"] == 1
        assert result[0]["priority"] == "high"
        assert result[0]["sub_topic"] == "Climate change impacts on agriculture"
        assert "search_query" not in result[0]

    def test_multiple_todos(self):
        text = (
            "1. [HIGH] Topic A\n"
            "2. [MEDIUM] Topic B\n"
            "3. [LOW] Topic C"
        )
        result = self.mgr._parse_todos(text)
        assert len(result) == 3
        assert [r["priority"] for r in result] == ["high", "medium", "low"]
        assert [r["sub_topic"] for r in result] == ["Topic A", "Topic B", "Topic C"]

    def test_sub_topic_prefix_stripped(self):
        text = (
            "1. [HIGH] Sub-topic: Some topic about AI\n"
            "2. [LOW] Another topic about ML"
        )
        result = self.mgr._parse_todos(text)
        assert len(result) == 2
        assert result[0]["sub_topic"] == "Some topic about AI"
        assert result[1]["sub_topic"] == "Another topic about ML"

    def test_chinese_prefix_stripped(self):
        text = "1. [HIGH] 子主题：人工智能发展"
        result = self.mgr._parse_todos(text)
        assert len(result) == 1
        assert result[0]["sub_topic"] == "人工智能发展"

    def test_todos_tag_cleanup(self):
        text = "1. [HIGH] Topic X</todos>"
        result = self.mgr._parse_todos(text)
        assert len(result) == 1
        assert "</todos>" not in result[0]["sub_topic"]

    def test_fallback_plain_text(self):
        text = "Just a plain research question about quantum computing"
        result = self.mgr._parse_todos(text)
        assert len(result) == 1
        assert result[0]["priority"] == "high"
        assert result[0]["sub_topic"] == text

    def test_empty_string(self):
        result = self.mgr._parse_todos("")
        assert result == []

    def test_mixed_priority_case_insensitive(self):
        text = (
            "1. [high] Topic A\n"
            "2. [Medium] Topic B\n"
            "3. [LOW] Topic C"
        )
        result = self.mgr._parse_todos(text)
        assert len(result) == 3
        assert all(r["priority"] in ("high", "medium", "low") for r in result)


class TestGroupFindings:

    def setup_method(self):
        self.mgr = _make_manager()

    def test_correct_grouping(self):
        exec_msgs = ["finding A1", "finding A2", "finding B1", "finding B2"]
        mapping = [0, 0, 1, 1]
        result = self.mgr._group_findings(exec_msgs, mapping, 2)
        assert len(result) == 2
        assert "finding A1" in result[0] or "Finding 1" in result[0]
        assert "finding B1" in result[1] or "Finding 3" in result[1]

    def test_answer_tag_extraction(self):
        exec_msgs = ["<think>reasoning</think><answer>The answer is 42</answer>"]
        mapping = [0]
        result = self.mgr._group_findings(exec_msgs, mapping, 1)
        assert "The answer is 42" in result[0]

    def test_no_answer_tag_fallback(self):
        exec_msgs = ["<think>reasoning</think>some content after think"]
        mapping = [0]
        result = self.mgr._group_findings(exec_msgs, mapping, 1)
        assert "some content after think" in result[0]

    def test_empty_findings_placeholder(self):
        exec_msgs = ["finding for Q0"]
        mapping = [0]
        result = self.mgr._group_findings(exec_msgs, mapping, 2)
        assert "[No findings available]" in result[1]

    def test_exec_shorter_than_mapping(self):
        exec_msgs = ["only one"]
        mapping = [0, 0, 1]
        result = self.mgr._group_findings(exec_msgs, mapping, 2)
        assert len(result) == 2
        assert "only one" in result[0]


class TestBuildExecTrajectories:

    def setup_method(self):
        self.mgr = _make_manager()

    def test_normal_grouping(self):
        msgs = [
            '<tool_call>{"name":"web_search","arguments":{}}</tool_call>\n<observation>res0</observation>',
            '<tool_call>{"name":"browse_webpage","arguments":{}}</tool_call>\n<observation>res1</observation>',
            '<tool_call>{"name":"web_search","arguments":{}}</tool_call>\n<observation>res2</observation>',
        ]
        mapping = [0, 0, 1]
        result = self.mgr._build_exec_trajectories(msgs, mapping, 2)
        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[1]) == 1
        assert result[0][0]["tool"] == "web_search"
        assert result[0][0]["result"] == "res0"
        assert result[1][0]["tool"] == "web_search"

    def test_empty_mapping(self):
        result = self.mgr._build_exec_trajectories([], [], 3)
        assert len(result) == 3
        assert all(len(t) == 0 for t in result)

    def test_exec_shorter_than_mapping(self):
        msgs = ['<tool_call>{"name":"web_search","arguments":{}}</tool_call>\n<observation>only one</observation>']
        mapping = [0, 1]
        result = self.mgr._build_exec_trajectories(msgs, mapping, 2)
        assert len(result[0]) == 1
        assert result[0][0]["result"] == "only one"
        # index 1 maps to exec_idx=1 which is out of range → empty string → no tool steps parsed
        assert len(result[1]) == 0


class TestExtractLastResponse:

    def setup_method(self):
        self.mgr = _make_manager()

    def test_with_think_block(self):
        msg = "<think>some reasoning</think>actual content here"
        result = self.mgr._extract_last_response(msg)
        assert "actual content here" in result

    def test_multiple_think_blocks(self):
        msg = (
            "<think>first</think>response1"
            "<think>second</think>final response"
        )
        result = self.mgr._extract_last_response(msg)
        assert "final response" in result

    def test_no_think_tag_fallback(self):
        msg = "x" * 1000
        result = self.mgr._extract_last_response(msg)
        assert len(result) == 500

    def test_empty_string(self):
        result = self.mgr._extract_last_response("")
        assert result == ""


# ===========================================================================
# B. Tests requiring mock tokenizer
# ===========================================================================


def _make_tokenizer():
    """Create a mock tokenizer with the methods used by the manager."""
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.pad_token = "<pad>"
    return tok


class TestBuildExecutorBatch:

    def _setup_with_tokenize_mock(self):
        mgr = _make_manager()
        mgr._tokenize_messages_to_batch = MagicMock(
            return_value=_FakeDataProto.from_dict({
                "input_ids": torch.zeros((4, 10), dtype=torch.long),
                "attention_mask": torch.ones((4, 10), dtype=torch.long),
                "position_ids": torch.arange(10).unsqueeze(0).expand(4, -1),
            })
        )
        return mgr

    def test_flatten_todos(self):
        mgr = self._setup_with_tokenize_mock()
        questions = ["Q1", "Q2"]
        todos = [
            [{"sub_topic": "T1a"}, {"sub_topic": "T1b"}],
            [{"sub_topic": "T2a"}, {"sub_topic": "T2b"}],
        ]
        ref = _FakeDataProto()
        batch, mapping = mgr._build_executor_batch(questions, todos, ref)
        assert mapping == [0, 0, 1, 1]
        mgr._tokenize_messages_to_batch.assert_called_once()
        call_args = mgr._tokenize_messages_to_batch.call_args
        assert len(call_args[0][0]) == 4

    def test_empty_todos(self):
        mgr = self._setup_with_tokenize_mock()
        batch, mapping = mgr._build_executor_batch(
            ["Q1", "Q2"], [[], []], _FakeDataProto()
        )
        assert mapping == []
        assert len(batch) == 0

    def test_partial_todos(self):
        mgr = self._setup_with_tokenize_mock()
        mgr._tokenize_messages_to_batch.return_value = _FakeDataProto.from_dict({
            "input_ids": torch.zeros((1, 10), dtype=torch.long),
            "attention_mask": torch.ones((1, 10), dtype=torch.long),
            "position_ids": torch.arange(10).unsqueeze(0),
        })
        todos = [[], [{"sub_topic": "T"}]]
        batch, mapping = mgr._build_executor_batch(["Q1", "Q2"], todos, _FakeDataProto())
        assert mapping == [1]

    def test_prompt_content(self):
        mgr = self._setup_with_tokenize_mock()
        todos = [[{"sub_topic": "AI Safety"}]]
        mgr._build_executor_batch(["Q1"], todos, _FakeDataProto())
        messages_list = mgr._tokenize_messages_to_batch.call_args[0][0]
        user_msg = messages_list[0][1]["content"]
        assert "AI Safety" in user_msg


class TestBuildPlannerBatch:

    def test_calls_tokenize(self):
        mgr = _make_manager()
        fake_batch = _FakeDataProto.from_dict({
            "input_ids": torch.zeros((2, 10), dtype=torch.long),
        })
        mgr._tokenize_messages_to_batch = MagicMock(return_value=fake_batch)
        result = mgr._build_planner_batch(["Q1", "Q2"], _FakeDataProto())
        mgr._tokenize_messages_to_batch.assert_called_once()
        messages_list = mgr._tokenize_messages_to_batch.call_args[0][0]
        assert len(messages_list) == 2
        assert messages_list[0][0]["role"] == "system"
        assert "Q1" in messages_list[0][1]["content"]

    def test_batch_size_matches(self):
        mgr = _make_manager()
        fake_batch = _FakeDataProto.from_dict({
            "input_ids": torch.zeros((3, 10), dtype=torch.long),
        })
        mgr._tokenize_messages_to_batch = MagicMock(return_value=fake_batch)
        result = mgr._build_planner_batch(["Q1", "Q2", "Q3"], _FakeDataProto())
        assert len(result) == 3


class TestBuildWriterBatch:

    def test_plan_and_findings_format(self):
        mgr = _make_manager()
        mgr._tokenize_messages_to_batch = MagicMock(
            return_value=_FakeDataProto.from_dict({
                "input_ids": torch.zeros((1, 10), dtype=torch.long),
            })
        )
        mgr._build_writer_batch(
            ["Q1"], ["plan text here"], ["findings here"], _FakeDataProto()
        )
        messages_list = mgr._tokenize_messages_to_batch.call_args[0][0]
        user_content = messages_list[0][1]["content"]
        assert "=== Research Plan ===" in user_content
        assert "plan text here" in user_content
        assert "=== Research Findings ===" in user_content
        assert "findings here" in user_content

    def test_multiple_questions_independent(self):
        mgr = _make_manager()
        mgr._tokenize_messages_to_batch = MagicMock(
            return_value=_FakeDataProto.from_dict({
                "input_ids": torch.zeros((2, 10), dtype=torch.long),
            })
        )
        mgr._build_writer_batch(
            ["Q1", "Q2"], ["plan1", "plan2"], ["find1", "find2"], _FakeDataProto()
        )
        messages_list = mgr._tokenize_messages_to_batch.call_args[0][0]
        assert len(messages_list) == 2
        assert "plan1" in messages_list[0][1]["content"]
        assert "plan2" in messages_list[1][1]["content"]


class TestTokenizeMessagesToBatch:

    def test_left_padding(self):
        tok = _make_tokenizer()
        tok.apply_chat_template.return_value = ["hello world", "hi"]
        tok.return_value = {
            "input_ids": torch.tensor([[1, 2, 3], [0, 0, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1], [0, 0, 1]]),
        }

        mgr = _make_manager(tokenizer=tok)
        mgr.tensor_fn.create_position_ids.return_value = torch.tensor(
            [[0, 1, 2], [0, 0, 0]]
        )

        result = mgr._tokenize_messages_to_batch(
            [[{"role": "user", "content": "hello"}]], _FakeDataProto()
        )
        input_ids = result.batch["input_ids"]
        assert input_ids[1, 0].item() == 0
        assert input_ids[1, 2].item() == 4

    def test_tools_passed_to_template(self):
        tok = _make_tokenizer()
        tok.apply_chat_template.return_value = ["formatted"]
        tok.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        mgr = _make_manager(tokenizer=tok)
        mgr.tensor_fn.create_position_ids.return_value = torch.tensor([[0, 1]])

        fake_tools = [{"type": "function", "function": {"name": "test"}}]
        mgr._tokenize_messages_to_batch(
            [[{"role": "user", "content": "hi"}]],
            _FakeDataProto(),
            tools=fake_tools,
        )
        call_kwargs = tok.apply_chat_template.call_args
        assert call_kwargs[1]["tools"] == fake_tools

    def test_position_ids_created(self):
        tok = _make_tokenizer()
        tok.apply_chat_template.return_value = ["text"]
        tok.return_value = {
            "input_ids": torch.tensor([[5, 6]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        mgr = _make_manager(tokenizer=tok)
        expected_pos = torch.tensor([[0, 1]])
        mgr.tensor_fn.create_position_ids.return_value = expected_pos

        result = mgr._tokenize_messages_to_batch(
            [[{"role": "user", "content": "x"}]], _FakeDataProto()
        )
        mgr.tensor_fn.create_position_ids.assert_called_once()
        assert torch.equal(result.batch["position_ids"], expected_pos)


class TestDecodeOutputs:

    def test_normal_decode(self):
        tok = _make_tokenizer()
        tok.batch_decode.return_value = ["hello <pad> world <pad>"]

        mgr = _make_manager(tokenizer=tok)
        outputs = _FakeDataProto.from_dict({
            "responses": torch.tensor([[1, 2, 3]]),
        })
        result = mgr._decode_outputs(outputs)
        assert len(result) == 1
        assert "<pad>" not in result[0]

    def test_no_responses_key(self):
        mgr = _make_manager()
        outputs = _FakeDataProto.from_dict({
            "input_ids": torch.tensor([[1, 2]]),
        })
        result = mgr._decode_outputs(outputs)
        assert result == []

    def test_pad_token_none_fallback(self):
        tok = _make_tokenizer()
        tok.pad_token = None
        tok.batch_decode.return_value = ["answer <|endoftext|> rest"]

        mgr = _make_manager(tokenizer=tok)
        outputs = _FakeDataProto.from_dict({
            "responses": torch.tensor([[1, 2]]),
        })
        result = mgr._decode_outputs(outputs)
        assert "<|endoftext|>" not in result[0]
