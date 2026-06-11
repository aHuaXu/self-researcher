"""Turn-level format penalty (DR-Venus eq.4) for info_gain.compute_score.

Loads info_gain.py directly (it only needs re/string/json) to avoid importing the
heavy `verl` package. Uses a char-level fake tokenizer so token positions are
deterministic and no model/transformers is required (runs on .venv-test, CPU).
"""
import importlib.util
import os

_INFO_GAIN_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "verl", "utils", "reward_score", "info_gain.py"
)
_spec = importlib.util.spec_from_file_location("info_gain_standalone", _INFO_GAIN_PATH)
ig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ig)


class CharTokenizer:
    """Each character is one token; offset_mapping is the identity char span."""

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False):
        out = {"input_ids": list(range(len(text)))}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out


TOK = CharTokenizer()
SEP = "\n<|im_start|>assistant\n"
IMEND = "<|im_end|>"


def _two_turn(turn0_asst, turn1_asst, toolresp="snippet <b> junk </b>"):
    """Render a 2-turn response string: turn0 (assistant + tool response) then turn1."""
    return (
        f"{turn0_asst}{IMEND}\n<|im_start|>tool\n{toolresp}{IMEND}"
        f"{SEP}{turn1_asst}{IMEND}"
    )


# ---- helper-level checks -------------------------------------------------

def test_turn_format_ok_intermediate_and_final():
    assert ig._turn_format_ok('reason <tool_call>{"name":"web_search"}</tool_call>', is_final=False)
    assert not ig._turn_format_ok("reason <tool_call>{oops", is_final=False)  # unbalanced
    assert not ig._turn_format_ok("reason only text", is_final=False)         # no tool_call
    assert ig._turn_format_ok("reason <answer>paris</answer>", is_final=True)
    assert not ig._turn_format_ok("reason paris", is_final=True)              # no answer


def test_answer_f1_pure_no_minus2():
    # correct answer -> 1.0 ; wrong -> 0.0 (NOT -2, format gate decoupled)
    assert ig._answer_f1_from_text("x <answer>paris</answer>", "paris", "deepresearch") == 1.0
    assert ig._answer_f1_from_text("x <answer>london</answer>", "paris", "deepresearch") == 0.0


def test_assistant_text_excludes_tool_response():
    s = _two_turn('a <tool_call>{"name":"web_search"}</tool_call>', "b <answer>paris</answer>")
    # turn0 assistant text must stop before tool response junk (the <b></b> in toolresp)
    asst0 = ig._assistant_text(s, 0, s.find(SEP))
    assert "<tool_call>" in asst0 and "snippet" not in asst0


# ---- compute_score scatter behaviour ------------------------------------

def test_both_turns_wellformed_keep_ig_and_f1():
    s = _two_turn('a <tool_call>{"name":"web_search"}</tool_call>', "b <answer>paris</answer>")
    scores = ig.compute_score(s, "paris", "deepresearch", info_gain_reward=[0.5], tokenizer=TOK)
    assert -1.0 not in scores                 # no format penalty
    assert any(abs(v - 0.5) < 1e-9 for v in scores)   # IG kept on turn0
    assert any(abs(v - 1.0) < 1e-9 for v in scores)   # F1=1.0 on final


def test_malformed_intermediate_only_that_turn_penalized():
    s = _two_turn("a <tool_call>{oops", "b <answer>paris</answer>")  # turn0 unbalanced
    scores = ig.compute_score(s, "paris", "deepresearch", info_gain_reward=[0.5], tokenizer=TOK)
    assert scores.count(-1.0) == 1                      # exactly one penalized turn
    assert not any(abs(v - 0.5) < 1e-9 for v in scores)  # IG dropped for bad turn
    assert any(abs(v - 1.0) < 1e-9 for v in scores)     # final F1 still kept


def test_malformed_final_turn_penalized_but_ig_kept():
    s = _two_turn('a <tool_call>{"name":"web_search"}</tool_call>', "b paris")  # final no <answer>
    scores = ig.compute_score(s, "paris", "deepresearch", info_gain_reward=[0.5], tokenizer=TOK)
    assert scores.count(-1.0) == 1
    assert any(abs(v - 0.5) < 1e-9 for v in scores)     # IG kept on good turn0
    assert not any(abs(v - 1.0) < 1e-9 for v in scores)  # no F1 (final penalized)


def test_single_turn_clean_wrong_is_zero_not_minus2():
    s = "reason <answer>london</answer>"   # 1 turn, wrong answer, well-formed
    scores = ig.compute_score(s, "paris", "deepresearch", info_gain_reward=[], tokenizer=TOK)
    assert -2.0 not in scores and -1.0 not in scores
    assert max(scores) == 0.0 and min(scores) == 0.0   # F1=0, no penalty


def test_single_turn_malformed_gets_penalty():
    s = "reason no answer here"  # 1 turn, no <answer>
    scores = ig.compute_score(s, "paris", "deepresearch", info_gain_reward=[], tokenizer=TOK)
    assert scores.count(-1.0) == 1


def test_format_penalty_value_configurable():
    s = "reason no answer here"
    scores = ig.compute_score(s, "paris", "deepresearch", info_gain_reward=[], tokenizer=TOK, format_penalty=2.5)
    assert scores.count(-2.5) == 1
