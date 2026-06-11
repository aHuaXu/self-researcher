"""CPU tests for the ported info_gain.compute_score token-level scatter.

Uses a char-level fake tokenizer (each char = 1 token, offset (i, i+1)) so token
index == char position, making expected turn-end placements exact and offset_mapping
behaviour deterministic — no real tokenizer / GPU needed.
"""
from verl.utils.reward_score import info_gain


class CharTokenizer:
    """Each character is one token; offset_mapping is (i, i+1)."""

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False):
        ids = list(range(len(text)))
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out


SEP = "\n<|im_start|>assistant\n"
TOK = CharTokenizer()


def test_two_turns_ig_then_f1_at_turn_ends():
    # turn0 must be format-valid (has <tool_call>) so the turn-level format gate keeps its IG.
    turn0 = 'first turn <tool_call>{"name":"web_search"}</tool_call>'
    turn1 = "second <answer>Paris</answer>"
    solution = turn0 + SEP + turn1

    scores = info_gain.compute_score(
        solution_str=solution, ground_truth="Paris", data_source="x",
        val_type="f1", info_gain_reward=[0.5], tokenizer=TOK,
    )

    assert len(scores) == len(solution)
    # IG=0.5 at last char of turn0
    assert scores[len(turn0) - 1] == 0.5
    # F1=1.0 (answer "Paris" == gt) at last char of the whole solution
    assert scores[len(solution) - 1] == 1.0
    # everything else zero
    nonzero = {i for i, s in enumerate(scores) if s != 0.0}
    assert nonzero == {len(turn0) - 1, len(solution) - 1}


def test_zero_ig_becomes_tiny_nonzero():
    turn0 = 'abc <tool_call>{"name":"web_search"}</tool_call>'  # format-valid so IG slot is kept
    turn1 = "d <answer>Paris</answer>"
    solution = turn0 + SEP + turn1
    scores = info_gain.compute_score(
        solution_str=solution, ground_truth="Paris", data_source="x",
        val_type="f1", info_gain_reward=[0.0], tokenizer=TOK,
    )
    # zero IG must be stored as 1e-10 (so reward!=0 turn detection doesn't skip it)
    assert scores[len(turn0) - 1] == 1e-10


def test_turn_count_mismatch_falls_back_to_f1_only():
    turn0 = "abc"
    turn1 = "d <answer>Paris</answer>"
    solution = turn0 + SEP + turn1  # 2 turns -> expects 1 IG
    scores = info_gain.compute_score(
        solution_str=solution, ground_truth="Paris", data_source="x",
        val_type="f1", info_gain_reward=[0.1, 0.2, 0.3], tokenizer=TOK,  # wrong length
    )
    nonzero = {i for i, s in enumerate(scores) if s != 0.0}
    assert nonzero == {len(solution) - 1}          # only F1 at last token
    assert scores[len(solution) - 1] == 1.0


def test_single_turn_f1_on_last_token():
    solution = "no separator here <answer>Paris</answer>"
    scores = info_gain.compute_score(
        solution_str=solution, ground_truth="Paris", data_source="x",
        val_type="f1", info_gain_reward=[], tokenizer=TOK,
    )
    nonzero = {i for i, s in enumerate(scores) if s != 0.0}
    assert nonzero == {len(solution) - 1}
    assert scores[len(solution) - 1] == 1.0
