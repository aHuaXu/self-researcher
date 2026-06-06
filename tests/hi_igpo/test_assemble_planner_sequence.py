"""Phase 2b: pure token-level assembly of the Planner training sequence.

`assemble_planner_sequence` interleaves the Planner's per-turn generations with the
(frozen) Executor's findings observations into ONE response frame, and returns:
  - response_ids:        concat([subtask_0][findings_0][subtask_1]...[answer])
  - loss_mask:           1 on Planner tokens (trained), 0 on findings observations
  - turn_end_positions:  last-token index (in the response frame) of each Planner turn,
                         ordered [subtask_0_end, ..., subtask_{T-1}_end, answer_end]
                         => length == len(planner_turn_ids) == #IG + 1, matching
                         scatter_planner_token_rewards' contract.

This is the model-independent part of `_assemble_planner_tensors` (belief/tokenization
are runtime/GPU and iterated on the server).
"""
import pytest

from scrl.llm_agent.interleaved_generation import assemble_planner_sequence, _parse_planner_turn


def test_parse_planner_turn_subtask_vs_answer():
    # <answer> -> finalize
    is_ans, payload = _parse_planner_turn("reasoning...\n<answer>Erling Haaland</answer>")
    assert is_ans is True and payload == "Erling Haaland"
    # <subtask> -> delegate one sub-question (content extracted, reasoning dropped)
    is_ans, payload = _parse_planner_turn("Let me check.\n<subtask>Who scored most in 2021-22?</subtask>")
    assert is_ans is False and payload == "Who scored most in 2021-22?"
    # malformed (no tag) -> treat whole text as subtask, keep loop alive
    is_ans, payload = _parse_planner_turn("just some text")
    assert is_ans is False and payload == "just some text"


def test_interleaves_planner_turns_with_findings_and_masks_observations():
    # 2 subtask turns + 1 answer turn; one findings per subtask.
    planner_turn_ids = [[1, 2], [3], [4, 5, 6]]   # subtask0, subtask1, answer
    findings_ids = [[7, 7], [8]]                   # findings0, findings1

    response_ids, loss_mask, turn_end_positions = assemble_planner_sequence(
        planner_turn_ids, findings_ids
    )

    # layout: [1,2][7,7][3][8][4,5,6]
    assert response_ids == [1, 2, 7, 7, 3, 8, 4, 5, 6]
    # planner tokens trained (1), findings observations masked out (0)
    assert loss_mask == [1, 1, 0, 0, 1, 0, 1, 1, 1]
    # subtask0 ends @1, subtask1 ends @4, answer ends @8
    assert turn_end_positions == [1, 4, 8]
    # contract with scatter_planner_token_rewards: len == #IG + 1 (== #subtasks + 1)
    assert len(turn_end_positions) == len(planner_turn_ids)


def test_answer_immediately_no_subtasks():
    planner_turn_ids = [[4, 5]]   # answer only, planner never delegated
    findings_ids = []

    response_ids, loss_mask, turn_end_positions = assemble_planner_sequence(
        planner_turn_ids, findings_ids
    )

    assert response_ids == [4, 5]
    assert loss_mask == [1, 1]
    assert turn_end_positions == [1]   # only the F1 (answer) position


def test_rejects_mismatched_findings_count():
    # findings must be exactly one per subtask turn (len(planner) - 1)
    with pytest.raises(ValueError):
        assemble_planner_sequence([[1], [2], [3]], [[9]])   # need 2 findings, got 1
