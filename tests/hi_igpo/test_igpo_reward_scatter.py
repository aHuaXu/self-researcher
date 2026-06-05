import pytest
import torch

from verl.trainer.ppo.igpo_utils import (
    extract_ground_truths_for_igpo,
    is_critic_free_adv_estimator,
    scatter_info_gain_rewards,
)


def test_scatter_info_gain_rewards_places_ig_and_f1_at_turn_boundaries():
    info_gain_rewards = [[0.3, 0.05, 0.45], [-0.1, 0.2]]
    f1_scores = torch.tensor([1.0, 0.0])
    turn_end_positions = [[1, 3, 5, 7], [1, 3, 5]]

    rewards, boundary = scatter_info_gain_rewards(
        info_gain_rewards=info_gain_rewards,
        f1_scores=f1_scores,
        turn_end_positions=turn_end_positions,
        response_length=8,
    )

    expected = torch.tensor([
        [0.0, 0.3, 0.0, 0.05, 0.0, 0.45, 0.0, 1.0],
        [0.0, -0.1, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0],
    ])
    expected_boundary = torch.tensor([
        [False, True, False, True, False, True, False, True],
        [False, True, False, True, False, True, False, False],
    ])

    assert torch.allclose(rewards, expected, atol=1e-6)
    assert torch.equal(boundary, expected_boundary)


def test_scatter_info_gain_rewards_rejects_misaligned_turn_positions():
    with pytest.raises(ValueError, match="len\\(turn_end_positions\\) must equal len\\(info_gain_rewards\\) \\+ 1"):
        scatter_info_gain_rewards(
            info_gain_rewards=[[0.1, 0.2]],
            f1_scores=torch.tensor([1.0]),
            turn_end_positions=[[1, 3]],
            response_length=6,
        )


def test_igpo_adv_estimator_is_critic_free():
    assert is_critic_free_adv_estimator("igpo")


def test_scatter_planner_token_rewards_from_beliefs():
    from verl.trainer.ppo.igpo_utils import scatter_planner_token_rewards
    # beliefs Bel_0..Bel_3 -> IG=[0.3, 0.05, 0.45]; F1=1.0 at answer end.
    beliefs = [0.1, 0.4, 0.45, 0.9]
    rewards, boundary = scatter_planner_token_rewards(
        beliefs=beliefs, f1=1.0, turn_end_positions=[1, 3, 5, 7], response_length=8)
    expect = torch.zeros(8); expect[1] = 0.3; expect[3] = 0.05; expect[5] = 0.45; expect[7] = 1.0
    assert torch.allclose(rewards, expect, atol=1e-6)
    assert boundary.tolist() == [False, True, False, True, False, True, False, True]


def test_scatter_planner_rejects_misaligned_positions():
    from verl.trainer.ppo.igpo_utils import scatter_planner_token_rewards
    with pytest.raises(ValueError):
        scatter_planner_token_rewards(beliefs=[0.1, 0.4, 0.9], f1=1.0,
                                      turn_end_positions=[1, 3], response_length=6)  # need 2 IG + 1 = 3


def test_extract_ground_truths_for_igpo_repeats_reward_model_entries():
    reward_models = [
        {"ground_truth": "alpha"},
        {"ground_truth": "beta"},
    ]

    ground_truths = extract_ground_truths_for_igpo(reward_models, repeat_times=2)

    assert ground_truths == [
        {"ground_truth": "alpha"},
        {"ground_truth": "alpha"},
        {"ground_truth": "beta"},
        {"ground_truth": "beta"},
    ]
