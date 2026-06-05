from typing import Sequence

import torch


def extract_ground_truths_for_igpo(reward_models: Sequence[dict], repeat_times: int = 1) -> list[dict]:
    """Extract `ground_truth` entries in the shape expected by igpo_generation."""
    if repeat_times < 1:
        raise ValueError("repeat_times must be >= 1")

    ground_truths: list[dict] = []
    for reward_model in reward_models:
        if "ground_truth" not in reward_model:
            raise KeyError("reward_model entry must contain 'ground_truth'")
        for _ in range(repeat_times):
            ground_truths.append({"ground_truth": reward_model["ground_truth"]})
    return ground_truths


def is_critic_free_adv_estimator(adv_estimator) -> bool:
    """Return whether an advantage estimator should skip critic construction."""
    return str(adv_estimator) in {
        "grpo",
        "reinforce_plus_plus",
        "remax",
        "rloo",
        "igpo",
    }


def scatter_info_gain_rewards(
    info_gain_rewards: Sequence[Sequence[float]],
    f1_scores: torch.Tensor,
    turn_end_positions: Sequence[Sequence[int]],
    response_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter per-turn IG and final F1 rewards to IGPO-native token tensors.

    IGPO expects IG rewards at intermediate turn-end tokens and the outcome
    reward at the final valid answer token. `turn_end_positions[i]` therefore
    contains all intermediate IG turn ends followed by the final F1 position.
    """
    batch_size = len(info_gain_rewards)
    if len(turn_end_positions) != batch_size:
        raise ValueError("turn_end_positions must have one entry per sample")
    if f1_scores.numel() != batch_size:
        raise ValueError("f1_scores must have one value per sample")

    device = f1_scores.device
    rewards = torch.zeros(batch_size, response_length, dtype=torch.float32, device=device)
    boundary = torch.zeros(batch_size, response_length, dtype=torch.bool, device=device)

    for row_idx, (row_igs, row_positions) in enumerate(zip(info_gain_rewards, turn_end_positions)):
        if len(row_positions) != len(row_igs) + 1:
            raise ValueError("len(turn_end_positions) must equal len(info_gain_rewards) + 1")
        for pos in row_positions:
            if pos < 0 or pos >= response_length:
                raise ValueError(f"turn end position {pos} out of response length {response_length}")

        for ig, pos in zip(row_igs, row_positions[:-1]):
            rewards[row_idx, pos] = float(ig)
            boundary[row_idx, pos] = True

        f1_pos = row_positions[-1]
        rewards[row_idx, f1_pos] = f1_scores[row_idx].to(dtype=torch.float32)
        boundary[row_idx, f1_pos] = True

    return rewards, boundary
