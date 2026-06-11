from typing import Sequence

import torch


def compute_igpo_token_level_scores(data, tokenizer, info_gain_rewards, val_type="f1", format_penalty=1.0):
    """Build IGPO token-level reward tensor (bs, response_len) from per-sample IG.

    Mirrors NaiveRewardManager's decode→score loop, but uses info_gain.compute_score
    so each sample gets per-turn IG at turn-end tokens + F1 at the final turn. Must be
    called while `data` rows still align with `info_gain_rewards` (i.e. after
    repeat+union but BEFORE _balance_batch reorders the batch); the resulting tensor
    then rides along through balancing as a normal batch field.
    """
    from verl.utils.reward_score import info_gain

    reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
    for i in range(len(data)):
        item = data[i]
        prompt_ids = item.batch['prompts']
        prompt_length = prompt_ids.shape[-1]
        response_ids = item.batch['responses']
        valid_response_length = int(item.batch['attention_mask'][prompt_length:].sum())
        if valid_response_length == 0:
            continue
        valid_response_ids = response_ids[:valid_response_length]
        response_str = tokenizer.decode(valid_response_ids)

        ground_truth = item.non_tensor_batch['reward_model']['ground_truth']
        data_source = item.non_tensor_batch['data_source']
        ig = info_gain_rewards[i] if (info_gain_rewards is not None and i < len(info_gain_rewards)) else []

        scores = info_gain.compute_score(
            solution_str=response_str,
            ground_truth=ground_truth,
            data_source=data_source,
            val_type=val_type,
            info_gain_reward=list(ig),
            tokenizer=tokenizer,
            format_penalty=format_penalty,
        )
        n = min(len(scores), valid_response_length)
        for j in range(n):
            reward_tensor[i, j] = float(scores[j])
    return reward_tensor


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


def scatter_planner_token_rewards(beliefs, f1, turn_end_positions, response_length):
    """Phase 2b: scatter one Planner trajectory's per-turn IG + final F1 to token rewards.

    beliefs = [Bel_0, ..., Bel_T] (initial belief + one per executed subtask). The per-turn
    information gain is IG_t = beliefs[t+1] - beliefs[t] (T values), written at the Planner
    turn-end tokens; F1 goes at the answer token. `turn_end_positions` therefore has T+1
    entries: the T IG turn-ends followed by the answer (F1) position.

    Returns (token_level_rewards: (L,), turn_boundary_mask: (L,)) — same token-level form as
    the single-agent path, fed (after stacking to (bs, L)) to compute_igpo_turn_advantage.
    """
    beliefs = torch.as_tensor(list(beliefs), dtype=torch.float32)
    igs = beliefs[1:] - beliefs[:-1]                       # (T,)
    if len(turn_end_positions) != igs.numel() + 1:
        raise ValueError(
            f"turn_end_positions ({len(turn_end_positions)}) must equal #IG ({igs.numel()}) + 1 (answer)"
        )
    rewards = torch.zeros(response_length, dtype=torch.float32)
    boundary = torch.zeros(response_length, dtype=torch.bool)
    for pos in turn_end_positions:
        if pos < 0 or pos >= response_length:
            raise ValueError(f"turn end position {pos} out of response length {response_length}")
    for t, pos in enumerate(turn_end_positions[:-1]):
        rewards[pos] = igs[t]
        boundary[pos] = True
    f1_pos = turn_end_positions[-1]
    rewards[f1_pos] = float(f1)
    boundary[f1_pos] = True
    return rewards, boundary


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
