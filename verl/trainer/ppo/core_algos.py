# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F


def _compute_turn_level_advantage(
    normalized_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float,
    bsz: int,
    seq_len: int,
    device: torch.device,
    turn_boundary_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Turn-level discounted accumulation + broadcast to all tokens in each turn.

    Ported from IGPO `/tmp/IGPO_ref/verl/trainer/ppo/core_algos.py` (line 28).
    Used internally by compute_igpo_turn_advantage.

    Args:
        normalized_rewards: (bsz, seq_len) already-normalized token-level rewards.
            Only turn-end positions carry non-zero values.
        response_mask: (bsz, seq_len) 1 for valid response tokens, 0 for padding.
        gamma: discount factor for turn-level accumulation.
        bsz: batch size.
        seq_len: sequence length.
        device: torch device.
        turn_boundary_mask: (bsz, seq_len) optional; marks turn-end positions.
            When provided, used instead of `normalized_rewards != 0` heuristic to
            avoid missing turns whose normalized reward happens to be zero.

    Returns:
        discounted_returns: (bsz, seq_len) turn advantage broadcast to every
            token in the corresponding turn.
    """
    discounted_returns = torch.zeros(bsz, seq_len, device=device, dtype=normalized_rewards.dtype)

    for sample_idx in range(bsz):
        sample_rewards = normalized_rewards[sample_idx]   # (seq_len,)
        sample_mask    = response_mask[sample_idx]         # (seq_len,)

        # Step 1: identify turn-end positions
        if turn_boundary_mask is not None:
            reward_positions = turn_boundary_mask[sample_idx].nonzero(as_tuple=True)[0].tolist()
        else:
            reward_positions = (sample_rewards != 0).nonzero(as_tuple=True)[0].tolist()

        if len(reward_positions) == 0:
            continue

        # Step 2: backward discounted accumulation
        turn_data = []
        next_turn_adv = 0.0
        for pos in reversed(reward_positions):
            turn_reward = sample_rewards[pos].item()
            turn_adv    = turn_reward + gamma * next_turn_adv
            turn_data.append((pos, turn_adv))
            next_turn_adv = turn_adv
        turn_data.reverse()   # forward order

        # Step 3: broadcast advantage to all tokens in the turn
        # Turn i covers [prev_end, reward_pos] (inclusive on both ends)
        prev_end = 0
        for reward_pos, adv in turn_data:
            for t in range(prev_end, reward_pos + 1):
                if sample_mask[t] == 1:
                    discounted_returns[sample_idx, t] = adv
            prev_end = reward_pos + 1

    return discounted_returns


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config):
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    
    assert not token_level_rewards.isnan().any(), "token_level_rewards is nan in compute_grpo_outcome_advantage"
    
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[int(index[i])].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[int(index[i])]) / (id2std[int(index[i])] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
    
    return scores, scores


def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num -
                                                        1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def _grouped_mean_std(
    values: torch.Tensor,
    group_id: torch.Tensor,
    norm_by_std: bool,
):
    """
    Compute per-group mean and std for a flat list of values.

    Uses the same std formula as compute_group_stats (population variance + 1e-8):
        std = sqrt(mean((v - v.mean())**2) + 1e-8)
    When group size <= 1, std = 1.0 (no normalization for singletons).
    When norm_by_std=False, std is always 1.0.

    Args:
        values:    (M,) float tensor
        group_id:  (M,) long tensor — consecutive group indices
        norm_by_std: if False, std is always 1.0 (no std division)

    Returns:
        mean: (M,) float — group mean broadcast back to each element
        std:  (M,) float — group std (1.0 if norm_by_std=False or group size ≤ 1)
        size: (M,) float — group size broadcast back to each element
    """
    mean = torch.zeros_like(values)
    std  = torch.ones_like(values)
    size = torch.zeros_like(values)  # zeros: unset elements won't be mistaken as size=1
    for g in torch.unique(group_id):
        m = group_id == g
        v = values[m]
        mean[m] = v.mean()
        size[m] = float(v.numel())
        if norm_by_std and v.numel() > 1:
            # Use same formula as compute_group_stats: population variance + 1e-8
            var = ((v - v.mean()) ** 2).mean()
            std[m] = torch.sqrt(var + 1e-8)
    return mean, std, size


def compute_igpo_turn_advantage(
    turn_records: dict,
    bs: int,
    response_len: int,
    gamma: float = 1.0,
    ig_group_mode: str = "global",
    info_gain_norm_mode: str = "separate",
    norm_by_std: bool = True,
    min_group_size: int = 2,
    epsilon: float = 1e-6,
):
    """
    Compute IGPO turn-level advantage from per-turn records.

    Port of IGPO `compute_grpo_outcome_advantage` + `_compute_turn_level_advantage`.
    Translates the per-turn `turn_records` dict (this repo's unified representation)
    into token-level (bs, response_len) tensors, then applies IGPO's
    normalize -> discounted accumulate -> scatter logic.

    Args:
        turn_records: dict with torch.Tensor fields of length N (number of turns total):
            turn_reward  (float): per-turn reward value
            prompt_id    (long):  prompt group id for global normalization
            traj_id      (long):  trajectory id (maps to sample_row in batch)
            turn_pos     (long):  turn index within trajectory
            is_outcome   (bool):  True for the final F1 turn, False for IG turns
            sample_row   (long):  row index in the (bs, response_len) output tensors
            span_start   (long):  token start position of this turn (inclusive)
            span_end     (long):  token end position of this turn (exclusive)
        bs: batch size
        response_len: response sequence length
        gamma: discount factor for turn-level accumulation (default 1.0)
        ig_group_mode: "global" = normalize IG turns across all trajectories sharing a prompt_id.
            "turn_group" = normalize IG turns by (prompt_id, turn_pos) group; groups with fewer
            than min_group_size samples fall back to the prompt-level statistics.
        info_gain_norm_mode: "separate" = F1 and IG pools normalized independently;
            "joint" = F1 + IG pooled together.
        norm_by_std: whether to divide by group std (IGPO default True)
        min_group_size: for turn_group mode, groups smaller than this fall back to prompt-level stats
        epsilon: small constant for division stability

    Returns:
        advantages: torch.Tensor (bs, response_len)
        returns:    torch.Tensor (bs, response_len)  — identical to advantages
    """
    if ig_group_mode not in ("global", "turn_group"):
        raise ValueError(
            f"ig_group_mode='{ig_group_mode}' is not supported. Use 'global' or 'turn_group'."
        )

    # ---------- Step A: Unpack turn_records ----------
    turn_reward = turn_records["turn_reward"]   # (N,) float
    prompt_id   = turn_records["prompt_id"]     # (N,) long
    turn_pos    = turn_records["turn_pos"]      # (N,) long — turn index within trajectory
    is_outcome  = turn_records["is_outcome"]    # (N,) bool
    sample_row  = turn_records["sample_row"]    # (N,) long — row in (bs, L)
    span_start  = turn_records["span_start"]    # (N,) long
    span_end    = turn_records["span_end"]      # (N,) long

    device = turn_reward.device
    N = turn_reward.shape[0]

    # ---------- Step B: Build token-level reward tensor (bs, response_len) ----------
    # Place each turn's reward at its last token position (span_end - 1),
    # matching IGPO convention: turn end token carries the reward.
    token_level_rewards = torch.zeros(bs, response_len, device=device, dtype=turn_reward.dtype)
    response_mask = torch.zeros(bs, response_len, device=device, dtype=torch.long)

    for i in range(N):
        row  = sample_row[i].item()
        s    = span_start[i].item()
        e    = span_end[i].item()    # exclusive
        # reward goes to the last token of the span
        token_level_rewards[row, e - 1] = turn_reward[i]
        # mark span as valid response tokens
        response_mask[row, s:e] = 1

    # ---------- Step C: Build masks — f1_mask and ig_mask ----------
    # f1_mask: positions where is_outcome turns land (last token of outcome spans)
    # ig_mask: positions where IG turns land (last token of IG spans)
    #
    # NOTE: This implementation uses the explicit `is_outcome` field in turn_records
    # to mark which turns are F1 (outcome) turns vs. IG (information-gain) turns,
    # placing each turn's reward at position span_end-1 (the last token of the span).
    # This is an intentional adaptation to the IGPO convention that "F1 reward lands
    # on the last valid token of each row" — turn_records is a more flexible per-turn
    # representation that makes the F1/IG distinction explicit rather than inferring
    # it from token position alone.
    f1_mask  = torch.zeros(bs, response_len, device=device, dtype=torch.bool)
    ig_mask  = torch.zeros(bs, response_len, device=device, dtype=torch.bool)
    # token_turn_pos: for each IG token drop-point (span_end-1), store its turn_pos.
    # Only meaningful at ig_mask positions; used by turn_group normalization.
    token_turn_pos = torch.zeros(bs, response_len, device=device, dtype=torch.long)

    for i in range(N):
        row = sample_row[i].item()
        e   = span_end[i].item()
        if is_outcome[i].item():
            f1_mask[row, e - 1] = True
        else:
            ig_mask[row, e - 1] = True
            token_turn_pos[row, e - 1] = turn_pos[i]

    # ---------- Step D: Build group_ids (bs,) from prompt_id ----------
    # For ig_group_mode="global": one unique group per prompt_id value.
    # For ig_group_mode="turn_group": IG turns are further grouped by (prompt_id, turn_pos);
    #   the (bs,) group_ids here are still prompt-level (used for F1 normalization and fallback).
    # We need one prompt_id per sample_row; use the first turn belonging to each row.
    # All turns of the same sample share the same prompt_id by construction.
    row_prompt = torch.zeros(bs, device=device, dtype=torch.long)
    for i in range(N):
        row_prompt[sample_row[i].item()] = prompt_id[i]

    unique_prompts, inverse = torch.unique(row_prompt, return_inverse=True)
    group_ids = inverse           # (bs,)  consecutive 0-based group ids
    num_groups = unique_prompts.shape[0]

    # Expand to (bs, response_len) for scatter ops
    group_ids_expanded = group_ids.unsqueeze(1).expand(-1, response_len)  # (bs, L)

    # ---------- Step E: Compute group statistics ----------
    def compute_group_stats(mask):
        """Compute per-group mean and std at masked positions."""
        flat_mask     = mask.view(-1)                          # (bs*L,)
        flat_rewards  = token_level_rewards.view(-1)          # (bs*L,)
        flat_gids     = group_ids_expanded.reshape(-1)        # (bs*L,)

        valid_idx = flat_mask.nonzero(as_tuple=True)[0]
        if valid_idx.numel() == 0:
            return (
                torch.zeros(num_groups, device=device),
                torch.ones(num_groups, device=device),
            )

        valid_rewards = flat_rewards[valid_idx]
        valid_gids    = flat_gids[valid_idx]

        group_sum   = torch.zeros(num_groups, device=device).scatter_add_(0, valid_gids, valid_rewards)
        group_count = torch.zeros(num_groups, device=device).scatter_add_(
            0, valid_gids, torch.ones_like(valid_rewards)
        )
        group_mean  = group_sum / group_count.clamp(min=1.0)

        expanded_mean = group_mean[valid_gids]
        sq_diff       = (valid_rewards - expanded_mean) ** 2
        group_sq_sum  = torch.zeros(num_groups, device=device).scatter_add_(0, valid_gids, sq_diff)
        group_var     = group_sq_sum / group_count.clamp(min=1.0)
        group_std     = torch.sqrt(group_var + 1e-8)
        group_std     = torch.where(group_count <= 1, torch.ones_like(group_std), group_std)

        return group_mean, group_std

    # ---------- Step F: Normalize ----------
    normalized_rewards = torch.zeros_like(token_level_rewards)

    if info_gain_norm_mode == "separate":
        # F1 part
        f1_mean, f1_std = compute_group_stats(f1_mask)
        f1_mean_map = f1_mean[group_ids_expanded]
        f1_std_map  = f1_std[group_ids_expanded]
        norm_f1 = token_level_rewards - f1_mean_map
        if norm_by_std:
            norm_f1 = norm_f1 / (f1_std_map + epsilon)
        normalized_rewards = torch.where(f1_mask, norm_f1, normalized_rewards)

        # IG part
        if ig_group_mode == "global":
            # Global: normalize all IG turns sharing a prompt_id together (IGPO default)
            ig_mean, ig_std = compute_group_stats(ig_mask)
            ig_mean_map = ig_mean[group_ids_expanded]
            ig_std_map  = ig_std[group_ids_expanded]
            norm_ig = token_level_rewards - ig_mean_map
            if norm_by_std:
                norm_ig = norm_ig / (ig_std_map + epsilon)
            normalized_rewards = torch.where(ig_mask, norm_ig, normalized_rewards)
        else:
            # turn_group: normalize IG turns by (prompt_id, turn_pos) group
            # Extract values at IG drop-points
            flat_ig_mask   = ig_mask.view(-1)
            ig_valid_idx   = flat_ig_mask.nonzero(as_tuple=True)[0]

            if ig_valid_idx.numel() > 0:
                flat_rewards      = token_level_rewards.view(-1)
                flat_row_prompt   = group_ids.unsqueeze(1).expand(-1, response_len).reshape(-1)
                flat_turn_pos     = token_turn_pos.view(-1)

                ig_rewards    = flat_rewards[ig_valid_idx]       # (M,) rewards at IG positions
                ig_prompt_ids = flat_row_prompt[ig_valid_idx]    # (M,) prompt group ids
                ig_turn_pos   = flat_turn_pos[ig_valid_idx]      # (M,) turn positions

                # Build fine-grained turn-group ids from (prompt_id, turn_pos) pairs
                tg_keys = torch.stack([ig_prompt_ids, ig_turn_pos], dim=1)  # (M, 2)
                _, tg_ids = torch.unique(tg_keys, dim=0, return_inverse=True)  # (M,)

                # Compute turn-group mean/std
                tg_mean, tg_std, tg_size = _grouped_mean_std(ig_rewards, tg_ids, norm_by_std)

                # Compute prompt-level (global) mean/std for fallback
                g_mean, g_std, _ = _grouped_mean_std(ig_rewards, ig_prompt_ids, norm_by_std)

                # Apply fallback: where turn-group size < min_group_size, use prompt-level stats.
                # The fallback uses prompt-level statistics (including the sample itself) —
                # this is standard GRPO normalization semantics (not leave-one-out), which is the intended design.
                fallback = tg_size < min_group_size
                final_mean = torch.where(fallback, g_mean, tg_mean)
                final_std  = torch.where(fallback, g_std, tg_std)

                # Compute normalized values at IG positions
                norm_ig_values = ig_rewards - final_mean
                if norm_by_std:
                    norm_ig_values = norm_ig_values / (final_std + epsilon)

                # Scatter back to (bs, response_len) at IG drop-points
                norm_ig_flat = torch.zeros(bs * response_len, device=device, dtype=token_level_rewards.dtype)
                norm_ig_flat[ig_valid_idx] = norm_ig_values
                norm_ig = norm_ig_flat.view(bs, response_len)
                normalized_rewards = torch.where(ig_mask, norm_ig, normalized_rewards)

    else:  # joint
        joint_mask = f1_mask | ig_mask
        g_mean, g_std = compute_group_stats(joint_mask)
        mean_map = g_mean[group_ids_expanded]
        std_map  = g_std[group_ids_expanded]
        norm_val = token_level_rewards - mean_map
        if norm_by_std:
            norm_val = norm_val / (std_map + epsilon)
        normalized_rewards = torch.where(joint_mask, norm_val, normalized_rewards)

    # ---------- Step G: Turn-level discounted accumulation + scatter ----------
    discounted_returns = _compute_turn_level_advantage(
        normalized_rewards=normalized_rewards,
        response_mask=response_mask,
        gamma=gamma,
        bsz=bs,
        seq_len=response_len,
        device=device,
        turn_boundary_mask=f1_mask | ig_mask,
    )

    return discounted_returns, discounted_returns


def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, eos_mask: torch.Tensor,
                                                  gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++. 
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * eos_mask[:, t]

        advantages = verl_F.masked_whiten(returns, eos_mask)
        advantages = advantages * eos_mask

    return advantages, returns


def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor,
                                    eos_mask: torch.Tensor):
    """
    Compute advantage for ReMax, operating only on Outcome reward 
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        returns = (token_level_rewards * eos_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-10.0, max=10.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
