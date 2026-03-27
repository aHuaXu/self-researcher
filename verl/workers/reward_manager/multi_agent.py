"""Multi-agent reward manager for dual-agent (Planner + Executor) pipeline.

Computes per-sample rewards using F1/EM matching + rule-based rewards:
  reward_p = alpha * rule_p + (1 - alpha) * f1_reward   (planner)
  reward_e = beta  * rule_e + (1 - beta)  * f1_reward   (executor)

No LLM Judge dependency — uses token-level F1 between final_answer and
golden_answer for the outcome reward signal.
"""

from dataclasses import dataclass
from typing import List

from verl import DataProto
from verl.utils.reward_score.format_and_f1 import preprocess_text
from verl.utils.reward_score.rule_reward import executor_rules, planner_rules


@dataclass
class DualAgentRewards:
    """Reward computation results for dual-agent pipeline."""

    planner: List[float]  # Per-question planner rewards
    executor: List[float]  # Per-question executor rewards
    f1_scores: List[float]  # Raw F1 scores (for logging)
    rule_p_scores: List[float]  # Raw planner rule scores
    rule_e_scores: List[float]  # Raw executor rule scores


class MultiAgentRewardManager:
    """Reward manager for dual-agent (Planner + Executor) GRPO training.

    Uses F1 token matching between the executor's final answer and the
    golden answer as the outcome reward. Blends with rule-based rewards
    for each agent.
    """

    def __init__(self, tokenizer, config):
        """
        Args:
            tokenizer: HuggingFace tokenizer (kept for interface compatibility).
            config: OmegaConf-like object with attributes:
                reward.alpha          - planner rule weight in [0, 1]
                reward.beta           - executor rule weight in [0, 1]
                reward.max_turns      - maximum allowed executor turns
        """
        self.tokenizer = tokenizer
        self.alpha = config.reward.alpha
        self.beta = config.reward.beta
        self.max_turns = config.reward.max_turns

    def __call__(self, data: DataProto) -> DualAgentRewards:
        """Compute rewards for planner and executor.

        Expects ``data.non_tensor_batch`` to contain:
          - ``final_answers``      : list[str]  -- executor's final answers
          - ``golden_answers``     : list[str]  -- ground truth answers
          - ``plan_texts``         : list[str]  -- planner outputs
          - ``exec_trajectories``  : list[list[dict]] -- executor tool-call trajectories
          - ``exec_actual_turns``  : list[int]  -- actual turns used by executor

        Returns:
            DualAgentRewards with per-sample reward lists.
        """
        final_answers = data.non_tensor_batch["final_answers"]
        golden_answers = data.non_tensor_batch["golden_answers"]
        plan_texts = data.non_tensor_batch["plan_texts"]
        exec_trajectories = data.non_tensor_batch["exec_trajectories"]
        exec_actual_turns = data.non_tensor_batch["exec_actual_turns"]

        batch_size = len(final_answers)

        # --- F1 outcome scores ---
        f1_scores = []
        for i in range(batch_size):
            score = compute_f1_reward(final_answers[i], golden_answers[i])
            f1_scores.append(score)

        # --- Rule-based scores ---
        rule_p_scores = [planner_rules(plan_texts[i]) for i in range(batch_size)]
        rule_e_scores = [
            executor_rules(
                exec_trajectories[i],
                self.max_turns,
                exec_actual_turns[i],
            )
            for i in range(batch_size)
        ]

        # --- Combine per formula ---
        alpha = self.alpha
        beta = self.beta

        rewards_planner = [
            alpha * rule_p_scores[i] + (1 - alpha) * f1_scores[i]
            for i in range(batch_size)
        ]
        rewards_executor = [
            beta * rule_e_scores[i] + (1 - beta) * f1_scores[i]
            for i in range(batch_size)
        ]

        # --- Log a few samples for debugging ---
        num_examine = min(2, batch_size)
        for i in range(num_examine):
            print(
                f"[DualAgentReward] sample {i}: "
                f"f1={f1_scores[i]:.3f} "
                f"rule_p={rule_p_scores[i]:.3f} rule_e={rule_e_scores[i]:.3f} | "
                f"planner={rewards_planner[i]:.3f} "
                f"executor={rewards_executor[i]:.3f}",
                flush=True,
            )

        return DualAgentRewards(
            planner=rewards_planner,
            executor=rewards_executor,
            f1_scores=f1_scores,
            rule_p_scores=rule_p_scores,
            rule_e_scores=rule_e_scores,
        )


def compute_f1_reward(prediction: str, golden_answer: str) -> float:
    """Compute F1 reward between prediction and golden answer.

    Handles multi-answer format (separated by <|answer_split|>),
    returning the maximum F1 across all valid answers.

    Args:
        prediction: The executor's final answer string.
        golden_answer: Ground truth, possibly with <|answer_split|> separators.

    Returns:
        F1 score in [0.0, 1.0].
    """
    if not prediction or not golden_answer:
        return 0.0

    pred_clean = preprocess_text(prediction.lower())
    if not pred_clean:
        return 0.0

    ground_truths = golden_answer.lower().split("<|answer_split|>")

    max_f1 = 0.0
    for gt in ground_truths:
        gt_clean = preprocess_text(gt)
        if not gt_clean:
            continue

        pred_tokens = set(pred_clean.split())
        gt_tokens = set(gt_clean.split())

        if not pred_tokens or not gt_tokens:
            continue

        common_tokens = pred_tokens & gt_tokens
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(gt_tokens)

        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            max_f1 = max(max_f1, f1)

    return max_f1
