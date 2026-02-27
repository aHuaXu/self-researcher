"""Multi-agent reward manager combining LLM Judge scores with rule-based rewards.

Computes per-sample rewards for three agents (planner, executor, writer) using:
  reward_w = final_reward                                (writer: pure judge score)
  reward_e = beta * rule_e + (1 - beta) * final_reward   (executor: rule + judge blend)
  reward_p = alpha * rule_p + (1 - alpha) * final_reward  (planner: rule + judge blend)
"""

import asyncio

from verl import DataProto
from verl.utils.reward_score.llm_judge import LLMJudge
from verl.utils.reward_score.rule_reward import executor_rules, planner_rules


class MultiAgentRewardManager:
    """Reward manager for multi-agent LoRA GRPO training.

    Combines an async LLM judge (for final report quality) with rule-based
    rewards for the planner and executor agents.  Returns plain lists of
    floats so the caller can construct token-level reward tensors.
    """

    def __init__(self, tokenizer, config):
        """
        Args:
            tokenizer: HuggingFace tokenizer (kept for interface compatibility).
            config: OmegaConf-like object with attributes:
                reward.alpha            - planner rule weight in [0, 1]
                reward.beta             - executor rule weight in [0, 1]
                reward.judge_model      - model name for the LLM judge
                reward.judge_base_url   - OpenAI-compatible base URL
                reward.judge_api_key    - API key
                reward.judge_max_concurrent - concurrency limit for judge calls
        """
        self.tokenizer = tokenizer
        self.alpha = config.reward.alpha
        self.beta = config.reward.beta
        self.judge = LLMJudge(
            model=config.reward.judge_model,
            base_url=config.reward.judge_base_url,
            api_key=config.reward.judge_api_key,
            max_concurrent=config.reward.judge_max_concurrent,
        )

    def __call__(self, data: DataProto) -> dict:
        """Compute rewards for all three agents.

        Expects ``data.non_tensor_batch`` to contain:
          - ``queries``            : list[str]  -- research questions
          - ``plan_texts``         : list[str]  -- planner outputs
          - ``exec_trajectories``  : list[list[dict]] -- executor tool-call trajectories
          - ``exec_actual_turns``  : list[int]  -- actual turns used by executor
          - ``exec_max_turns``     : int        -- maximum allowed turns
          - ``final_reports``      : list[str]  -- writer outputs

        Returns:
            dict with keys ``'planner'``, ``'executor'``, ``'writer'``.
            Each value is a ``list[float]`` of per-sample rewards in [0, 1].
        """
        queries = data.non_tensor_batch['queries']
        plan_texts = data.non_tensor_batch['plan_texts']
        exec_trajectories = data.non_tensor_batch['exec_trajectories']
        exec_actual_turns = data.non_tensor_batch['exec_actual_turns']
        exec_max_turns = data.non_tensor_batch['exec_max_turns']
        final_reports = data.non_tensor_batch['final_reports']

        batch_size = len(queries)

        # --- LLM Judge scores (async) ---
        try:
            final_rewards = asyncio.run(
                self.judge.score_batch(list(queries), list(final_reports))
            )
        except Exception as e:
            print(
                f"[MultiAgentReward] ERROR: judge.score_batch failed: {e}, "
                f"using 0.0 for all {batch_size} samples",
                flush=True,
            )
            final_rewards = [0.0] * batch_size

        # --- Rule-based scores ---
        rule_p_scores = [planner_rules(plan_texts[i]) for i in range(batch_size)]
        rule_e_scores = [
            executor_rules(
                exec_trajectories[i],
                exec_max_turns if isinstance(exec_max_turns, int) else exec_max_turns[i],
                exec_actual_turns[i],
            )
            for i in range(batch_size)
        ]

        # --- Combine per formula ---
        alpha = self.alpha
        beta = self.beta

        rewards_planner = [
            alpha * rule_p_scores[i] + (1 - alpha) * final_rewards[i]
            for i in range(batch_size)
        ]
        rewards_executor = [
            beta * rule_e_scores[i] + (1 - beta) * final_rewards[i]
            for i in range(batch_size)
        ]
        rewards_writer = list(final_rewards)

        # --- Log a few samples for debugging ---
        num_examine = min(2, batch_size)
        for i in range(num_examine):
            print(
                f"[MultiAgentReward] sample {i}: "
                f"judge={final_rewards[i]:.3f} "
                f"rule_p={rule_p_scores[i]:.3f} rule_e={rule_e_scores[i]:.3f} | "
                f"planner={rewards_planner[i]:.3f} "
                f"executor={rewards_executor[i]:.3f} "
                f"writer={rewards_writer[i]:.3f}"
            )

        return {
            'planner': rewards_planner,
            'executor': rewards_executor,
            'writer': rewards_writer,
        }
