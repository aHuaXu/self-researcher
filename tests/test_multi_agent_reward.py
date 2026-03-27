"""Unit tests for the dual-agent reward manager."""

import pytest
import importlib.util
import sys
import os
from types import ModuleType

# We import compute_f1_reward directly from the file to avoid verl's
# heavy dependency chain. The reward manager class needs rule_reward imports
# which are lightweight.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock verl.DataProto so the module can import
_verl_mod = ModuleType("verl")
_verl_mod.__path__ = [os.path.join(os.path.dirname(__file__), "..", "verl")]
_verl_mod.__package__ = "verl"


class _FakeDataProto:
    pass


_verl_mod.DataProto = _FakeDataProto

# Need to override before importing submodules
sys.modules["verl"] = _verl_mod
sys.modules["verl.protocol"] = ModuleType("verl.protocol")

# Mock the LLM judge (not needed for dual-agent)
_judge_mod = ModuleType("verl.utils.reward_score.llm_judge")
_judge_mod.LLMJudge = None
sys.modules["verl.utils.reward_score.llm_judge"] = _judge_mod

from verl.workers.reward_manager.multi_agent import (
    compute_f1_reward,
    MultiAgentRewardManager,
    DualAgentRewards,
)


class TestComputeF1Reward:
    def test_exact_match(self):
        score = compute_f1_reward("Stuart Rosenberg", "Stuart Rosenberg")
        assert score == 1.0

    def test_case_insensitive(self):
        score = compute_f1_reward("stuart rosenberg", "Stuart Rosenberg")
        assert score == 1.0

    def test_partial_match(self):
        score = compute_f1_reward("Stuart", "Stuart Rosenberg")
        # precision=1.0 (1/1), recall=0.5 (1/2), f1=2*1*0.5/1.5=0.667
        assert abs(score - 2/3) < 0.01

    def test_no_match(self):
        score = compute_f1_reward("completely wrong", "Stuart Rosenberg")
        assert score == 0.0

    def test_empty_prediction(self):
        assert compute_f1_reward("", "answer") == 0.0

    def test_empty_golden(self):
        assert compute_f1_reward("answer", "") == 0.0

    def test_both_empty(self):
        assert compute_f1_reward("", "") == 0.0

    def test_multi_answer_split(self):
        golden = "France<|answer_split|>French Republic<|answer_split|>République française"
        score = compute_f1_reward("France", golden)
        assert score == 1.0

    def test_multi_answer_best_match(self):
        golden = "Republic of Turkey<|answer_split|>Turkey<|answer_split|>Türkiye"
        score = compute_f1_reward("Turkey", golden)
        assert score == 1.0

    def test_punctuation_ignored(self):
        score = compute_f1_reward("Stuart Rosenberg!", "Stuart Rosenberg")
        assert score == 1.0

    def test_extra_whitespace(self):
        score = compute_f1_reward("  Stuart   Rosenberg  ", "Stuart Rosenberg")
        assert score == 1.0

    def test_superset_prediction(self):
        score = compute_f1_reward(
            "The answer is Stuart Rosenberg who directed the film",
            "Stuart Rosenberg"
        )
        # recall=1.0 (2/2), precision=2/9, f1=2*(2/9)*1/(2/9+1)=4/11≈0.364
        assert score > 0.3
        assert score < 1.0

    def test_yes_no_exact(self):
        assert compute_f1_reward("yes", "yes") == 1.0
        assert compute_f1_reward("no", "no") == 1.0
        assert compute_f1_reward("yes", "no") == 0.0


class TestMultiAgentRewardManager:
    """Test the reward manager with mocked data."""

    @pytest.fixture
    def config(self):
        class RewardConfig:
            alpha = 0.3
            beta = 0.4
            max_turns = 5

        class Config:
            reward = RewardConfig()

        return Config()

    @pytest.fixture
    def manager(self, config):
        return MultiAgentRewardManager(tokenizer=None, config=config)

    def test_returns_dual_agent_rewards(self, manager):
        class FakeData:
            non_tensor_batch = {
                "final_answers": ["France", "Stuart Rosenberg"],
                "golden_answers": ["France", "Stuart Rosenberg"],
                "plan_texts": [
                    "<plan>\n1. [INDEPENDENT] Q1\n2. [DEPENDS:1] Q2\n3. [DEPENDS:2] Q3\n</plan>",
                    "<plan>\n1. [INDEPENDENT] Q1\n2. [DEPENDS:1] Q2\n3. [DEPENDS:2] Q3\n</plan>",
                ],
                "exec_trajectories": [
                    [{"tool": "web_search", "result": "some result"}],
                    [{"tool": "web_search", "result": "result"}, {"tool": "browse_webpage", "result": "page"}],
                ],
                "exec_actual_turns": [2, 3],
            }

        result = manager(FakeData())
        assert isinstance(result, DualAgentRewards)
        assert len(result.planner) == 2
        assert len(result.executor) == 2
        assert len(result.f1_scores) == 2

    def test_perfect_f1_gives_high_reward(self, manager):
        class FakeData:
            non_tensor_batch = {
                "final_answers": ["France"],
                "golden_answers": ["France"],
                "plan_texts": ["<plan>\n1. [INDEPENDENT] Q1\n2. [DEPENDS:1] Q2\n3. [DEPENDS:2] Q3\n</plan>"],
                "exec_trajectories": [[{"tool": "web_search", "result": "ok"}]],
                "exec_actual_turns": [2],
            }

        result = manager(FakeData())
        assert result.f1_scores[0] == 1.0
        # planner: 0.3 * rule_p + 0.7 * 1.0 >= 0.7
        assert result.planner[0] >= 0.7
        # executor: 0.4 * rule_e + 0.6 * 1.0 >= 0.6
        assert result.executor[0] >= 0.6

    def test_zero_f1_gives_low_reward(self, manager):
        class FakeData:
            non_tensor_batch = {
                "final_answers": ["completely wrong answer"],
                "golden_answers": ["France"],
                "plan_texts": ["no plan tags here"],
                "exec_trajectories": [[]],
                "exec_actual_turns": [1],
            }

        result = manager(FakeData())
        assert result.f1_scores[0] == 0.0
        # With f1=0 and likely rule_p=0 (no plan tags), planner reward should be low
        assert result.planner[0] <= 0.3

    def test_alpha_beta_blending(self, config):
        config.reward.alpha = 0.5
        config.reward.beta = 0.5
        mgr = MultiAgentRewardManager(tokenizer=None, config=config)

        class FakeData:
            non_tensor_batch = {
                "final_answers": ["France"],
                "golden_answers": ["France"],
                "plan_texts": ["<plan>\n1. [INDEPENDENT] Q1\n2. [DEPENDS:1] Q2\n3. [DEPENDS:2] Q3\n</plan>"],
                "exec_trajectories": [[{"tool": "web_search", "result": "ok"}]],
                "exec_actual_turns": [2],
            }

        result = mgr(FakeData())
        # f1=1.0, so planner = 0.5*rule_p + 0.5*1.0
        # executor = 0.5*rule_e + 0.5*1.0
        assert result.planner[0] == pytest.approx(
            0.5 * result.rule_p_scores[0] + 0.5 * 1.0, abs=0.01
        )
        assert result.executor[0] == pytest.approx(
            0.5 * result.rule_e_scores[0] + 0.5 * 1.0, abs=0.01
        )

    def test_no_writer_in_result(self, manager):
        """Verify no writer reward exists."""
        class FakeData:
            non_tensor_batch = {
                "final_answers": ["x"],
                "golden_answers": ["x"],
                "plan_texts": ["text"],
                "exec_trajectories": [[]],
                "exec_actual_turns": [1],
            }

        result = manager(FakeData())
        assert not hasattr(result, "writer")
