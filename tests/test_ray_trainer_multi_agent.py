"""Unit tests for ray_trainer.py multi-agent loop integration.

Verifies:
1. No 'writer' agent remains in the training loop
2. The trainer correctly accesses MultiAgentResult fields (dataclass, not dict)
3. The trainer correctly accesses DualAgentRewards fields (.planner, .executor)
4. Reward data is assembled correctly from rollout result + gen_batch
"""

import ast
import os
import re

import pytest


TRAINER_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "verl", "trainer", "ppo", "ray_trainer.py",
)


@pytest.fixture
def trainer_source():
    with open(TRAINER_PATH) as f:
        return f.read()


@pytest.fixture
def trainer_ast(trainer_source):
    return ast.parse(trainer_source)


class TestWriterRemoval:
    """Verify all references to writer agent are removed."""

    def test_no_writer_in_agent_loop(self, trainer_source):
        assert "'writer'" not in trainer_source
        assert '"writer"' not in trainer_source

    def test_agent_loop_iterates_planner_executor_only(self, trainer_source):
        match = re.search(
            r"for agent_name in \[(.+?)\]", trainer_source
        )
        assert match is not None, "Agent loop not found"
        agents = match.group(1)
        assert "planner" in agents
        assert "executor" in agents
        assert "writer" not in agents


class TestMultiAgentResultAccess:
    """Verify trainer uses dataclass attribute access (not dict)."""

    def test_accesses_final_answers_attribute(self, trainer_source):
        assert "rollout_result.final_answers" in trainer_source

    def test_accesses_queries_attribute(self, trainer_source):
        assert "rollout_result.queries" in trainer_source

    def test_accesses_plan_texts_attribute(self, trainer_source):
        assert "rollout_result.plan_texts" in trainer_source

    def test_accesses_todo_mapping_attribute(self, trainer_source):
        assert "rollout_result.todo_mapping" in trainer_source

    def test_accesses_planner_outputs_attribute(self, trainer_source):
        assert "rollout_result.planner_outputs" in trainer_source

    def test_accesses_executor_outputs_attribute(self, trainer_source):
        assert "rollout_result.executor_outputs" in trainer_source

    def test_no_dict_style_access(self, trainer_source):
        """No rollout_result['key'] style access in multi-agent block."""
        multi_agent_block = trainer_source[
            trainer_source.find("# --- Multi-agent Rollout ---"):
            trainer_source.find("# --- Validation (shared with single-agent path) ---")
        ]
        assert "rollout_result[" not in multi_agent_block


class TestDualAgentRewardsAccess:
    """Verify trainer accesses DualAgentRewards via attributes."""

    def test_accesses_rewards_planner(self, trainer_source):
        assert "rewards.planner" in trainer_source

    def test_accesses_rewards_executor(self, trainer_source):
        assert "rewards.executor" in trainer_source

    def test_no_dict_style_reward_access(self, trainer_source):
        """No rewards['planner'] or rewards['writer'] style access."""
        multi_agent_block = trainer_source[
            trainer_source.find("# --- Multi-agent Reward ---"):
            trainer_source.find("# --- Validation (shared with single-agent path) ---")
        ]
        assert "rewards['planner']" not in multi_agent_block
        assert "rewards['executor']" not in multi_agent_block
        assert "rewards['writer']" not in multi_agent_block


class TestRewardDataAssembly:
    """Verify reward_data.non_tensor_batch is assembled correctly."""

    def test_contains_required_fields(self, trainer_source):
        multi_agent_block = trainer_source[
            trainer_source.find("# --- Multi-agent Reward ---"):
            trainer_source.find("with _timer('adv'")
        ]
        assert "'final_answers'" in multi_agent_block
        assert "'golden_answers'" in multi_agent_block
        assert "'plan_texts'" in multi_agent_block
        assert "'exec_trajectories'" in multi_agent_block
        assert "'exec_actual_turns'" in multi_agent_block

    def test_golden_answers_from_gen_batch(self, trainer_source):
        """Golden answers must come from gen_batch.non_tensor_batch, not rollout."""
        assert "gen_batch.non_tensor_batch['reward_model']" in trainer_source

    def test_no_exec_max_turns_in_non_tensor_batch(self, trainer_source):
        """exec_max_turns is now in config, not non_tensor_batch."""
        multi_agent_block = trainer_source[
            trainer_source.find("# --- Multi-agent Reward ---"):
            trainer_source.find("with _timer('adv'")
        ]
        assert "'exec_max_turns'" not in multi_agent_block


class TestSyntaxValidity:
    """Verify the file parses without errors."""

    def test_valid_python_syntax(self, trainer_ast):
        assert trainer_ast is not None
