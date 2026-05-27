"""Unit tests for dual-agent training configuration and script.

Verifies:
1. ppo_trainer.yaml has correct dual-agent fields (no writer, no judge)
2. Training script enables multi_agent and references correct data paths
3. Reward config has alpha, beta, max_turns (no judge fields)
"""

import os

import pytest
import yaml


CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "verl", "trainer", "config", "ppo_trainer.yaml",
)

SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "scripts", "train", "grpo_dual_agent.sh",
)


@pytest.fixture
def config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture
def script_content():
    with open(SCRIPT_PATH) as f:
        return f.read()


class TestPPOTrainerYAML:
    """Verify ppo_trainer.yaml multi_agent section."""

    def test_multi_agent_section_exists(self, config):
        assert "multi_agent" in config

    def test_default_disabled(self, config):
        assert config["multi_agent"]["enable"] is False

    def test_agents_only_planner_executor(self, config):
        agents = config["multi_agent"]["agents"]
        assert "planner" in agents
        assert "executor" in agents
        assert "writer" not in agents

    def test_reward_has_alpha_beta_max_turns(self, config):
        reward = config["multi_agent"]["reward"]
        assert "alpha" in reward
        assert "beta" in reward
        assert "max_turns" in reward

    def test_reward_no_judge_fields(self, config):
        reward = config["multi_agent"]["reward"]
        assert "judge_model" not in reward
        assert "judge_base_url" not in reward
        assert "judge_api_key" not in reward
        assert "judge_max_concurrent" not in reward

    def test_lora_config_present(self, config):
        lora = config["multi_agent"]["lora"]
        assert lora["rank"] == 64
        assert lora["alpha"] == 128
        assert "target_modules" in lora


class TestDualAgentScript:
    """Verify the dual-agent training script."""

    def test_script_exists_and_executable(self):
        assert os.path.isfile(SCRIPT_PATH)
        assert os.access(SCRIPT_PATH, os.X_OK)

    def test_enables_multi_agent(self, script_content):
        assert "multi_agent.enable=true" in script_content

    def test_references_deepresearch_data(self, script_content):
        assert "deepresearch_phase1.parquet" in script_content
        assert "deepresearch_phase1_val.parquet" in script_content

    def test_sets_lora_params(self, script_content):
        assert "multi_agent.lora.rank=64" in script_content
        assert "multi_agent.lora.alpha=128" in script_content

    def test_sets_reward_params(self, script_content):
        assert "multi_agent.reward.alpha=" in script_content
        assert "multi_agent.reward.beta=" in script_content
        assert "multi_agent.reward.max_turns=" in script_content

    def test_no_writer_references(self, script_content):
        assert "writer" not in script_content

    def test_no_judge_reward_config(self, script_content):
        assert "judge_model" not in script_content
        assert "judge_base_url" not in script_content
        assert "judge_max_concurrent" not in script_content

    def test_uses_lower_batch_size(self, script_content):
        assert "data.train_batch_size=16" in script_content

    def test_uses_lower_lr(self, script_content):
        assert "lr=5e-7" in script_content
