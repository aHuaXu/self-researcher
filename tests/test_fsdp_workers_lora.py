"""Unit tests for LoRA adapter configuration in fsdp_workers.py.

Verifies:
1. Only 2 adapters are initialized (planner, executor) - no writer
2. adapter_names list in rollout config has exactly 2 entries
3. No 'writer' references remain in the file
"""

import ast
import os
import re

import pytest


FSDP_WORKERS_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "verl", "workers", "fsdp_workers.py",
)

SHARDING_MGR_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "verl", "workers", "sharding_manager", "fsdp_vllm.py",
)

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "verl", "trainer", "config", "ppo_trainer.yaml",
)


@pytest.fixture
def fsdp_source():
    with open(FSDP_WORKERS_PATH) as f:
        return f.read()


@pytest.fixture
def sharding_source():
    with open(SHARDING_MGR_PATH) as f:
        return f.read()


@pytest.fixture
def config_source():
    with open(CONFIG_PATH) as f:
        return f.read()


class TestLoRAAdapterInit:
    """Verify LoRA adapter initialization uses only 2 adapters."""

    def test_no_writer_adapter_init(self, fsdp_source):
        assert 'add_adapter("writer"' not in fsdp_source
        assert "add_adapter('writer'" not in fsdp_source

    def test_planner_adapter_exists(self, fsdp_source):
        assert 'adapter_name="planner"' in fsdp_source

    def test_executor_adapter_added(self, fsdp_source):
        assert 'add_adapter("executor"' in fsdp_source

    def test_adapter_names_list_has_two(self, fsdp_source):
        match = re.search(r"'adapter_names':\s*\[(.+?)\]", fsdp_source)
        assert match is not None, "adapter_names not found"
        names = match.group(1)
        assert "'planner'" in names
        assert "'executor'" in names
        assert "'writer'" not in names

    def test_no_writer_in_fsdp_workers(self, fsdp_source):
        assert "writer" not in fsdp_source

    def test_no_writer_in_sharding_manager(self, sharding_source):
        assert "writer" not in sharding_source


class TestConfigNoWriter:
    """Verify the default config no longer has a writer agent section."""

    def test_no_writer_agent_in_config(self, config_source):
        lines = config_source.split('\n')
        for i, line in enumerate(lines):
            assert 'writer:' not in line or 'agents:' not in config_source[
                max(0, config_source.rfind('\n', 0, config_source.find(line))):
            ], f"Writer agent still in config at line {i+1}"

    def test_planner_agent_in_config(self, config_source):
        assert "planner:" in config_source

    def test_executor_agent_in_config(self, config_source):
        assert "executor:" in config_source


class TestSyntaxValidity:
    """Verify modified files parse without errors."""

    def test_fsdp_workers_valid_syntax(self):
        with open(FSDP_WORKERS_PATH) as f:
            ast.parse(f.read())

    def test_sharding_manager_valid_syntax(self):
        with open(SHARDING_MGR_PATH) as f:
            ast.parse(f.read())
