# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DeepResearcher** is an end-to-end RL training framework for LLM-based deep research agents. It trains a Qwen2.5-7B model to iteratively search the web, read pages, reason, and produce answers — trained with GRPO (Group Relative Policy Optimization). Built on top of [veRL](https://github.com/volcengine/verl).

Paper: https://arxiv.org/abs/2504.03160

## Installation

```bash
conda create -n deepresearcher python=3.10 && conda activate deepresearcher
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation
pip3 install -e .
pip3 install -r requirements.txt
```

## Commands

### Training
```bash
export PET_NODE_RANK=0
export VLLM_ATTENTION_BACKEND=XFORMERS
ray start --head
bash train_grpo.sh        # runs verl.trainer.main_ppo with Hydra config
```

### Evaluation
```bash
bash evaluate.sh          # generates rollout_step_0.json
python ./evaluate/cacluate_metrics.py {experiment_name}
```

### Search Handler (required for `online_search` mode)
```bash
# On each search server (configure API keys in scrl/handler/config.yaml first):
python -m scrl.handler.server_handler    # Flask server on :5000

# On the training node (after updating server_url_list in config.yaml):
python -m scrl.handler.handler           # polling coordinator
```

### Formatting
```bash
bash scripts/format.sh    # yapf Google style over verl/, tests/, examples/
```

### Tests
```bash
pip install -e ".[test]"
pytest tests/
pytest tests/ray/test_worker_group_basics.py::test_function_name  # single test
```

> Most tests require a live Ray cluster and GPUs — they are integration tests, not unit tests.

## Architecture

DeepResearcher has two subsystems communicating via **file-based IPC** (designed for cross-machine deployment):

```
Training Process                        Search Handler
────────────────────────────────────    ──────────────────────────────────────
verl.trainer.main_ppo                   scrl/handler/handler.py  (coordinator)
  → RayPPOTrainer.fit()                   → polls signal/signal.json
  → LLMGenerationManager                  → dispatches to server_handler.py
      multi-turn rollout loop           scrl/handler/server_handler.py (Flask :5000)
      vLLM generates tool calls           → WebSearchAgent (Serper/Bing API)
      ──── signal/data.json ────►         → ReadingAgent (Qwen-Plus LLM)
      ◄─── signal/data.json ────
  → NaiveRewardManager
      F1/EM score on <answer> tag
  → GRPO update (n=16 rollouts/prompt)
```

**IPC files:**
- `signal/signal.json` — `{"signal": 1}` = training has queries; `{"signal": 0}` = handler done
- `signal/data.json` — tool call payloads (training writes); search results (handler writes back)

### Key Files

| Path | Purpose |
|---|---|
| `verl/trainer/main_ppo.py` | Entry point; wires Ray workers |
| `verl/trainer/ppo/ray_trainer.py` | Core training loop: rollout → reward → update |
| `verl/trainer/ppo/core_algos.py` | GRPO/GAE advantage computation, KL control |
| `verl/trainer/config/ppo_trainer.yaml` | All hyperparameters (Hydra config) |
| `verl/workers/fsdp_workers.py` | FSDP-backed Actor/Critic/RewardModel workers |
| `verl/workers/rollout/vllm_rollout/` | vLLM rollout engine |
| `verl/utils/reward_score/format_and_f1.py` | Token-level F1 reward for QA datasets |
| `verl/protocol.py` | `DataProto` — central batched tensor data structure |
| `scrl/llm_agent/generation.py` | `LLMGenerationManager` — multi-turn agent loop, tool call parsing |
| `scrl/handler/handler.py` | Training↔search bridge via file signals |
| `scrl/handler/server_handler.py` | Flask API server for search execution |

### Agent Output Format

The model is trained to produce:
```
<think>reasoning</think>
<tool_call>{"name": "web_search", "arguments": {"query": ["..."]}} </tool_call>

OR

<think>reasoning</think>
<answer>final answer</answer>
```

### Search Engine Modes

Controlled by `search_engine` config:
- `"rag"` — Wikipedia/RAG retrieval (offline, single `web_search` tool)
- `"online_search"` — real Google/Bing search + `browse_webpage` tool

### Training Data Flow

1. `RLHFDataset` loads `.parquet` files (NQ, HotpotQA, 2WikiMultihopQA, MuSiQue, TriviaQA, PopQA, Bamboogle)
2. `LLMGenerationManager` runs up to `max_turns=10` of tool call → search → observation cycles
3. Reward: extract `<answer>` tag, compute F1 against ground truth (returns -1.0 for malformed output)
4. GRPO advantage = normalized reward within a group of 16 rollouts per prompt

## Configuration Notes

- `PET_NODE_RANK` must be set before running (single-node: `export PET_NODE_RANK=0`)
- `VLLM_ATTENTION_BACKEND=XFORMERS` must be set before training/evaluation
- API keys go in `scrl/handler/config.yaml` (`serper_api_key` or `azure_bing_search_subscription_key`) and `scrl/handler/server_handler.py` (Qwen-Plus key)
- In `ppo_trainer.yaml`: `rollout.n` is unused by DeepResearcher — use `agent_grpo.n` instead
- Checkpoints: `./ckpts/{project_name}/{experiment_name}/`
- Rollout outputs: `./outputs/{project_name}/{experiment_name}/rollout/`
