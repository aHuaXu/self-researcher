# Multi-Agent LoRA GRPO Training — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the verl GRPO training framework to support 3-agent (Planner/Executor/Writer) joint training with shared base model + independent LoRA adapters.

**Architecture:** Shared DeepResearcher-7B base model with 3 independent LoRA adapters (one per agent). Rollout runs 3 stages serially (Planner → Executor × N todos → Writer), reward combines LLM Judge scores with rule-based rewards, and each LoRA is updated independently via GRPO advantage.

**Tech Stack:** verl (GRPO trainer), PEFT (LoRA), vLLM (LoRA inference), asyncio + OpenAI SDK (LLM Judge), Hydra (config)

**Spec:** `docs/superpowers/specs/2026-04-23-multi-agent-lora-training-design.md`

---

## File Structure

### New Files

| File | Responsibility |
|---|---|
| `verl/utils/reward_score/rule_reward.py` | Planner/Executor 规则 reward 计算 |
| `verl/utils/reward_score/llm_judge.py` | 异步 LLM Judge 批量打分 |
| `verl/workers/reward_manager/multi_agent.py` | 三 agent reward 组合 + 分发 |
| `scrl/llm_agent/multi_agent_generation.py` | 三阶段 rollout 编排 |
| `train_multi_agent.sh` | 多 agent 训练入口脚本 |
| `data/multi-research.parquet` | 训练数据（研究问题） |
| `tests/test_rule_reward.py` | 规则 reward 单元测试 |
| `tests/test_llm_judge.py` | LLM Judge 单元测试（mock API） |

### Modified Files

| File | Change |
|---|---|
| `verl/trainer/config/ppo_trainer.yaml` | 新增 `multi_agent` 配置段 |
| `scrl/llm_agent/generation.py` | `_generate_with_gpu_padding()` 和 `run_llm_loop()` 加 `lora_request` 参数 |
| `verl/workers/rollout/vllm_rollout/vllm_rollout.py` | `__init__` 加 `enable_lora`；`generate_sequences` 加 `lora_request` 透传 |
| `verl/workers/fsdp_workers.py` | `_build_model_optimizer()` 加 PEFT LoRA 初始化分支 |
| `verl/workers/actor/dp_actor.py` | `update_policy()` 加 `lora_name` 参数做梯度隔离 |
| `verl/trainer/ppo/ray_trainer.py` | `fit()` 加 multi_agent 分支 |

### Unchanged Files

| File | Reason |
|---|---|
| `verl/trainer/ppo/core_algos.py` | `compute_grpo_outcome_advantage()` 调三次即可 |
| `verl/utils/dataset/rl_dataset.py` | `RLHFDataset` 天然兼容新数据格式 |
| `verl/workers/reward_manager/naive.py` | 单 agent 训练继续用 |
| `research_agent/` | 推理侧不变，只复用 prompts 和 tools |

---

## Task 1: 规则 Reward 函数

**Files:**
- Create: `verl/utils/reward_score/rule_reward.py`
- Test: `tests/test_rule_reward.py`

这是最独立的模块，无外部依赖，先实现并测试。

- [ ] **Step 1: Write failing tests for planner_rules**

```python
# tests/test_rule_reward.py
import pytest
from verl.utils.reward_score.rule_reward import planner_rules, executor_rules


class TestPlannerRules:
    def test_good_plan(self):
        """3-7 个不重复子任务，优先级分布合理 → 高分"""
        plan_text = (
            "1. [HIGH] Sub-topic: AI芯片市场规模\n   Search Query: AI chip market size 2024\n"
            "2. [MEDIUM] Sub-topic: 主要玩家竞争格局\n   Search Query: NVIDIA AMD Intel AI chip competition\n"
            "3. [LOW] Sub-topic: 未来技术趋势\n   Search Query: AI chip technology trends 2025\n"
            "4. [HIGH] Sub-topic: 中国AI芯片产业\n   Search Query: China AI chip industry\n"
        )
        score = planner_rules(plan_text)
        assert 0.8 <= score <= 1.0

    def test_too_few_tasks(self):
        """少于 3 个子任务 → 数量分不得"""
        plan_text = (
            "1. [HIGH] Sub-topic: AI芯片\n   Search Query: AI chip\n"
            "2. [HIGH] Sub-topic: 市场\n   Search Query: market\n"
        )
        score = planner_rules(plan_text)
        assert score < 0.8

    def test_too_many_tasks(self):
        """超过 7 个 → 数量分不得"""
        lines = ""
        for i in range(1, 10):
            lines += f"{i}. [HIGH] Sub-topic: topic{i}\n   Search Query: query{i}\n"
        score = planner_rules(lines)
        assert score < 0.8

    def test_all_same_priority(self):
        """全部同一优先级 → 优先级分不得"""
        plan_text = (
            "1. [HIGH] Sub-topic: topic1\n   Search Query: query1\n"
            "2. [HIGH] Sub-topic: topic2\n   Search Query: query2\n"
            "3. [HIGH] Sub-topic: topic3\n   Search Query: query3\n"
        )
        score = planner_rules(plan_text)
        assert score < 1.0

    def test_duplicate_subtopics(self):
        """重复子任务 → 重复分不得"""
        plan_text = (
            "1. [HIGH] Sub-topic: AI chip market size\n   Search Query: AI chip market\n"
            "2. [MEDIUM] Sub-topic: AI chip market size analysis\n   Search Query: AI chip market analysis\n"
            "3. [LOW] Sub-topic: AI chip future\n   Search Query: AI chip future\n"
        )
        score = planner_rules(plan_text)
        assert score < 1.0

    def test_empty_plan(self):
        score = planner_rules("")
        assert score == 0.0


class TestExecutorRules:
    def test_good_execution(self):
        """搜索成功 + 有 browse + 内容非空 + 没超轮次 → 满分"""
        trajectory = [
            {"tool": "web_search", "result": '{"results": [{"title": "AI"}]}'},
            {"tool": "browse_webpage", "result": '{"url": "https://...", "information": [{"page_summary": "x" * 200}]}'},
        ]
        score = executor_rules(trajectory, max_turns=10, actual_turns=5)
        assert score == 1.0

    def test_no_search(self):
        """没有搜索 → 搜索分不得"""
        trajectory = []
        score = executor_rules(trajectory, max_turns=10, actual_turns=0)
        assert score < 0.5

    def test_no_browse(self):
        """有搜索但没 browse → browse 分不得"""
        trajectory = [
            {"tool": "web_search", "result": '{"results": [{"title": "AI"}]}'},
        ]
        score = executor_rules(trajectory, max_turns=10, actual_turns=3)
        assert score < 1.0

    def test_hit_max_turns(self):
        """达到 max_turns → 效率分不得"""
        trajectory = [
            {"tool": "web_search", "result": '{"results": [{"title": "AI"}]}'},
            {"tool": "browse_webpage", "result": '{"information": [{"page_summary": "x" * 200}]}'},
        ]
        score = executor_rules(trajectory, max_turns=10, actual_turns=10)
        assert score < 1.0

    def test_empty_search_result(self):
        """搜索返回空 → 搜索分不得"""
        trajectory = [
            {"tool": "web_search", "result": "[]"},
        ]
        score = executor_rules(trajectory, max_turns=10, actual_turns=1)
        assert score < 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rule_reward.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'verl.utils.reward_score.rule_reward'`

- [ ] **Step 3: Implement rule_reward.py**

```python
# verl/utils/reward_score/rule_reward.py
"""Rule-based rewards for Planner and Executor agents."""
import re
from typing import List, Dict


def planner_rules(plan_text: str) -> float:
    """Compute rule-based reward for Planner output.

    Scoring:
        - Task count in [3, 7]: +1.0
        - No duplicate sub-topics (keyword overlap < 0.6): +1.0
        - Priority distribution (not all same level): +0.5
    Normalized to [0, 1] by dividing by 2.5.
    """
    if not plan_text.strip():
        return 0.0

    pattern = r'(\d+)\.\s*\[(HIGH|MEDIUM|LOW)\]\s*(?:Sub-topic:|子主题[：:])\s*(.+?)(?:\n|$)'
    matches = re.findall(pattern, plan_text, re.IGNORECASE)

    if not matches:
        return 0.0

    score = 0.0
    num_tasks = len(matches)
    priorities = [m[1].upper() for m in matches]
    subtopics = [m[2].strip().lower() for m in matches]

    # Rule 1: task count in [3, 7]
    if 3 <= num_tasks <= 7:
        score += 1.0

    # Rule 2: no duplicate sub-topics (pairwise keyword overlap < 0.6)
    has_duplicate = False
    for i in range(len(subtopics)):
        words_i = set(subtopics[i].split())
        if not words_i:
            continue
        for j in range(i + 1, len(subtopics)):
            words_j = set(subtopics[j].split())
            if not words_j:
                continue
            overlap = len(words_i & words_j) / min(len(words_i), len(words_j))
            if overlap >= 0.6:
                has_duplicate = True
                break
        if has_duplicate:
            break
    if not has_duplicate:
        score += 1.0

    # Rule 3: priority distribution (at least 2 distinct levels)
    if len(set(priorities)) >= 2:
        score += 0.5

    return score / 2.5


def executor_rules(trajectory: List[Dict], max_turns: int, actual_turns: int) -> float:
    """Compute rule-based reward for Executor output.

    Scoring:
        - Successful search with non-empty result: +1.0
        - Called browse_webpage: +1.0
        - Browse returned non-empty content (len >= 50): +1.0
        - Did not hit max_turns: +1.0
    Normalized to [0, 1] by dividing by 4.
    """
    score = 0.0

    has_search = False
    has_browse = False
    has_content = False

    for step in trajectory:
        tool = step.get("tool", "")
        result = step.get("result", "")

        if tool == "web_search" and len(result) > 10:
            has_search = True

        if tool == "browse_webpage":
            has_browse = True
            if len(result) > 50:
                has_content = True

    if has_search:
        score += 1.0
    if has_browse:
        score += 1.0
    if has_content:
        score += 1.0
    if actual_turns < max_turns:
        score += 1.0

    return score / 4.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rule_reward.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add verl/utils/reward_score/rule_reward.py tests/test_rule_reward.py
git commit -m "feat: add rule-based reward functions for planner and executor"
```

---

## Task 2: LLM Judge（异步外部 API 打分）

**Files:**
- Create: `verl/utils/reward_score/llm_judge.py`
- Test: `tests/test_llm_judge.py`

独立模块，通过外部 API 调用 LLM 对报告打分。

- [ ] **Step 1: Write failing tests for LLMJudge**

```python
# tests/test_llm_judge.py
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from verl.utils.reward_score.llm_judge import LLMJudge, parse_score


class TestParseScore:
    def test_plain_number(self):
        assert parse_score("7") == 7.0

    def test_number_with_text(self):
        assert parse_score("Score: 8/10") == 8.0

    def test_decimal(self):
        assert parse_score("7.5") == 7.5

    def test_no_number(self):
        assert parse_score("no score here") == 0.0

    def test_multiple_numbers_takes_first(self):
        assert parse_score("7 out of 10 points") == 7.0

    def test_clamp_above_10(self):
        assert parse_score("15") == 10.0

    def test_clamp_below_1(self):
        assert parse_score("0") == 0.0


class TestLLMJudge:
    def test_score_batch(self):
        """Mock API, verify batch scoring returns normalized scores."""
        judge = LLMJudge(
            model="test-model",
            base_url="http://fake",
            api_key="fake-key",
            max_concurrent=5,
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "8"

        with patch.object(judge.client.chat.completions, 'create',
                          new_callable=AsyncMock, return_value=mock_response):
            queries = ["query1", "query2", "query3"]
            reports = ["report1", "report2", "report3"]
            scores = asyncio.run(judge.score_batch(queries, reports))

        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)
        assert all(s == 0.8 for s in scores)  # 8/10 = 0.8

    def test_score_batch_with_api_error(self):
        """API error returns 0.0 for that item."""
        judge = LLMJudge(
            model="test-model",
            base_url="http://fake",
            api_key="fake-key",
            max_concurrent=5,
        )

        async def side_effect(*args, **kwargs):
            raise Exception("API error")

        with patch.object(judge.client.chat.completions, 'create',
                          new_callable=AsyncMock, side_effect=side_effect):
            scores = asyncio.run(judge.score_batch(["q"], ["r"]))

        assert scores == [0.0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm_judge.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement llm_judge.py**

```python
# verl/utils/reward_score/llm_judge.py
"""Async LLM Judge for scoring research reports."""
import asyncio
import re
from openai import AsyncOpenAI


JUDGE_PROMPT = """You are evaluating a research report generated for a given query.

Query: {query}

Report:
{report}

Rate this report on a scale of 1-10 based on:
- Accuracy: Is the information correct?
- Completeness: Does it cover the core aspects of the query?
- Structure: Is the report logically organized?
- Readability: Is the language clear and fluent?

Output ONLY a single number (1-10), nothing else."""


def parse_score(text: str) -> float:
    """Extract numeric score from judge response. Returns 0.0 if no number found."""
    match = re.search(r'(\d+(?:\.\d+)?)', text.strip())
    if not match:
        return 0.0
    score = float(match.group(1))
    return max(0.0, min(10.0, score))


class LLMJudge:
    def __init__(self, model: str, base_url: str, api_key: str, max_concurrent: int = 50):
        self.model = model
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def score_batch(self, queries: list, reports: list) -> list:
        """Score a batch of (query, report) pairs. Returns list of floats in [0, 1]."""
        tasks = [self._score_one(q, r) for q, r in zip(queries, reports)]
        return await asyncio.gather(*tasks)

    async def _score_one(self, query: str, report: str) -> float:
        async with self.semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{
                        "role": "user",
                        "content": JUDGE_PROMPT.format(query=query, report=report),
                    }],
                    max_tokens=16,
                )
                raw = response.choices[0].message.content or ""
                return parse_score(raw) / 10.0
            except Exception as e:
                print(f"LLM Judge error: {e}")
                return 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm_judge.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add verl/utils/reward_score/llm_judge.py tests/test_llm_judge.py
git commit -m "feat: add async LLM Judge for scoring research reports"
```

---

## Task 3: Hydra 配置 — 新增 multi_agent 段

**Files:**
- Modify: `verl/trainer/config/ppo_trainer.yaml`

在 yaml 末尾加配置段。`multi_agent.enable: false` 默认不走新逻辑。

- [ ] **Step 1: Add multi_agent config section**

在 `verl/trainer/config/ppo_trainer.yaml` 文件末尾追加：

```yaml
# Multi-agent LoRA GRPO training (disabled by default)
multi_agent:
  enable: false
  base_model: GAIR/DeepResearcher-7b

  lora:
    rank: 64
    alpha: 128
    target_modules:
      - q_proj
      - v_proj
      - k_proj
      - o_proj
    dropout: 0.05

  agents:
    planner:
      max_tokens: 1024
    executor:
      max_turns: 10
    writer:
      max_tokens: 4096

  reward:
    judge_model: ""
    judge_base_url: ""
    judge_api_key: ""
    judge_max_concurrent: 50
    alpha: 0.2
    beta: 0.3
```

- [ ] **Step 2: Verify existing config is not broken**

Run: `python -c "from omegaconf import OmegaConf; cfg = OmegaConf.load('verl/trainer/config/ppo_trainer.yaml'); print(cfg.multi_agent.enable)"`
Expected: `False`

Run: `python -c "from omegaconf import OmegaConf; cfg = OmegaConf.load('verl/trainer/config/ppo_trainer.yaml'); print(cfg.actor.use_kl_loss)"`
Expected: Prints existing value, confirming original config unaffected.

- [ ] **Step 3: Commit**

```bash
git add verl/trainer/config/ppo_trainer.yaml
git commit -m "feat: add multi_agent config section to ppo_trainer.yaml"
```

---

## Task 4: MultiAgentRewardManager — 三 agent reward 组合

**Files:**
- Create: `verl/workers/reward_manager/multi_agent.py`

依赖 Task 1 (rule_reward) 和 Task 2 (llm_judge)。

- [ ] **Step 1: Implement MultiAgentRewardManager**

```python
# verl/workers/reward_manager/multi_agent.py
"""Reward manager for multi-agent LoRA training.

Combines LLM Judge final_reward with rule-based rewards for each agent:
  reward_w = final_reward
  reward_e = beta * rule_e + (1 - beta) * final_reward
  reward_p = alpha * rule_p + (1 - alpha) * final_reward
"""
import asyncio
import torch
from verl.protocol import DataProto
from verl.utils.reward_score.llm_judge import LLMJudge
from verl.utils.reward_score.rule_reward import planner_rules, executor_rules


class MultiAgentRewardManager:
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.alpha = config.reward.alpha  # planner rule weight
        self.beta = config.reward.beta    # executor rule weight
        self.judge = LLMJudge(
            model=config.reward.judge_model,
            base_url=config.reward.judge_base_url,
            api_key=config.reward.judge_api_key,
            max_concurrent=config.reward.judge_max_concurrent,
        )

    def __call__(self, data: DataProto) -> dict:
        """Compute rewards for all three agents.

        Expects data.non_tensor_batch to contain:
          - 'queries': list of research questions
          - 'plan_texts': list of planner outputs (raw text)
          - 'exec_trajectories': list of executor trajectories (list of dicts)
          - 'exec_actual_turns': list of ints
          - 'exec_max_turns': int
          - 'final_reports': list of writer outputs (report text)
          - 'planner_response_length': int
          - 'executor_response_length': int
          - 'writer_response_length': int

        Returns dict with keys: 'planner', 'executor', 'writer'
        Each value is a reward tensor of shape (batch_size, respective_response_length).
        """
        queries = data.non_tensor_batch['queries']
        plan_texts = data.non_tensor_batch['plan_texts']
        exec_trajectories = data.non_tensor_batch['exec_trajectories']
        exec_actual_turns = data.non_tensor_batch['exec_actual_turns']
        exec_max_turns = data.non_tensor_batch['exec_max_turns']
        final_reports = data.non_tensor_batch['final_reports']
        batch_size = len(queries)

        # 1. LLM Judge → final_reward for each report
        final_rewards = asyncio.run(
            self.judge.score_batch(queries, final_reports)
        )

        # 2. Rule rewards
        rule_p_scores = [planner_rules(text) for text in plan_texts]
        rule_e_scores = [
            executor_rules(traj, exec_max_turns, turns)
            for traj, turns in zip(exec_trajectories, exec_actual_turns)
        ]

        # 3. Weighted combination
        rewards = {'planner': [], 'executor': [], 'writer': []}
        for i in range(batch_size):
            fr = final_rewards[i]
            rp = rule_p_scores[i]
            re_ = rule_e_scores[i]

            rewards['planner'].append(self.alpha * rp + (1 - self.alpha) * fr)
            rewards['executor'].append(self.beta * re_ + (1 - self.beta) * fr)
            rewards['writer'].append(fr)

        # 4. Convert to token-level reward tensors (reward at last token)
        result = {}
        for agent_name in ['planner', 'executor', 'writer']:
            resp_len = data.non_tensor_batch[f'{agent_name}_response_length']
            reward_tensor = torch.zeros(batch_size, resp_len, dtype=torch.float32)
            valid_lengths = data.non_tensor_batch[f'{agent_name}_valid_lengths']
            for i in range(batch_size):
                reward_tensor[i, valid_lengths[i] - 1] = rewards[agent_name][i]
            result[agent_name] = reward_tensor

        return result
```

- [ ] **Step 2: Commit**

```bash
git add verl/workers/reward_manager/multi_agent.py
git commit -m "feat: add MultiAgentRewardManager with LLM Judge + rule rewards"
```

---

## Task 5: vLLM Rollout — 添加 LoRA 支持

**Files:**
- Modify: `verl/workers/rollout/vllm_rollout/vllm_rollout.py` (lines 57-241)

需要做两件事：(1) 初始化时启用 `enable_lora`；(2) `generate_sequences()` 透传 `lora_request`。

- [ ] **Step 1: Add enable_lora to vLLMRollout.__init__**

在 `verl/workers/rollout/vllm_rollout/vllm_rollout.py` 的 `__init__` 方法中，找到 `LLM()` 初始化（约 line 101-115），在构建 kwargs 时加入 LoRA 参数：

```python
# 在 LLM() 构造参数中添加（约 line 101-115 之间）
# 读取 multi_agent 配置
enable_lora = kwargs.get('enable_lora', False)
max_lora_rank = kwargs.get('max_lora_rank', 64)

# 传给 LLM()
# 在现有的 LLM 参数 dict 中加入：
if enable_lora:
    llm_kwargs['enable_lora'] = True
    llm_kwargs['max_lora_rank'] = max_lora_rank
```

具体改动：需阅读 `__init__` 中 LLM 初始化的精确代码结构来确定插入位置。关键是 `self.inference_engine = LLM(...)` 这行。

- [ ] **Step 2: Add lora_request parameter to generate_sequences**

在 `generate_sequences()` 方法（约 line 156-241）的签名和 `self.inference_engine.generate()` 调用中透传：

```python
# 改前（约 line 156）
def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:

# generate_sequences 内部，找到 self.inference_engine.generate() 调用（约 line 190-194）
# 改前
output = self.inference_engine.generate(...)
# 改后
lora_request = kwargs.get('lora_request', None)
generate_kwargs = {}
if lora_request is not None:
    generate_kwargs['lora_request'] = lora_request
output = self.inference_engine.generate(..., **generate_kwargs)
```

原有调用不传 `lora_request`（默认 None），行为不变。

- [ ] **Step 3: Verify original code path not affected**

Run: `python -c "from verl.workers.rollout.vllm_rollout.vllm_rollout import vLLMRollout; print('import OK')"`
Expected: `import OK`

- [ ] **Step 4: Commit**

```bash
git add verl/workers/rollout/vllm_rollout/vllm_rollout.py
git commit -m "feat: add LoRA support to vLLM rollout (enable_lora + lora_request passthrough)"
```

---

## Task 6: LLMGenerationManager — 添加 lora_request 透传

**Files:**
- Modify: `scrl/llm_agent/generation.py` (lines 292-540)

给 `_generate_with_gpu_padding()` 和 `run_llm_loop()` 加 `lora_request` 可选参数。

- [ ] **Step 1: Add lora_request to _generate_with_gpu_padding**

在 `scrl/llm_agent/generation.py` 中：

```python
# 改前（line 292）
def _generate_with_gpu_padding(self, gen_batch, ...):

# 改后
def _generate_with_gpu_padding(self, gen_batch, ..., lora_request=None):

# 内部调用 self.actor_rollout_wg.generate_sequences() 时（约 line 300, 320）
# 改前
output = self.actor_rollout_wg.generate_sequences(gen_batch)
# 改后
gen_kwargs = {}
if lora_request is not None:
    gen_kwargs['lora_request'] = lora_request
output = self.actor_rollout_wg.generate_sequences(gen_batch, **gen_kwargs)
```

两处调用都要改（line ~300 直接调用 和 line ~320 padding 后调用）。

- [ ] **Step 2: Add lora_request to run_llm_loop**

```python
# 改前（line 387）
def run_llm_loop(self, gen_batch: DataProto, global_steps: int) -> Tuple[List[str], DataProto]:

# 改后
def run_llm_loop(self, gen_batch: DataProto, global_steps: int, lora_request=None) -> Tuple[List[str], DataProto]:

# 内部调用 self._generate_with_gpu_padding() 时（约 line 452）
# 改前
output = self._generate_with_gpu_padding(rollings_active)
# 改后
output = self._generate_with_gpu_padding(rollings_active, lora_request=lora_request)
```

- [ ] **Step 3: Verify import**

Run: `python -c "from scrl.llm_agent.generation import LLMGenerationManager; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add scrl/llm_agent/generation.py
git commit -m "feat: add lora_request parameter to LLMGenerationManager methods"
```

---

## Task 7: MultiAgentGenerationManager — 三阶段 Rollout

**Files:**
- Create: `scrl/llm_agent/multi_agent_generation.py`

继承 `LLMGenerationManager`，实现三阶段串行 rollout。

- [ ] **Step 1: Implement MultiAgentGenerationManager**

```python
# scrl/llm_agent/multi_agent_generation.py
"""Three-stage rollout manager for multi-agent LoRA training.

Stage 1: Planner (single-turn) → TODO list
Stage 2: Executor (multi-turn tool-calling) × N todos → findings per todo
Stage 3: Writer (single-turn) → final report
"""
import re
import copy
from typing import List, Tuple, Dict
from vllm.lora.request import LoRARequest
from verl.protocol import DataProto
from scrl.llm_agent.generation import LLMGenerationManager, GenerationConfig
from research_agent.prompts.planner import get_planner_prompt
from research_agent.prompts.executor import get_executor_prompt
from research_agent.prompts.writer import get_writer_prompt


class MultiAgentGenerationManager(LLMGenerationManager):
    """Three-stage serial rollout, reuses parent class generation and tool-calling."""

    def __init__(self, tokenizer, actor_rollout_wg, config: GenerationConfig,
                 lora_paths: Dict[str, str], is_validation: bool = False):
        """
        Args:
            lora_paths: dict mapping agent name to LoRA adapter path, e.g.
                {"planner": "/path/to/lora_planner",
                 "executor": "/path/to/lora_executor",
                 "writer": "/path/to/lora_writer"}
        """
        super().__init__(tokenizer, actor_rollout_wg, config, is_validation)
        self.lora_planner = LoRARequest("planner", 1, lora_paths["planner"])
        self.lora_executor = LoRARequest("executor", 2, lora_paths["executor"])
        self.lora_writer = LoRARequest("writer", 3, lora_paths["writer"])

    def run_multi_agent_loop(
        self, gen_batch: DataProto, global_steps: int
    ) -> Dict:
        """Run three-stage rollout for multi-agent training.

        Returns dict with keys: planner, executor, writer, metadata.
        Each agent entry contains output DataProto with token sequences + log_probs.
        metadata contains: queries, plan_texts, exec_trajectories, final_reports, etc.
        """
        questions = self._extract_questions(gen_batch)

        # Stage 1 — Planner (single-turn generation)
        planner_batch = self._build_planner_batch(questions, gen_batch)
        plan_outputs = self._generate_with_gpu_padding(
            planner_batch, lora_request=self.lora_planner
        )
        plan_texts = self._decode_outputs(plan_outputs)
        parsed_todos = [self._parse_todos(text) for text in plan_texts]

        # Stage 2 — Executor (multi-turn tool-calling, each TODO independently)
        exec_prompts, todo_mapping = self._build_executor_batch(
            questions, parsed_todos, gen_batch
        )
        exec_message_strs, exec_outputs = self.run_llm_loop(
            exec_prompts, global_steps, lora_request=self.lora_executor
        )
        grouped_findings = self._group_findings(
            exec_message_strs, todo_mapping, len(questions)
        )

        # Stage 3 — Writer (single-turn generation)
        writer_batch = self._build_writer_batch(
            questions, plan_texts, grouped_findings, gen_batch
        )
        writer_outputs = self._generate_with_gpu_padding(
            writer_batch, lora_request=self.lora_writer
        )
        final_reports = self._decode_outputs(writer_outputs)

        return {
            'planner': plan_outputs,
            'executor': exec_outputs,
            'writer': writer_outputs,
            'metadata': {
                'queries': questions,
                'plan_texts': plan_texts,
                'parsed_todos': parsed_todos,
                'exec_trajectories': self._extract_trajectories(exec_message_strs),
                'final_reports': final_reports,
            },
        }

    def _extract_questions(self, gen_batch: DataProto) -> List[str]:
        """Extract research questions from batch."""
        prompts = gen_batch.non_tensor_batch.get('raw_prompt', [])
        questions = []
        for p in prompts:
            if isinstance(p, list):
                for msg in p:
                    if msg.get('role') == 'user':
                        questions.append(msg['content'])
                        break
            else:
                questions.append(str(p))
        return questions

    def _build_planner_batch(self, questions: List[str], ref_batch: DataProto) -> DataProto:
        """Build tokenized batch for Planner from questions."""
        messages_list = [get_planner_prompt(q) for q in questions]
        return self._tokenize_messages_to_batch(messages_list, ref_batch)

    def _build_executor_batch(
        self, questions: List[str], parsed_todos: List[List[Dict]], ref_batch: DataProto
    ) -> Tuple[DataProto, List[int]]:
        """Build tokenized batch for Executor, one prompt per TODO item.

        Returns (batch, todo_mapping) where todo_mapping[i] = index of original question.
        """
        messages_list = []
        todo_mapping = []
        for q_idx, (question, todos) in enumerate(zip(questions, parsed_todos)):
            for todo in todos:
                sub_topic = todo.get('sub_topic', question)
                context = todo.get('search_query', '')
                msgs = get_executor_prompt(sub_topic, context)
                messages_list.append(msgs)
                todo_mapping.append(q_idx)
        batch = self._tokenize_messages_to_batch(messages_list, ref_batch)
        return batch, todo_mapping

    def _build_writer_batch(
        self, questions: List[str], plan_texts: List[str],
        grouped_findings: List[str], ref_batch: DataProto
    ) -> DataProto:
        """Build tokenized batch for Writer."""
        messages_list = []
        for q, findings_text in zip(questions, grouped_findings):
            msgs = get_writer_prompt(q, findings_text)
            messages_list.append(msgs)
        return self._tokenize_messages_to_batch(messages_list, ref_batch)

    def _tokenize_messages_to_batch(
        self, messages_list: List[List[Dict]], ref_batch: DataProto
    ) -> DataProto:
        """Tokenize a list of message lists into a DataProto batch.

        Uses self.tokenizer.apply_chat_template() for each message list,
        then pads to equal length and constructs DataProto.
        """
        import torch
        token_ids_list = []
        for msgs in messages_list:
            ids = self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
            token_ids_list.append(ids)

        max_len = max(len(ids) for ids in token_ids_list)
        pad_id = self.tokenizer.pad_token_id or 0
        padded = []
        attention_masks = []
        for ids in token_ids_list:
            pad_len = max_len - len(ids)
            padded.append([pad_id] * pad_len + ids)
            attention_masks.append([0] * pad_len + [1] * len(ids))

        batch = DataProto.from_dict({
            'input_ids': torch.tensor(padded, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
            'position_ids': torch.stack([
                torch.arange(max_len) for _ in range(len(messages_list))
            ]),
        })
        return batch

    def _decode_outputs(self, outputs: DataProto) -> List[str]:
        """Decode token IDs from output DataProto to strings."""
        responses = outputs.batch['responses']
        texts = []
        for i in range(responses.shape[0]):
            ids = responses[i].tolist()
            ids = [t for t in ids if t != self.tokenizer.pad_token_id and t != 0]
            texts.append(self.tokenizer.decode(ids, skip_special_tokens=True))
        return texts

    def _parse_todos(self, plan_text: str) -> List[Dict]:
        """Parse Planner output into TODO list."""
        pattern = r'(\d+)\.\s*\[(HIGH|MEDIUM|LOW)\]\s*(?:Sub-topic:|子主题[：:])\s*(.+?)(?:\n\s*(?:Search Query:|搜索查询[：:])\s*(.+?))?(?=\n\d+\.\s*\[|</todos>|$)'
        matches = re.findall(pattern, plan_text, re.IGNORECASE | re.DOTALL)
        todos = []
        for m in matches:
            todos.append({
                'index': int(m[0]),
                'priority': m[1].upper(),
                'sub_topic': m[2].strip(),
                'search_query': m[3].strip() if m[3] else '',
            })
        if not todos:
            todos = [{'index': 1, 'priority': 'HIGH', 'sub_topic': plan_text[:200], 'search_query': ''}]
        return todos

    def _group_findings(
        self, exec_outputs: List[str], todo_mapping: List[int], num_questions: int
    ) -> List[str]:
        """Group executor outputs by original question index."""
        grouped = [[] for _ in range(num_questions)]
        for i, output in enumerate(exec_outputs):
            q_idx = todo_mapping[i]
            grouped[q_idx].append(output)
        return ["\n\n---\n\n".join(findings) for findings in grouped]

    def _extract_trajectories(self, exec_outputs: List[str]) -> List[List[Dict]]:
        """Extract tool call trajectories from executor output strings.

        This is a simplified extraction — actual trajectories are tracked
        during run_llm_loop and should be stored in non_tensor_batch.
        """
        trajectories = []
        for output in exec_outputs:
            traj = []
            for tool_name in ['web_search', 'browse_webpage']:
                if tool_name in output:
                    traj.append({'tool': tool_name, 'result': output})
            trajectories.append(traj)
        return trajectories
```

- [ ] **Step 2: Verify import**

Run: `python -c "from scrl.llm_agent.multi_agent_generation import MultiAgentGenerationManager; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scrl/llm_agent/multi_agent_generation.py
git commit -m "feat: add MultiAgentGenerationManager for three-stage LoRA rollout"
```

---

## Task 8: FSDP Workers — PEFT LoRA 初始化

**Files:**
- Modify: `verl/workers/fsdp_workers.py` (line 143-295, `_build_model_optimizer`)

当 `config.multi_agent.enable` 时，用 PEFT 给模型加三个 LoRA adapter。

- [ ] **Step 1: Add LoRA initialization branch**

在 `_build_model_optimizer()` 方法中，在模型创建后（约 line 212）、FSDP wrap 前（约 line 243）插入：

```python
# 在 actor_module = AutoModelForCausalLM.from_pretrained(...) 之后
# 在 auto_wrap_policy = get_fsdp_wrap_policy(...) 之前

# Multi-agent LoRA initialization
if hasattr(self.config, 'multi_agent') and self.config.multi_agent.get('enable', False):
    from peft import get_peft_model, LoraConfig

    lora_cfg = self.config.multi_agent.lora
    lora_config = LoraConfig(
        r=lora_cfg.rank,
        lora_alpha=lora_cfg.alpha,
        target_modules=list(lora_cfg.target_modules),
        lora_dropout=lora_cfg.dropout,
    )
    actor_module = get_peft_model(actor_module, lora_config, adapter_name="planner")
    actor_module.add_adapter("executor", lora_config)
    actor_module.add_adapter("writer", lora_config)

    # Override wrap policy for LoRA
    auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, is_lora=True)
```

注意：需要确保 `get_fsdp_wrap_policy` 调用使用 `is_lora=True`。原有代码路径不传 `is_lora`（默认 False），行为不变。

- [ ] **Step 2: Verify import not broken**

Run: `python -c "from verl.workers.fsdp_workers import ActorRolloutRefWorker; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add verl/workers/fsdp_workers.py
git commit -m "feat: add PEFT LoRA initialization for multi-agent training in FSDP workers"
```

---

## Task 9: Actor Update — LoRA 梯度隔离

**Files:**
- Modify: `verl/workers/actor/dp_actor.py` (line 226-332, `update_policy`)

加 `lora_name` 参数，更新时只开启目标 LoRA 的梯度。

- [ ] **Step 1: Add lora_name parameter to update_policy**

在 `verl/workers/actor/dp_actor.py` 的 `update_policy` 方法中：

```python
# 改前（line 226）
def update_policy(self, data: DataProto, tokenizer) -> dict:

# 改后
def update_policy(self, data: DataProto, tokenizer, lora_name: str = None) -> dict:

# 在方法开头（约 line 230 之后），forward pass 之前：
if lora_name is not None:
    self.actor_module.set_adapter(lora_name)
    for name, param in self.actor_module.named_parameters():
        if "lora_" in name:
            param.requires_grad = (lora_name in name)
```

方法末尾不需要恢复，因为下次调用会重新设置。

原有调用不传 `lora_name`（默认 None），跳过这段逻辑，行为不变。

- [ ] **Step 2: Verify import**

Run: `python -c "from verl.workers.actor.dp_actor import DataParallelPPOActor; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add verl/workers/actor/dp_actor.py
git commit -m "feat: add LoRA gradient isolation to actor update_policy"
```

---

## Task 10: Training Loop — multi_agent 分支

**Files:**
- Modify: `verl/trainer/ppo/ray_trainer.py` (line 942-1177, `fit()`)

在 `fit()` 中新增 `multi_agent.enable` 分支：替换 generation manager、reward manager，三次 advantage + update。

- [ ] **Step 1: Add multi-agent initialization in fit()**

在 `RayPPOTrainer` 的初始化阶段（`__init__` 或 `fit()` 开头），根据 config 选择 generation manager 和 reward manager：

```python
# 在 fit() 方法开头，现有初始化之后
if self.config.multi_agent.enable:
    from scrl.llm_agent.multi_agent_generation import MultiAgentGenerationManager
    from verl.workers.reward_manager.multi_agent import MultiAgentRewardManager

    self.generation_manager = MultiAgentGenerationManager(
        tokenizer=self.tokenizer,
        actor_rollout_wg=self.actor_rollout_wg,
        config=self.generation_config,
        lora_paths={
            "planner": self._get_lora_path("planner"),
            "executor": self._get_lora_path("executor"),
            "writer": self._get_lora_path("writer"),
        },
    )
    self.reward_manager = MultiAgentRewardManager(
        tokenizer=self.tokenizer,
        config=self.config.multi_agent,
    )
```

- [ ] **Step 2: Add multi-agent rollout + reward + update in training loop**

在 `fit()` 的主循环内（现有的 generation → reward → advantage → update 流程之后/替代），加入分支：

```python
if self.config.multi_agent.enable:
    # --- Rollout ---
    rollout_result = self.generation_manager.run_multi_agent_loop(
        gen_batch, global_steps
    )

    # --- Reward ---
    reward_data = DataProto()
    reward_data.non_tensor_batch = rollout_result['metadata']
    # Add response length info for reward tensor construction
    for agent in ['planner', 'executor', 'writer']:
        agent_output = rollout_result[agent]
        resp_len = agent_output.batch['responses'].shape[1]
        reward_data.non_tensor_batch[f'{agent}_response_length'] = resp_len
        valid_lens = (agent_output.batch['attention_mask'][:, -resp_len:].sum(dim=1)).tolist()
        reward_data.non_tensor_batch[f'{agent}_valid_lengths'] = valid_lens
    reward_data.non_tensor_batch['exec_max_turns'] = self.config.multi_agent.agents.executor.max_turns

    rewards = self.reward_manager(reward_data)

    # --- Advantage + Update (per agent) ---
    for agent_name in ['planner', 'executor', 'writer']:
        agent_output = rollout_result[agent_name]
        agent_reward = rewards[agent_name]

        # GRPO advantage (same function, different reward)
        index = agent_output.non_tensor_batch.get('agent_grpo_idx',
            torch.arange(agent_output.batch['responses'].shape[0]))
        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=agent_reward,
            eos_mask=agent_output.batch['attention_mask'][:, -agent_reward.shape[1]:],
            index=torch.tensor(index),
        )

        # Pack into DataProto for update
        agent_output.batch['advantages'] = advantages
        agent_output.batch['old_log_probs'] = agent_output.batch.get('log_probs',
            torch.zeros_like(advantages))

        # Update only this agent's LoRA
        self.actor_rollout_wg.update_actor(agent_output, lora_name=agent_name)

else:
    # --- Original single-agent logic (unchanged) ---
    # ... existing code ...
```

- [ ] **Step 3: Verify syntax**

Run: `python -c "from verl.trainer.ppo.ray_trainer import RayPPOTrainer; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add verl/trainer/ppo/ray_trainer.py
git commit -m "feat: add multi-agent training branch to ray_trainer.fit()"
```

---

## Task 11: 训练入口脚本

**Files:**
- Create: `train_multi_agent.sh`

- [ ] **Step 1: Create train_multi_agent.sh**

```bash
#!/bin/bash
# Multi-agent LoRA GRPO training entry script
# Usage: bash train_multi_agent.sh

set -euo pipefail

export PET_NODE_RANK=${PET_NODE_RANK:-0}
export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_ppo \
    data.train_files=./data/multi-research.parquet \
    data.val_files=./data/multi-research_dev.parquet \
    data.train_batch_size=256 \
    data.val_batch_size=1312 \
    data.max_prompt_length=4096 \
    data.max_response_length=8192 \
    actor_rollout_ref.model.path=GAIR/DeepResearcher-7b \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    algorithm.adv_estimator=grpo \
    trainer.logger=['console','wandb'] \
    trainer.project_name=multi_agent_research \
    trainer.experiment_name=multi_agent_lora_grpo \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.total_epochs=200 \
    agent_grpo.n=16 \
    multi_agent.enable=true \
    multi_agent.lora.rank=64 \
    multi_agent.lora.alpha=128 \
    multi_agent.reward.judge_model=claude-opus-4-7 \
    multi_agent.reward.judge_base_url=${JUDGE_BASE_URL} \
    multi_agent.reward.judge_api_key=${JUDGE_API_KEY} \
    multi_agent.reward.judge_max_concurrent=50 \
    multi_agent.reward.alpha=0.2 \
    multi_agent.reward.beta=0.3
```

- [ ] **Step 2: Make executable**

```bash
chmod +x train_multi_agent.sh
```

- [ ] **Step 3: Commit**

```bash
git add train_multi_agent.sh
git commit -m "feat: add multi-agent LoRA GRPO training entry script"
```

---

## Task 12: 训练数据准备

**Files:**
- Create: `data/multi-research.parquet`

用 Python 脚本生成样例训练数据。

- [ ] **Step 1: Create data generation script and sample data**

```python
# scripts/generate_research_queries.py
"""Generate sample research queries for multi-agent training."""
import pandas as pd

queries = [
    {"data_source": "research_query", "prompt": [{"role": "user", "content": "分析2024年全球AI芯片市场格局"}], "reward_model": {}, "extra_info": {"domain": "tech", "index": "rq_00001"}},
    {"data_source": "research_query", "prompt": [{"role": "user", "content": "比较中美新能源汽车产业链差异"}], "reward_model": {}, "extra_info": {"domain": "industry", "index": "rq_00002"}},
    {"data_source": "research_query", "prompt": [{"role": "user", "content": "总结最近一周加密货币市场走势及影响因素"}], "reward_model": {}, "extra_info": {"domain": "finance", "index": "rq_00003"}},
    {"data_source": "research_query", "prompt": [{"role": "user", "content": "分析大语言模型在医疗诊断中的应用现状和挑战"}], "reward_model": {}, "extra_info": {"domain": "healthcare", "index": "rq_00004"}},
    {"data_source": "research_query", "prompt": [{"role": "user", "content": "评估2024年全球半导体供应链风险"}], "reward_model": {}, "extra_info": {"domain": "tech", "index": "rq_00005"}},
    # ... more queries to be added via LLM batch generation
]

df = pd.DataFrame(queries)
df.to_parquet("data/multi-research.parquet", index=False)
print(f"Generated {len(df)} queries → data/multi-research.parquet")

# Dev set (subset)
df_dev = df.head(2)
df_dev.to_parquet("data/multi-research_dev.parquet", index=False)
print(f"Generated {len(df_dev)} queries → data/multi-research_dev.parquet")
```

Run: `python scripts/generate_research_queries.py`

后续用 LLM 批量生成更多研究问题扩充数据集。

- [ ] **Step 2: Verify data loads with RLHFDataset**

```bash
python -c "
from verl.utils.dataset.rl_dataset import RLHFDataset
ds = RLHFDataset(parquet_files='data/multi-research.parquet', tokenizer=None)
print(f'Loaded {len(ds)} items')
print(ds[0])
"
```

Expected: Loads without error, shows first item with `prompt` field.

- [ ] **Step 3: Commit**

```bash
git add scripts/generate_research_queries.py data/multi-research.parquet data/multi-research_dev.parquet
git commit -m "feat: add sample research query training data"
```

---

## Task 13: Checkpoint 保存/加载

**Files:**
- Modify: `verl/trainer/ppo/ray_trainer.py` (checkpoint section in `fit()`)

训练时只保存三个 LoRA adapter，不保存基座。

- [ ] **Step 1: Add multi-agent checkpoint save logic**

在 `fit()` 的 checkpoint 保存部分（现有的 `save_checkpoint` 逻辑附近），加分支：

```python
if self.config.multi_agent.enable:
    ckpt_dir = f"ckpts/{self.config.trainer.experiment_name}/global_step_{global_steps}"
    for adapter_name in ["planner", "executor", "writer"]:
        adapter_dir = os.path.join(ckpt_dir, f"lora_{adapter_name}")
        os.makedirs(adapter_dir, exist_ok=True)
        self.actor_module.set_adapter(adapter_name)
        self.actor_module.save_pretrained(adapter_dir)
    print(f"Saved LoRA checkpoints to {ckpt_dir}")
```

- [ ] **Step 2: Add multi-agent checkpoint load logic**

在 `fit()` 开头或 `_build_model_optimizer()` 中，如果存在 checkpoint 则加载：

```python
if self.config.multi_agent.enable and resume_from:
    from peft import PeftModel
    for adapter_name in ["planner", "executor", "writer"]:
        adapter_path = os.path.join(resume_from, f"lora_{adapter_name}")
        if os.path.exists(adapter_path):
            actor_module.load_adapter(adapter_path, adapter_name=adapter_name)
            print(f"Loaded LoRA adapter: {adapter_name} from {adapter_path}")
```

- [ ] **Step 3: Commit**

```bash
git add verl/trainer/ppo/ray_trainer.py
git commit -m "feat: add LoRA checkpoint save/load for multi-agent training"
```

---

## Task 14: 端到端集成验证

- [ ] **Step 1: Verify single-agent training is unaffected**

```bash
python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('verl/trainer/config/ppo_trainer.yaml')
assert cfg.multi_agent.enable == False
print('multi_agent.enable is False by default — single agent path safe')
"
```

- [ ] **Step 2: Verify all new modules import correctly**

```bash
python -c "
from verl.utils.reward_score.rule_reward import planner_rules, executor_rules
from verl.utils.reward_score.llm_judge import LLMJudge
from verl.workers.reward_manager.multi_agent import MultiAgentRewardManager
from scrl.llm_agent.multi_agent_generation import MultiAgentGenerationManager
print('All imports OK')
"
```

- [ ] **Step 3: Run unit tests**

```bash
pytest tests/test_rule_reward.py tests/test_llm_judge.py -v
```

Expected: All tests pass.

- [ ] **Step 4: Dry-run config parsing**

```bash
python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('verl/trainer/config/ppo_trainer.yaml')
# Simulate multi-agent enable
cfg.multi_agent.enable = True
print(f'LoRA rank: {cfg.multi_agent.lora.rank}')
print(f'Judge model: {cfg.multi_agent.reward.judge_model}')
print(f'Alpha: {cfg.multi_agent.reward.alpha}')
print(f'Planner max_tokens: {cfg.multi_agent.agents.planner.max_tokens}')
print('Config validation OK')
"
```

- [ ] **Step 5: Final commit — update design doc status**

```bash
git add -A
git commit -m "feat: complete multi-agent LoRA GRPO training implementation"
```

---

## Execution Order & Dependencies

```
Task 1 (rule_reward) ──┐
Task 2 (llm_judge)  ───┤
Task 3 (config)     ───┼──→ Task 4 (reward_manager) ──┐
                       │                               │
Task 5 (vllm_rollout) ─┤                               │
Task 6 (generation.py) ┼──→ Task 7 (multi_agent_gen) ──┼──→ Task 10 (ray_trainer)
                       │                               │         │
Task 8 (fsdp_workers) ─┤                               │    Task 13 (checkpoint)
Task 9 (dp_actor)   ───┘                               │         │
                                                       │    Task 11 (script)
Task 12 (data)      ───────────────────────────────────┘         │
                                                            Task 14 (验证)
```

Tasks 1, 2, 3, 5, 6, 8, 9, 12 可以并行开发。
Task 4 依赖 1+2+3。
Task 7 依赖 5+6。
Task 10 依赖 4+7+8+9。
Task 11, 13 依赖 10。
Task 14 最后。
