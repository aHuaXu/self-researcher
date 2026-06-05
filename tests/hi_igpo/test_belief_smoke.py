"""
Belief (ground-truth log-prob) smoke test — Task 3.

belief 计算依赖真实 HF 模型(Qwen3-4B)+ GPU,本地 .venv-test(CPU、无 transformers)
跑不了,故本测试在缺 CUDA / transformers / 模型路径时自动 skip,在**服务器** GPU 环境手验。

服务器验证步骤(对应计划 Task 3 Step 3):
  1. 设环境变量指向本地模型:export IGPO_BELIEF_TEST_MODEL=/path/to/Qwen3-4B-Instruct
  2. conda activate deepresearcher && python -m pytest tests/hi_igpo/test_belief_smoke.py -v
  3. 断言:每个 Bel_t ∈ (0,1);含 golden 答案的上下文 belief > 无关上下文 belief(方向单调)。

验证点(= IGPO belief 定义):
  Bel_t = exp( mean_j log π(a_j | context_t, a_<j) )   # golden 答案的 token 几何平均概率
  全程 torch.no_grad()(stop-gradient)。
"""
import os
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

_MODEL = os.getenv("IGPO_BELIEF_TEST_MODEL")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="belief smoke needs GPU"),
    pytest.mark.skipif(_MODEL is None, reason="set IGPO_BELIEF_TEST_MODEL to a local HF model path"),
]


def _load():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        _MODEL, torch_dtype=torch.float16, trust_remote_code=True
    ).cuda().eval()
    return tok, model


def test_belief_in_unit_interval_and_increases_with_answer_in_context():
    """含答案的上下文应给出更高 belief;所有 belief 落在 (0,1)。"""
    from scrl.llm_agent.vectorized_gt_logprob import (
        VectorizedGTConfig,
        VectorizedGTLogProbComputer,
    )

    tok, model = _load()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    config = VectorizedGTConfig(pad_token_id=pad_id, eos_token_id=tok.eos_token_id)
    computer = VectorizedGTLogProbComputer(tok, config)

    question = "What is the capital of France?"
    golden = "Paris"

    # 本项目 THINKING OFF:reasoning 是 <answer> 前的纯文本,无 <think> 标签。
    # belief 的 PREFIX 已配置为 no-think("...\n<answer>\n"),所以上下文用纯文本即可,
    # 拼上 PREFIX 后是良构的 <reasoning>...<answer>golden</answer>。
    head = f"Question: {question}\n"
    ctx_with = head + "Findings: The capital of France is Paris."
    ctx_without = head + "Findings: Bananas are rich in potassium."

    beliefs = {}
    for name, ctx in [("with", ctx_with), ("without", ctx_without)]:
        ids = tok(ctx, return_tensors="pt", add_special_tokens=False).input_ids[0].cuda()
        attn = torch.ones_like(ids)
        pos = torch.arange(ids.shape[0], device=ids.device)
        gt_start, gt_end = computer.get_gt_answer_token_range(golden)
        # 单 turn:turn_end_positions = [序列末]
        with torch.no_grad():
            gt_log_probs, gt_ranges = computer.compute_all_turns_vectorized(
                model, ids, attn, pos, golden, turn_end_positions=[ids.shape[0] - 1]
            )
        assert len(gt_log_probs) >= 1, f"no per-turn log probs returned for {name}"
        # gt_log_probs[0] 是整段 GT(PREFIX+答案+SUFFIX)的逐 token logp;
        # belief = P(答案 token),必须用 gt_ranges[0] 切到答案区间再取均值。
        lp_full = torch.as_tensor(gt_log_probs[0], dtype=torch.float32)
        a_start, a_end = gt_ranges[0]
        lp = lp_full[a_start:a_end]
        bel = torch.exp(lp.mean()).item()
        print(f"[belief] {name:8s} bel={bel:.6f}  ctx_tokens={ids.shape[0]} "
              f"answer_range={(a_start, a_end)} full_logp={lp_full.tolist()} answer_logp={lp.tolist()}")
        assert 0.0 < bel < 1.0, f"belief out of (0,1): {name}={bel}"
        beliefs[name] = bel

    assert beliefs["with"] > beliefs["without"], (
        f"belief should be higher with answer in context: {beliefs}"
    )
