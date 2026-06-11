"""force-answer 截断 (_truncate_to_first_answer)。

thinking 模型被预填进 `<answer>` 后,常在吐完首个 `</answer>` 又自发续写第二个
`</think>`/`<answer>`(双标签 → check_tags_balance 失败 → 末轮 f1 被误罚 -2.0,
且畸形 token 经 _assemble_planner_tensors re-tokenize 进入 planner 训练 response)。
修复:截断到第一个 `</answer>`(含);无则补一个。

importlib 直接 load interleaved_generation.py(顶部仅 dataclass/typing,重依赖均在
函数内 lazy import)→ 无需 torch/verl,CPU/.venv 即可跑。
"""
import importlib.util
import os

_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "scrl", "llm_agent", "interleaved_generation.py"
)
_spec = importlib.util.spec_from_file_location("interleaved_standalone", _PATH)
ig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ig)

trunc = ig._truncate_to_first_answer


def test_double_answer_truncated_to_first():
    # 实测 idx0 畸形:首个 </answer> 后又 </think> + 第二个 <answer>
    s = ("\n</think>\n<answer>Newcastle Lifeboat Station</answer>"
         "\n</think>\n\n<answer>We found that...</answer>")
    out = trunc(s)
    assert out == "\n</think>\n<answer>Newcastle Lifeboat Station</answer>"
    # 标签平衡:各恰好一个
    assert out.count("<answer>") == 1 and out.count("</answer>") == 1
    assert out.count("</think>") == 1


def test_clean_single_answer_unchanged():
    s = "\n</think>\n<answer>Paris</answer>"
    assert trunc(s) == s


def test_no_closing_answer_appended():
    # 被 max_tokens 截断、未闭合 → 补一个 </answer>(保持原 fallback 行为)
    s = "\n</think>\n<answer>Paris"
    assert trunc(s) == "\n</think>\n<answer>Paris</answer>"


def test_trailing_garbage_after_answer_dropped():
    # 首个 </answer> 后的 token salad 应被丢弃
    s = "\n</think>\n<answer>42</answer> portletAm prosecutors Directions"
    assert trunc(s) == "\n</think>\n<answer>42</answer>"


def test_empty_answer_kept_minimal():
    # 空 answer 也只保留到首个 </answer>
    s = "\n</think>\n<answer></answer>\n</think>\n<answer>x</answer>"
    assert trunc(s) == "\n</think>\n<answer></answer>"
