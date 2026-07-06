#!/usr/bin/env python3
"""Build clean Planner-SFT messages from interleaved rollout dumps.

Input rollout files are the JSON dumps produced by Hi-IGPO Phase 2b, e.g.
``outputs/.../rollout/planner_rollout_step_*.json``. The script keeps only
clean Planner turns and masks Executor findings as user messages in the final
multi-turn SFT dataset.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import string
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_agent.config import get_config
from research_agent.prompts.planner import get_interleaved_planner_prompt


BANNED_PLANNER_RE = re.compile(
    r"<tool_call>|<tool_response>|</?think>|search for|open result|browse_webpage|web_search",
    re.IGNORECASE,
)
BANNED_FINDING_RE = re.compile(
    r"<tool_call>|<tool_response>|</?think>|</?answer>|search for|open result|"
    r"web search results|search results|browse.*fail|failed to|the user is asking|"
    r"\bi (found|searched|analy[sz]ed)\b",
    re.IGNORECASE,
)


def preprocess_text(text: str) -> str:
    for punct in string.punctuation:
        text = text.replace(punct, " ")
    return re.sub(r"\s+", " ", text.lower()).strip()


def token_f1(prediction: str, ground_truth: str) -> float:
    if not prediction or not ground_truth:
        return 0.0
    pred_tokens = set(preprocess_text(prediction).split())
    if not pred_tokens:
        return 0.0
    best = 0.0
    for gt in ground_truth.split("<|answer_split|>"):
        gt_tokens = set(preprocess_text(gt).split())
        if not gt_tokens:
            continue
        common = pred_tokens & gt_tokens
        if not common:
            continue
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gt_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def exact_match(prediction: str, ground_truth: str) -> bool:
    pred = preprocess_text(prediction)
    return any(pred == preprocess_text(gt) for gt in ground_truth.split("<|answer_split|>"))


def llm_judge_correct(client: OpenAI, model: str, question: str, reference: str, prediction: str) -> bool:
    prompt = f"""Decide whether the predicted answer correctly answers the question, using the reference answer.

Question:
{question}

Reference answer:
{reference}

Predicted answer:
{prediction}

Return only JSON: {{"correct": true}} or {{"correct": false}}."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a strict but semantic answer judge."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            temperature=0.0,
        )
        text = response.choices[0].message.content or ""
    except Exception:
        return False

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return bool(json.loads(match.group(0)).get("correct", False))
        except Exception:
            pass
    return "true" in text.lower() and "false" not in text.lower()


def load_source_map(source_data: str | None) -> dict[str, dict[str, Any]]:
    if not source_data:
        return {}
    df = pd.read_parquet(source_data)
    mapping = {}
    for _, row in df.iterrows():
        if "question" in row and "gt" in row:
            question = str(row["question"]).strip()
            gt = str(row["gt"]).strip()
            difficulty = int(row.get("difficulty", -1))
            source_index = str(row.get("source_index", ""))
        else:
            prompt = row.get("prompt")
            if hasattr(prompt, "tolist"):
                prompt = prompt.tolist()
            question = ""
            if isinstance(prompt, list):
                for message in prompt:
                    if isinstance(message, dict) and message.get("role") == "user":
                        question = str(message.get("content", "")).strip()
                        break
            reward_model = row.get("reward_model") if isinstance(row.get("reward_model"), dict) else {}
            extra_info = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
            gt = str(reward_model.get("ground_truth", "")).strip()
            difficulty = int(extra_info.get("difficulty", -1))
            source_index = str(extra_info.get("index", ""))
        if question and gt:
            mapping[question] = {"gt": gt, "difficulty": difficulty, "source_index": source_index}
    return mapping


def extract_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text or "", re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def clean_subtask(turn: dict[str, Any]) -> str:
    payload = str(turn.get("payload") or "").strip()
    raw = str(turn.get("planner_output_raw") or "")
    subtask = payload or extract_tag(raw, "subtask")
    subtask = re.sub(r"\s+", " ", subtask).strip()
    if not subtask or BANNED_PLANNER_RE.search(subtask):
        return ""
    if re.search(r"^\s*\d+\.", subtask) or "\n1." in subtask:
        return ""
    return subtask


def clean_answer(row: dict[str, Any], turn: dict[str, Any]) -> str:
    payload = str(turn.get("payload") or row.get("answer") or "").strip()
    raw = str(turn.get("planner_output_raw") or "")
    answer = payload or extract_tag(raw, "answer")
    answer = re.sub(r"\s+", " ", answer).strip()
    if not answer or BANNED_PLANNER_RE.search(answer):
        return ""
    if len(answer.split()) > 24:
        return ""
    return answer


def clean_finding(text: Any, max_chars: int) -> str:
    finding = re.sub(r"\s+", " ", str(text or "")).strip()
    if not finding or BANNED_FINDING_RE.search(finding):
        return ""
    if len(finding) < 8:
        return ""
    if len(finding) > max_chars:
        finding = finding[:max_chars].rstrip() + " ...[truncated]"
    return finding


def convert_rollout_row(
    row: dict[str, Any],
    source_map: dict[str, dict[str, Any]],
    *,
    min_f1: float,
    require_em: bool,
    max_finding_chars: int,
    min_subtasks: int,
    judge_client: OpenAI | None = None,
    judge_model: str = "",
) -> tuple[dict[str, Any] | None, str]:
    question = str(row.get("question", "")).strip()
    source = source_map.get(question, {})
    gt = str(row.get("gt") or source.get("gt") or "").strip()
    if not question or not gt:
        return None, "missing_question_or_gt"

    messages = list(get_interleaved_planner_prompt(question))
    final_answer = ""
    subtask_count = 0

    for turn in row.get("turns", []):
        kind = str(turn.get("parsed_kind") or "").strip().lower()
        if kind == "answer" or ("<answer>" in str(turn.get("planner_output_raw", "")).lower()):
            final_answer = clean_answer(row, turn)
            if not final_answer:
                return None, "bad_answer"
            messages.append({"role": "assistant", "content": f"<answer>{final_answer}</answer>"})
            break

        subtask = clean_subtask(turn)
        if not subtask:
            return None, "bad_subtask"
        finding = clean_finding(turn.get("executor_finding"), max_finding_chars)
        if not finding:
            return None, "bad_finding"
        messages.append({"role": "assistant", "content": f"<subtask>{subtask}</subtask>"})
        messages.append({"role": "user", "content": finding})
        subtask_count += 1

    if not final_answer:
        return None, "missing_final_answer"
    if subtask_count < min_subtasks:
        return None, "too_few_subtasks"

    f1 = token_f1(final_answer, gt)
    em = exact_match(final_answer, gt)
    judge_correct = (
        llm_judge_correct(judge_client, judge_model, question, gt, final_answer)
        if judge_client is not None
        else False
    )
    if require_em and not em:
        return None, "answer_em_filter"
    if not judge_correct and f1 < min_f1:
        return None, "answer_f1_filter"

    return (
        {
            "messages": messages,
            "question": question,
            "gt": gt,
            "answer": final_answer,
            "f1": f1,
            "em": em,
            "judge_correct": judge_correct,
            "num_subtasks": subtask_count,
            "difficulty": int(source.get("difficulty", row.get("difficulty", -1))),
            "source_index": str(source.get("source_index", row.get("source_index", ""))),
        },
        "kept",
    )


def iter_rollout_rows(paths: list[str]):
    for pattern in paths:
        expanded = sorted(glob.glob(pattern)) if any(ch in pattern for ch in "*?[]") else [pattern]
        for path_str in expanded:
            path = Path(path_str)
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"{path} should contain a list of rollout rows")
            for row in data:
                yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", nargs="+", required=True, help="Rollout JSON file(s), glob supported")
    parser.add_argument("--source-data", default="data/planner_sft_seed_l3_500.parquet")
    parser.add_argument("--output-train", default="data/planner_sft/train.parquet")
    parser.add_argument("--output-val", default="data/planner_sft/val.parquet")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-f1", type=float, default=0.8)
    parser.add_argument("--require-em", action="store_true")
    parser.add_argument("--max-finding-chars", type=int, default=1000)
    parser.add_argument("--min-subtasks", type=int, default=0)
    parser.add_argument("--best-of-n", action="store_true",
                        help="When --n-samples>1 at generate time, keep only the highest-F1 passing "
                             "sample per (source_index, question). Off => keep all passing samples.")
    parser.add_argument("--llm-judge", action="store_true", help="Use MiniMax/OpenAI-compatible semantic judge")
    parser.add_argument("--judge-model", default=os.getenv("LLM_MODEL", "MiniMax-M2.7"))
    args = parser.parse_args()

    source_map = load_source_map(args.source_data)
    judge_client = None
    if args.llm_judge:
        config = get_config()
        judge_client = OpenAI(
            base_url=config.llm.base_url,
            api_key=config.llm.api_key,
            timeout=config.llm.timeout,
            max_retries=3,
        )

    kept = []
    reasons = Counter()
    for row in iter_rollout_rows(args.rollout):
        converted, reason = convert_rollout_row(
            row,
            source_map,
            min_f1=args.min_f1,
            require_em=args.require_em,
            max_finding_chars=args.max_finding_chars,
            min_subtasks=args.min_subtasks,
            judge_client=judge_client,
            judge_model=args.judge_model,
        )
        reasons[reason] += 1
        if converted is not None:
            kept.append(converted)

    if args.best_of_n:
        # Group passing samples by (source_index, question); keep highest-F1 per group.
        # Falls back to all passing samples when n_samples==1 (no duplicates per group).
        best_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        dropped_dup = 0
        for row in kept:
            key = (str(row.get("source_index", "")), str(row.get("question", "")))
            cur = best_by_key.get(key)
            if cur is None or row["f1"] > cur["f1"]:
                if cur is not None:
                    dropped_dup += 1
                best_by_key[key] = row
        kept = list(best_by_key.values())
        if dropped_dup:
            print(f"best-of-n: dropped {dropped_dup} lower-F1 duplicate samples")

    if not kept:
        raise ValueError(f"No rows kept. Rejection reasons: {dict(reasons)}")

    random.Random(args.seed).shuffle(kept)
    val_n = max(1, int(len(kept) * args.val_ratio)) if len(kept) > 1 else 0
    val_rows = kept[:val_n]
    train_rows = kept[val_n:]

    for output, rows in [(args.output_train, train_rows), (args.output_val, val_rows)]:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path)
        print(f"Wrote {len(rows)} rows to {path}")

    print("Filter stats:", dict(reasons))
    print("Difficulty:", dict(Counter(row["difficulty"] for row in kept)))


if __name__ == "__main__":
    main()
