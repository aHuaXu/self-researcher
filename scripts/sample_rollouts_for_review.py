#!/usr/bin/env python3
"""Sample built Planner-SFT rows and emit a flattened, reviewer-friendly JSONL.

Reads the train.parquet produced by build_planner_sft_from_rollouts.py (which has
the clean `messages`, `question`, `gt`, `answer`, `f1`, `em`, `num_subtasks`,
`difficulty`, `source_index`), randomly samples N rows, and writes a JSONL where
each line is one trajectory rendered as a readable sequence of turns so that a
review subagent (or human) can eyeball quality without parsing parquet.

Output fields per line:
  - id, question, gt, predicted_answer, f1, em, num_subtasks, difficulty
  - trajectory: [{role, content}, ...]   (the clean SFT messages)
  - subtasks: [str, ...]                  (extracted for quick scan)
  - findings: [str, ...]                  (extracted for quick scan)

Usage:
    python scripts/sample_rollouts_for_review.py \\
        --input data/planner_sft/train.parquet \\
        --output tmp/planner_sft_review_sample.jsonl \\
        --n 100 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


def _series_to_item(value):
    while hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    return value


def extract_role(messages: list[dict], role: str) -> list[str]:
    return [str(m.get("content", "")).strip() for m in messages if m.get("role") == role]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Built train/val parquet with messages column")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--n", type=int, default=100, help="Number of rows to sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    n = min(args.n, len(df))
    if n == 0:
        raise ValueError(f"input has 0 rows: {args.input}")

    rng = random.Random(args.seed)
    indices = rng.sample(range(len(df)), n)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for idx in indices:
            row = df.iloc[idx]
            messages = _series_to_item(row.get("messages"))
            subtasks = [m for m in extract_role(messages, "assistant") if "<subtask>" in m]
            findings = extract_role(messages, "user")[1:]  # drop the original question (first user)
            record = {
                "id": str(row.get("source_index", idx)),
                "question": str(row.get("question", "")),
                "gt": str(row.get("gt", "")),
                "predicted_answer": str(row.get("answer", "")),
                "f1": float(row.get("f1", 0.0)),
                "em": bool(row.get("em", False)),
                "num_subtasks": int(row.get("num_subtasks", 0)),
                "difficulty": int(row.get("difficulty", -1)),
                "subtasks": subtasks,
                "findings": findings,
                "trajectory": messages,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {n} sampled rows to {out}")


if __name__ == "__main__":
    main()
