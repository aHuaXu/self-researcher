#!/usr/bin/env python3
"""Sample DeepResearch questions for Planner SFT trajectory generation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


def _first_user_question(prompt: Any) -> str:
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if not isinstance(prompt, list):
        return ""
    for message in prompt:
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content", "")).strip()
    return ""


def _field_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    return {}


def normalize_source_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        extra_info = _field_dict(row.get("extra_info"))
        reward_model = _field_dict(row.get("reward_model"))
        question = _first_user_question(row.get("prompt"))
        gt = str(reward_model.get("ground_truth", "")).strip()
        difficulty = extra_info.get("difficulty")
        if not question or not gt or difficulty not in {2, 3}:
            continue
        rows.append(
            {
                "question": question,
                "gt": gt,
                "difficulty": int(difficulty),
                "source_index": str(extra_info.get("index", "")),
                "data_source": row.get("data_source", "deepresearch"),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/deepresearch_phase2.parquet")
    parser.add_argument("--output", default="data/planner_sft_seed_l3_500.parquet")
    parser.add_argument("--l2-count", type=int, default=0)
    parser.add_argument("--l3-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = normalize_source_rows(pd.read_parquet(args.input))
    pieces = []
    for difficulty, count in [(2, args.l2_count), (3, args.l3_count)]:
        level_df = df[df["difficulty"] == difficulty]
        if len(level_df) < count:
            raise ValueError(f"Need {count} L{difficulty} samples, got {len(level_df)}")
        pieces.append(level_df.sample(n=count, random_state=args.seed).reset_index(drop=True))

    out = pd.concat(pieces).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output)

    print(f"Wrote {len(out)} samples to {output}")
    print(out["difficulty"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
