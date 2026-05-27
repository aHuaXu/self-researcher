"""
Convert DeepResearch-9K dataset from HuggingFace to training parquet files.

Produces two files:
  - data/deepresearch_phase1.parquet: L1+L2 for single-agent warmup
  - data/deepresearch_phase2.parquet: L2+L3 for dual-agent GRPO training

Schema matches existing train.parquet format:
  {data_source, prompt, ability, reward_model, extra_info}
"""

import argparse
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def convert_row(row, idx):
    """Convert a single DeepResearch-9K row to training format."""
    return {
        "data_source": "deepresearch",
        "prompt": [{"role": "user", "content": row["question"]}],
        "ability": "multi_hop_qa",
        "reward_model": {
            "ground_truth": row["final answer"],
            "style": "factoid",
        },
        "extra_info": {
            "difficulty": row["difficulty"],
            "index": f"deepresearch_{idx}",
        },
    }


def build_split(ds, difficulty_levels, split_ratio=0.9, seed=42):
    """Filter by difficulty levels and split into train/val."""
    filtered = [row for row in ds if row["difficulty"] in difficulty_levels]

    records = [convert_row(row, i) for i, row in enumerate(filtered)]
    df = pd.DataFrame(records)

    train_df, val_df = train_test_split(
        df, train_size=split_ratio, random_state=seed,
        stratify=df["extra_info"].apply(lambda x: x["difficulty"]),
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare DeepResearch-9K data")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--split-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading DeepResearch-9K from HuggingFace...")
    ds = load_dataset("artillerywu/DeepResearch-9K", split="train")

    print(f"Total samples: {len(ds)}")
    for lvl in [1, 2, 3]:
        count = sum(1 for row in ds if row["difficulty"] == lvl)
        print(f"  Level {lvl}: {count}")

    # Phase 1: L1 + L2 (single-agent warmup)
    print("\nBuilding Phase 1 (L1+L2)...")
    train1, val1 = build_split(ds, [1, 2], args.split_ratio, args.seed)
    train1.to_parquet(f"{args.output_dir}/deepresearch_phase1.parquet")
    val1.to_parquet(f"{args.output_dir}/deepresearch_phase1_val.parquet")
    print(f"  Train: {len(train1)}, Val: {len(val1)}")

    # Phase 2: L2 + L3 (dual-agent GRPO)
    print("\nBuilding Phase 2 (L2+L3)...")
    train2, val2 = build_split(ds, [2, 3], args.split_ratio, args.seed)
    train2.to_parquet(f"{args.output_dir}/deepresearch_phase2.parquet")
    val2.to_parquet(f"{args.output_dir}/deepresearch_phase2_val.parquet")
    print(f"  Train: {len(train2)}, Val: {len(val2)}")

    print("\nDone!")


if __name__ == "__main__":
    main()
