"""Unit tests for DeepResearch-9K data preparation."""

import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.prepare_deepresearch_data import convert_row, build_split


class TestConvertRow:
    def test_basic_conversion(self):
        row = {
            "question": "Who directed Move (1970)?",
            "difficulty": 1,
            "final answer": "Stuart Rosenberg",
            "search trajectory": [],
        }
        result = convert_row(row, 42)

        assert result["data_source"] == "deepresearch"
        assert result["ability"] == "multi_hop_qa"
        assert result["prompt"] == [{"role": "user", "content": "Who directed Move (1970)?"}]
        assert result["reward_model"]["ground_truth"] == "Stuart Rosenberg"
        assert result["reward_model"]["style"] == "factoid"
        assert result["extra_info"]["difficulty"] == 1
        assert result["extra_info"]["index"] == "deepresearch_42"

    def test_prompt_is_chat_format(self):
        row = {
            "question": "Test question?",
            "difficulty": 2,
            "final answer": "answer",
            "search trajectory": [],
        }
        result = convert_row(row, 0)
        prompt = result["prompt"]

        assert isinstance(prompt, list)
        assert len(prompt) == 1
        assert prompt[0]["role"] == "user"
        assert prompt[0]["content"] == "Test question?"

    def test_preserves_difficulty_in_extra_info(self):
        for level in [1, 2, 3]:
            row = {
                "question": "Q",
                "difficulty": level,
                "final answer": "A",
                "search trajectory": [],
            }
            result = convert_row(row, 0)
            assert result["extra_info"]["difficulty"] == level


class TestBuildSplit:
    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset with known distribution."""

        class MockRow:
            def __init__(self, q, d, a):
                self._data = {
                    "question": q,
                    "difficulty": d,
                    "final answer": a,
                    "search trajectory": [],
                }

            def __getitem__(self, key):
                return self._data[key]

        rows = []
        for i in range(30):
            rows.append(MockRow(f"L1 question {i}", 1, f"answer_{i}"))
        for i in range(40):
            rows.append(MockRow(f"L2 question {i}", 2, f"answer_{i}"))
        for i in range(50):
            rows.append(MockRow(f"L3 question {i}", 3, f"answer_{i}"))
        return rows

    def test_filters_by_difficulty(self, mock_dataset):
        train, val = build_split(mock_dataset, [1, 2], split_ratio=0.9, seed=42)
        total = len(train) + len(val)
        assert total == 70  # 30 L1 + 40 L2

    def test_excludes_other_levels(self, mock_dataset):
        train, val = build_split(mock_dataset, [2, 3], split_ratio=0.9, seed=42)
        total = len(train) + len(val)
        assert total == 90  # 40 L2 + 50 L3

    def test_split_ratio(self, mock_dataset):
        train, val = build_split(mock_dataset, [1, 2, 3], split_ratio=0.8, seed=42)
        total = len(train) + len(val)
        assert total == 120
        assert abs(len(train) / total - 0.8) < 0.05

    def test_output_schema(self, mock_dataset):
        train, _ = build_split(mock_dataset, [1], split_ratio=0.9, seed=42)
        assert list(train.columns) == ["data_source", "prompt", "ability", "reward_model", "extra_info"]

    def test_deterministic_with_seed(self, mock_dataset):
        train1, _ = build_split(mock_dataset, [1, 2], split_ratio=0.9, seed=42)
        train2, _ = build_split(mock_dataset, [1, 2], split_ratio=0.9, seed=42)
        pd.testing.assert_frame_equal(train1, train2)


class TestParquetIntegration:
    """Integration test: verify generated parquet files have correct schema."""

    @pytest.fixture
    def temp_parquet(self, tmp_path, mock_dataset):
        """Generate a temp parquet file for testing."""
        from scripts.prepare_deepresearch_data import convert_row

        rows = []
        for i in range(50):
            rows.append({
                "question": f"Question {i}",
                "difficulty": (i % 3) + 1,
                "final answer": f"Answer {i}",
                "search trajectory": [],
            })

        records = [convert_row(row, i) for i, row in enumerate(rows)]
        df = pd.DataFrame(records)
        path = tmp_path / "test.parquet"
        df.to_parquet(path)
        return path

    @pytest.fixture
    def mock_dataset(self):
        return None  # not needed for this fixture

    def test_parquet_roundtrip(self, tmp_path):
        """Verify parquet read/write preserves schema."""
        rows = []
        for i in range(20):
            rows.append({
                "question": f"Question {i}",
                "difficulty": (i % 3) + 1,
                "final answer": f"Answer {i}",
                "search trajectory": [],
            })

        records = [convert_row(row, i) for i, row in enumerate(rows)]
        df = pd.DataFrame(records)
        path = tmp_path / "roundtrip.parquet"
        df.to_parquet(path)

        loaded = pd.read_parquet(path)
        assert list(loaded.columns) == ["data_source", "prompt", "ability", "reward_model", "extra_info"]
        assert len(loaded) == 20

        first = loaded.iloc[0]
        assert first["data_source"] == "deepresearch"
        prompt = list(first["prompt"]) if not isinstance(first["prompt"], list) else first["prompt"]
        assert prompt[0]["role"] == "user"
        assert isinstance(first["reward_model"], dict)
        assert "ground_truth" in first["reward_model"]
        assert isinstance(first["extra_info"], dict)
        assert "difficulty" in first["extra_info"]

    def test_schema_matches_existing_train(self, tmp_path):
        """Verify our schema matches the existing train.parquet."""
        existing = pd.read_parquet("data/train.parquet")
        existing_cols = set(existing.columns)

        row = {
            "question": "Test",
            "difficulty": 1,
            "final answer": "Answer",
            "search trajectory": [],
        }
        record = convert_row(row, 0)
        new_cols = set(record.keys())

        assert new_cols == existing_cols

        existing_rm_keys = set(existing.iloc[0]["reward_model"].keys())
        new_rm_keys = set(record["reward_model"].keys())
        assert existing_rm_keys == new_rm_keys
