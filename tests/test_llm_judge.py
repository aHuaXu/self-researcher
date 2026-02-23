"""Tests for the async LLM Judge module."""

import asyncio
import importlib.util
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# Load llm_judge.py directly to avoid importing the full verl package,
# which requires torch, tensordict, and other heavy dependencies.
_spec = importlib.util.spec_from_file_location(
    "llm_judge",
    "/Users/jiahua.xu/dl_learn/self-researcher/verl/utils/reward_score/llm_judge.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

parse_score = _mod.parse_score
LLMJudge = _mod.LLMJudge


# --- parse_score tests ---


class TestParseScore:
    def test_plain_number(self):
        assert parse_score("7") == 7.0

    def test_with_text(self):
        assert parse_score("Score: 8/10") == 8.0

    def test_decimal(self):
        assert parse_score("7.5") == 7.5

    def test_no_number(self):
        assert parse_score("no score") == 0.0

    def test_above_10(self):
        assert parse_score("15") == 10.0

    def test_zero(self):
        assert parse_score("0") == 0.0


# --- LLMJudge tests ---


class TestLLMJudge:
    def _make_judge(self):
        judge = LLMJudge(
            model="test-model",
            base_url="http://localhost:8000/v1",
            api_key="test-key",
        )
        return judge

    def test_score_batch_returns_normalized_scores(self):
        judge = self._make_judge()

        # Mock the API response
        mock_choice = MagicMock()
        mock_choice.message.content = "8"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        judge.client.chat.completions.create = AsyncMock(return_value=mock_response)

        queries = ["What is Python?", "What is Rust?"]
        reports = ["Python is a language.", "Rust is a language."]

        scores = asyncio.run(judge.score_batch(queries, reports))

        assert len(scores) == 2
        assert scores[0] == pytest.approx(0.8)
        assert scores[1] == pytest.approx(0.8)

    def test_api_error_returns_zero(self):
        judge = self._make_judge()

        judge.client.chat.completions.create = AsyncMock(
            side_effect=Exception("API connection error")
        )

        queries = ["What is Python?"]
        reports = ["Python is a language."]

        scores = asyncio.run(judge.score_batch(queries, reports))

        assert len(scores) == 1
        assert scores[0] == 0.0

    def test_score_batch_mixed_success_and_failure(self):
        judge = self._make_judge()

        mock_choice = MagicMock()
        mock_choice.message.content = "9"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        # First call succeeds, second raises
        judge.client.chat.completions.create = AsyncMock(
            side_effect=[mock_response, Exception("timeout")]
        )

        queries = ["Q1", "Q2"]
        reports = ["R1", "R2"]

        scores = asyncio.run(judge.score_batch(queries, reports))

        assert len(scores) == 2
        assert scores[0] == pytest.approx(0.9)
        assert scores[1] == 0.0
