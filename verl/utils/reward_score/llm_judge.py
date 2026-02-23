"""Async LLM Judge for scoring research reports via external API calls."""

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
    """Extract first numeric value from text and clamp to [0, 10].

    Returns 0.0 if no number is found.
    """
    match = re.search(r"\d+\.?\d*", text)
    if match is None:
        return 0.0
    value = float(match.group())
    return min(value, 10.0)


class LLMJudge:
    """Async LLM-based judge that scores research reports."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        max_concurrent: int = 50,
    ):
        self.model = model
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def score_batch(
        self, queries: list, reports: list
    ) -> list[float]:
        """Score a batch of (query, report) pairs.

        Returns a list of floats in [0, 1].
        """
        tasks = [
            self._score_one(query, report)
            for query, report in zip(queries, reports)
        ]
        return await asyncio.gather(*tasks)

    async def _score_one(self, query: str, report: str) -> float:
        """Score a single (query, report) pair. Returns float in [0, 1]."""
        try:
            async with self.semaphore:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": JUDGE_PROMPT.format(
                                query=query, report=report
                            ),
                        }
                    ],
                )
                text = response.choices[0].message.content
                return parse_score(text) / 10.0
        except Exception as e:
            print(f"LLMJudge error: {e}")
            return 0.0
