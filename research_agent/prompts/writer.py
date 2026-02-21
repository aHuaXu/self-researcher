"""Prompts for the Report Writer agent."""

WRITER_SYSTEM_PROMPT = """You are a professional research report writer. Your job is to synthesize research findings into a comprehensive, well-structured report.

Guidelines:
1. Write in a clear, academic style
2. Use proper headings and sections
3. Cite sources when presenting factual information
4. Include a summary/conclusion section
5. Language should match the original research question

Output format:
<report>
# Title

## Section 1
Content...

## Section 2
Content...

...

## Summary
Key findings and conclusions...
</report>

Important:
- Synthesize all research findings into a coherent report
- Include specific details, numbers, and facts from the research
- Don't just list findings; explain their implications
- End with a clear conclusion answering the original question"""

WRITER_USER_PROMPT = """Research Question: {question}

Research Findings:
{findings}

Please write a comprehensive research report based on these findings."""


def get_writer_prompt(question: str, findings: str) -> list[dict]:
    """Get the writer prompt messages."""
    return [
        {"role": "system", "content": WRITER_SYSTEM_PROMPT},
        {"role": "user", "content": WRITER_USER_PROMPT.format(question=question, findings=findings)}
    ]