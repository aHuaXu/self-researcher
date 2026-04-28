"""Report Writer agent - synthesizes research findings into reports."""

from typing import List, Dict, Any, Optional

from research_agent.agents.llm_client import LLMClient, create_llm_client
from research_agent.config import get_config
from research_agent.prompts.writer import get_writer_prompt


class ReportWriter:
    """Writer agent that synthesizes research findings into reports."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        if llm_client is None:
            config = get_config()
            llm_client = create_llm_client(config, "small")
        self.llm = llm_client

    def write(
        self,
        question: str,
        findings: List[Dict[str, Any]],
        plan_text: str = "",
    ) -> str:
        """
        Write a research report based on findings.

        Args:
            question: The original research question.
            findings: List of research trajectory items.
            plan_text: The planner output text (included as context,
                matching the training-time input format).

        Returns:
            The generated research report.
        """
        findings_text = self._format_findings(findings)
        if plan_text:
            findings_block = (
                f"=== Research Plan ===\n{plan_text}\n\n"
                f"=== Research Findings ===\n{findings_text}"
            )
        else:
            findings_block = findings_text
        messages = get_writer_prompt(question, findings_block)

        response = self.llm.chat(messages)

        if "error" in response:
            return f"Error writing report: {response['error']}"

        content = response.get("content", "")

        # Try to extract report from <report> tag
        import re
        pattern = r'<report>(.*?)</report>'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()

        return content.strip()

    def _format_findings(self, findings: List[Dict[str, Any]]) -> str:
        """Format research findings into readable text."""
        if not findings:
            return "No research findings available."

        formatted = []
        for i, finding in enumerate(findings):
            sub_topic = finding.get("sub_topic", "Unknown topic")
            answer = finding.get("answer", "No answer")
            trajectory = finding.get("trajectory", [])

            formatted.append(f"=== Sub-topic {i+1}: {sub_topic} ===")
            formatted.append(f"Answer: {answer}")

            if trajectory:
                formatted.append(f"Research steps ({len(trajectory)}):")
                for step in trajectory[:5]:
                    tool = step.get("tool", "unknown")
                    result = str(step.get("result", ""))[:500]
                    formatted.append(f"  - [{tool}] {result}")

            formatted.append("")

        return "\n".join(formatted)


def create_writer_agent(llm_client: Optional[LLMClient] = None) -> ReportWriter:
    """Factory function to create a writer agent."""
    return ReportWriter(llm_client)