import re
from typing import Dict, List


def _parse_plan_tasks(plan_text: str) -> List[Dict[str, str]]:
    """Parse planner output into structured tasks.

    Supports the new <plan> format:
      1. [INDEPENDENT] Sub-question text
      2. [DEPENDS:1] Another sub-question

    Also supports legacy format for backward compatibility:
      1. [HIGH] Sub-topic description
    """
    # New format: inside <plan> tags
    plan_match = re.search(r"<plan>(.*?)</plan>", plan_text, re.DOTALL)
    if plan_match:
        plan_body = plan_match.group(1)
    else:
        plan_body = plan_text

    pattern = re.compile(
        r"^\d+\.\s*\[([^\]]+)\]\s*(.+?)$",
        re.MULTILINE,
    )
    tasks = []
    for match in pattern.finditer(plan_body):
        tag = match.group(1).strip().upper()
        content = match.group(2).strip()
        if content:
            tasks.append({"tag": tag, "content": content})
    return tasks


def _keyword_overlap(a: str, b: str) -> float:
    """Pairwise keyword overlap ratio between two strings."""
    words_a = set(re.findall(r"\w+", a.lower()))
    words_b = set(re.findall(r"\w+", b.lower()))
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    return len(intersection) / min(len(words_a), len(words_b))


def planner_rules(
    plan_text: str,
    min_tasks: int = 3,
    max_tasks: int = 5,
) -> float:
    """Score planner output on sub-task count and uniqueness.

    Scoring (normalized to [0, 1]):
      - Task count in [min_tasks, max_tasks]: +1.0
      - No duplicate sub-questions (overlap < 0.6): +1.0

    Args:
        plan_text: Raw planner output text.
        min_tasks: Minimum acceptable sub-task count (default 3).
        max_tasks: Maximum acceptable sub-task count (default 5).

    Returns:
        Score in [0.0, 1.0].
    """
    if not plan_text or not plan_text.strip():
        print("[RuleReward] WARNING: planner_rules received empty plan_text", flush=True)
        return 0.0

    tasks = _parse_plan_tasks(plan_text)
    if not tasks:
        print(
            f"[RuleReward] WARNING: planner_rules parsed 0 tasks from "
            f"plan_text ({len(plan_text)} chars)",
            flush=True,
        )
        return 0.0

    score = 0.0

    # Sub-task count in valid range
    if min_tasks <= len(tasks) <= max_tasks:
        score += 1.0

    # No near-duplicate sub-questions
    has_duplicate = False
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            if _keyword_overlap(tasks[i]["content"], tasks[j]["content"]) >= 0.6:
                has_duplicate = True
                break
        if has_duplicate:
            break
    if not has_duplicate:
        score += 1.0

    return score / 2.0


def executor_rules(trajectory: List[Dict], max_turns: int, actual_turns: int) -> float:
    """Score executor trajectory on search success, browsing, and turn efficiency."""
    score = 0.0

    has_good_search = any(
        step.get("tool") == "web_search" and len(step.get("result", "")) > 10
        for step in trajectory
    )
    if has_good_search:
        score += 1.0

    has_browse = any(step.get("tool") == "browse_webpage" for step in trajectory)
    if has_browse:
        score += 1.0

    has_good_browse = any(
        step.get("tool") == "browse_webpage" and len(step.get("result", "")) > 50
        for step in trajectory
    )
    if has_good_browse:
        score += 1.0

    if actual_turns < max_turns:
        score += 1.0

    return score / 4.0
