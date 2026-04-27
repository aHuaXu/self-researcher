import re
from typing import Dict, List


def _parse_tasks(plan_text: str) -> List[Dict[str, str]]:
    """Parse planner output into structured tasks."""
    pattern = re.compile(
        r'^\d+\.\s*\[(\w+)\]\s*(?:Sub-topic|子主题)\s*[:：]\s*(.+?)$',
        re.MULTILINE
    )
    tasks = []
    for match in pattern.finditer(plan_text):
        priority = match.group(1).upper()
        subtopic = match.group(2).strip()
        tasks.append({"priority": priority, "subtopic": subtopic})
    return tasks


def _keyword_overlap(a: str, b: str) -> float:
    """Pairwise keyword overlap ratio between two strings."""
    words_a = set(re.findall(r'\w+', a.lower()))
    words_b = set(re.findall(r'\w+', b.lower()))
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    return len(intersection) / min(len(words_a), len(words_b))


def planner_rules(plan_text: str) -> float:
    """Score planner output on task count, uniqueness, and priority distribution."""
    if not plan_text or not plan_text.strip():
        print("[RuleReward] WARNING: planner_rules received empty plan_text, returning 0.0", flush=True)
        return 0.0

    tasks = _parse_tasks(plan_text)
    if not tasks:
        print(
            f"[RuleReward] WARNING: planner_rules parsed 0 tasks from plan_text "
            f"({len(plan_text)} chars), returning 0.0",
            flush=True,
        )
        return 0.0

    score = 0.0

    if 3 <= len(tasks) <= 7:
        score += 1.0

    has_duplicate = False
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            if _keyword_overlap(tasks[i]["subtopic"], tasks[j]["subtopic"]) >= 0.6:
                has_duplicate = True
                break
        if has_duplicate:
            break
    if not has_duplicate:
        score += 1.0

    distinct_priorities = set(t["priority"] for t in tasks)
    valid_priorities = distinct_priorities & {"HIGH", "MEDIUM", "LOW"}
    if len(valid_priorities) >= 2:
        score += 0.5

    return score / 2.5


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
