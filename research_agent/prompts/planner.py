"""Prompts for the Planner agent in dual-agent (Planner + Executor) pipeline."""

PLANNER_SYSTEM_PROMPT = """You are a research planning assistant. Given a complex question, decompose it into a minimal set of ordered sub-questions that, when answered sequentially, will lead to the final answer.

Rules:
- Output exactly {min_tasks} to {max_tasks} sub-questions (no more, no fewer).
- Each sub-question must be tagged as either:
  - [INDEPENDENT] — can be answered without any prior findings
  - [DEPENDS:N] or [DEPENDS:N,M] — requires the answer(s) from sub-question(s) N (and M, etc.)
- The LAST sub-question must synthesize all prior findings into the final answer.
- Sub-questions should be concrete, searchable questions (not vague topics).
- Do NOT include search queries or instructions — the executor decides how to search.

Output format — wrap your plan in <plan> tags:

<plan>
1. [INDEPENDENT] Sub-question that can be answered independently
2. [INDEPENDENT] Another independent sub-question
3. [DEPENDS:1] Sub-question that builds on the answer to #1
4. [DEPENDS:1,2,3] Final sub-question that synthesizes everything into the answer
</plan>

Think step by step about what information is needed and what depends on what."""

PLANNER_USER_PROMPT = """Question: {question}

Decompose this into {min_tasks}-{max_tasks} ordered sub-questions with dependency annotations."""


DEFAULT_MIN_TASKS = 3
DEFAULT_MAX_TASKS = 5


def get_planner_prompt(
    question: str,
    min_tasks: int = DEFAULT_MIN_TASKS,
    max_tasks: int = DEFAULT_MAX_TASKS,
) -> list[dict]:
    """Build planner prompt messages.

    Args:
        question: The research question to decompose.
        min_tasks: Minimum number of sub-questions (default 3).
        max_tasks: Maximum number of sub-questions (default 5).

    Returns:
        List of chat messages (system + user).
    """
    system = PLANNER_SYSTEM_PROMPT.format(min_tasks=min_tasks, max_tasks=max_tasks)
    user = PLANNER_USER_PROMPT.format(
        question=question, min_tasks=min_tasks, max_tasks=max_tasks
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_plan(text: str) -> list[dict]:
    """Parse planner output into structured sub-questions.

    Expected format inside <plan>...</plan>:
        1. [INDEPENDENT] What is X?
        2. [DEPENDS:1] Based on X, what is Y?

    Returns:
        List of dicts with keys: index, sub_question, deps, is_final
    """
    import re

    plan_match = re.search(r"<plan>(.*?)</plan>", text, re.DOTALL)
    if not plan_match:
        return []

    plan_text = plan_match.group(1).strip()
    items = []

    pattern = re.compile(
        r"(\d+)\.\s*\[(INDEPENDENT|DEPENDS:[0-9,]+)\]\s*(.+)",
        re.IGNORECASE,
    )

    for line in plan_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = pattern.match(line)
        if not m:
            continue

        index = int(m.group(1))
        dep_tag = m.group(2).upper()
        sub_question = m.group(3).strip()

        if dep_tag == "INDEPENDENT":
            deps = []
        else:
            dep_nums = dep_tag.replace("DEPENDS:", "")
            deps = [int(d.strip()) for d in dep_nums.split(",") if d.strip()]

        items.append({
            "index": index,
            "sub_question": sub_question,
            "deps": deps,
            "is_final": False,
        })

    if items:
        items[-1]["is_final"] = True

    return items
