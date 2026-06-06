"""Prompts for the Planner agent in dual-agent (Planner + Executor) pipeline."""

from dataclasses import dataclass, field

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


# ---------------------------------------------------------------------------
# Interleaved (Hi-IGPO Phase 2b) planner: ONE subtask per turn, or <answer>.
# Unlike the DAG planner above (one-shot multi-step <plan>), the interleaved
# planner is called repeatedly; each turn it either delegates the single next
# sub-question to the (frozen) executor or finalizes the answer. See design §3.
# ---------------------------------------------------------------------------
INTERLEAVED_PLANNER_SYSTEM_PROMPT = """You are a research planner answering a complex question step by step. You do NOT search yourself — a separate research executor does that. You decide what to research next, one step at a time.

Each turn, output EXACTLY ONE of the following:
- If you still need more information, propose the SINGLE most useful next sub-question. The executor will research it and return findings:
<subtask>one concrete, self-contained, searchable sub-question</subtask>
- If the findings so far are enough to answer the original question, give the final answer:
<answer>final answer</answer>

Rules:
- Exactly ONE <subtask> per turn — never a numbered list or multi-step plan.
- Base each new sub-question on the findings returned so far.
- Switch to <answer> as soon as you can answer. Put ONLY the final answer (words, a number, or a short phrase) inside <answer></answer>, with no explanation. For a yes/no question, answer only yes or no."""

INTERLEAVED_PLANNER_USER_PROMPT = """Question: {question}

Propose the first sub-question to research (in <subtask></subtask>), or answer directly (in <answer></answer>) if you already can."""


def get_interleaved_planner_prompt(question: str) -> list[dict]:
    """Build interleaved planner prompt messages (one subtask per turn, or <answer>)."""
    return [
        {"role": "system", "content": INTERLEAVED_PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": INTERLEAVED_PLANNER_USER_PROMPT.format(question=question)},
    ]


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


@dataclass
class SubTask:
    """A single sub-task parsed from the planner's output."""

    index: int  # 1-based sub-question number
    sub_question: str  # The sub-question text to research
    deps: list[int] = field(default_factory=list)  # Indices of dependency sub-questions (empty = INDEPENDENT)
    is_final: bool = False  # True for the last sub-question that produces the final answer


def parse_plan(text: str) -> list[SubTask]:
    """Parse planner output into structured sub-questions.

    Expected format inside <plan>...</plan>:
        1. [INDEPENDENT] What is X?
        2. [DEPENDS:1] Based on X, what is Y?

    Returns:
        List of SubTask instances.
    """
    import re

    plan_match = re.search(r"<plan>(.*?)</plan>", text, re.DOTALL)
    if not plan_match:
        return []

    plan_text = plan_match.group(1).strip()
    items: list[SubTask] = []

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

        items.append(SubTask(
            index=index,
            sub_question=sub_question,
            deps=deps,
        ))

    if items:
        items[-1].is_final = True

    return items
