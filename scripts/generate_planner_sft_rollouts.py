#!/usr/bin/env python3
"""Generate Planner-SFT interleaved rollouts with MiniMax + real search/browse tools.

MiniMax is used as a teacher for:
  - deciding the next Planner action: <subtask>...</subtask> or <answer>...</answer>
  - compressing real search/browse outputs into a clean executor finding

The evidence path still goes through this repo's ``web_search`` and ``browse_webpage``.
The script writes raw rollout JSON compatible with ``build_planner_sft_from_rollouts.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_agent.config import get_config
from research_agent.tools import browse_webpage, get_tool_state, web_search


PLANNER_SYSTEM = """You are generating clean supervised data for a research Planner agent.

The Planner does not search or browse by itself. A separate Executor will research each subtask
and return findings. Your job is to decide the single next research subtask or the final answer.

Output exactly one tag and nothing else:
- <subtask>one concrete, self-contained, searchable sub-question</subtask>
- <answer>short final answer</answer>

Rules:
- Never output <think>, <tool_call>, <tool_response>, search instructions, URLs, or numbered plans.
- Each <subtask> must be a question the Executor can research independently.
- Use previous Executor findings when available.
- Use <answer> only when the findings are sufficient. The answer must be a short phrase, name, date, number, or yes/no."""

FINDING_SYSTEM = """You compress real web-search and webpage-browse outputs into one clean Executor finding.

Write only the useful factual finding text. Start directly with the finding.
Do not include XML tags, tool names, JSON, URLs, "let me", "search results", or other search-process language.
If the evidence is weak, say what was found and what remains uncertain. Do not invent evidence."""

BANNED_PLANNER_RE = re.compile(
    r"<tool_call>|<tool_response>|</?think>|web_search|browse_webpage|search for|open result",
    re.IGNORECASE,
)
PLACEHOLDER_PAYLOADS = {
    "",
    "...",
    "short final answer",
    "final answer",
    "one concrete searchable sub-question",
    "one concrete, self-contained, searchable sub-question",
}
LOW_VALUE_URL_HOSTS = (
    "facebook.com",
    "x.com",
    "twitter.com",
    "instagram.com",
    "youtube.com",
    "youtu.be",
    "tiktok.com",
)


def is_valid_payload(payload: str) -> bool:
    normalized = re.sub(r"\s+", " ", payload or "").strip().lower()
    return (
        normalized not in PLACEHOLDER_PAYLOADS
        and len(payload or "") <= 400
        and "<" not in (payload or "")
        and ">" not in (payload or "")
        and not BANNED_PLANNER_RE.search(payload or "")
    )


def last_valid_tag(text: str) -> tuple[str, str]:
    matches = list(
        re.finditer(r"<(answer|subtask)>([^<>]{1,400})</\1>", text or "", re.DOTALL | re.IGNORECASE)
    )
    for match in reversed(matches):
        kind = match.group(1).lower()
        payload = re.sub(r"\s+", " ", match.group(2)).strip()
        if is_valid_payload(payload):
            return kind, payload
    return "malformed", re.sub(r"\s+", " ", text or "").strip()


def call_llm(client: OpenAI, model: str, messages: list[dict[str, str]], *, max_tokens: int, temperature: float) -> str:
    last_error = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:  # pragma: no cover - network path
            last_error = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"LLM call failed after retries: {last_error}")


def extract_tag(text: str) -> tuple[str, str]:
    parse_text = text or ""
    if "<think>" in parse_text.lower():
        if "</think>" in parse_text.lower():
            after_think = re.split(r"</think>", parse_text, flags=re.IGNORECASE)[-1]
            kind, payload = last_valid_tag(after_think)
            if kind != "malformed":
                return kind, payload
        # Reasoning models may state the final XML tag inside <think> but not repeat it
        # after </think>. We keep only the extracted clean tag in SFT, never the raw think.
        return last_valid_tag(parse_text)

    return last_valid_tag(parse_text)


def planner_turn(
    client: OpenAI,
    model: str,
    question: str,
    findings: list[str],
    *,
    force_answer: bool,
    max_tokens: int,
) -> tuple[str, str, str]:
    user_lines = [f"Original question:\n{question}"]
    if findings:
        user_lines.append("Executor findings so far:")
        for idx, finding in enumerate(findings, start=1):
            user_lines.append(f"{idx}. {finding}")
    if force_answer:
        user_lines.append("This is the final Planner turn. Output <answer>...</answer> only.")
    else:
        user_lines.append("Output the next <subtask>...</subtask>, or <answer>...</answer> if enough is known.")

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": "\n\n".join(user_lines)},
    ]
    raw = call_llm(client, model, messages, max_tokens=max_tokens, temperature=0.2)
    kind, payload = extract_tag(raw)
    if kind == "malformed":
        repair_prompt = f"""The previous response was invalid because it did not contain exactly one clean XML tag.

Previous response:
{truncate(raw, 2000)}

Original question:
{question}

Executor findings so far:
{chr(10).join(f'- {finding}' for finding in findings) if findings else '(none)'}

Return exactly one tag and nothing else. Do not include reasoning or <think>.
{"Return <answer>short final answer</answer>." if force_answer else "Return <subtask>one concrete searchable sub-question</subtask>, or <answer>short final answer</answer> if enough is known."}"""
        raw = call_llm(
            client,
            model,
            [{"role": "system", "content": PLANNER_SYSTEM}, {"role": "user", "content": repair_prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        kind, payload = extract_tag(raw)
    return kind, payload, raw


def safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def extract_urls(search_text: str, top_k: int) -> list[str]:
    data = safe_json_loads(search_text)
    urls = []
    if isinstance(data, list):
        for block in data:
            for page in (block or {}).get("web_page_info_list", []):
                url = page.get("url")
                host = urlparse(url).netloc.lower() if isinstance(url, str) else ""
                if (
                    isinstance(url, str)
                    and url.startswith(("http://", "https://"))
                    and url not in urls
                    and not any(bad in host for bad in LOW_VALUE_URL_HOSTS)
                ):
                    urls.append(url)
                if len(urls) >= top_k:
                    return urls
    return urls


def truncate(text: str, max_chars: int) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


def make_finding(
    client: OpenAI,
    model: str,
    *,
    question: str,
    subtask: str,
    search_result: str,
    browse_result: str,
    max_tokens: int,
    context_chars: int,
) -> str:
    prompt = f"""Original question:
{question}

Executor subtask:
{subtask}

Real web_search result:
{truncate(search_result, context_chars)}

Real browse_webpage result:
{truncate(browse_result, context_chars)}

Compress the evidence into one concise finding that helps answer the original question."""
    raw = call_llm(
        client,
        model,
        [{"role": "system", "content": FINDING_SYSTEM}, {"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.1,
    )
    finding = re.sub(r"</?(think|answer|tool_call|tool_response|subtask)>", "", raw, flags=re.IGNORECASE)
    finding = re.sub(r"\s+", " ", finding).strip()
    return finding


def execute_subtask(
    client: OpenAI,
    model: str,
    *,
    question: str,
    subtask: str,
    browse_top_k: int,
    finding_max_tokens: int,
    finding_context_chars: int,
) -> tuple[str, dict[str, Any]]:
    t0 = time.time()
    search_result = web_search.invoke({"query": [subtask]})
    search_elapsed = time.time() - t0
    urls = extract_urls(search_result, browse_top_k)

    browse_result = "[]"
    browse_elapsed = 0.0
    if urls:
        t1 = time.time()
        browse_result = browse_webpage.invoke({"url_list": urls, "goal": subtask})
        browse_elapsed = time.time() - t1

    finding = make_finding(
        client,
        model,
        question=question,
        subtask=subtask,
        search_result=search_result,
        browse_result=browse_result,
        max_tokens=finding_max_tokens,
        context_chars=finding_context_chars,
    )
    meta = {
        "subtask": subtask,
        "urls": urls,
        "search_elapsed_sec": round(search_elapsed, 3),
        "browse_elapsed_sec": round(browse_elapsed, 3),
        "search_result": safe_json_loads(search_result) if safe_json_loads(search_result) is not None else search_result,
        "browse_result": safe_json_loads(browse_result) if safe_json_loads(browse_result) is not None else browse_result,
    }
    return finding, meta


def generate_one(
    client: OpenAI,
    model: str,
    row: dict[str, Any],
    args,
) -> dict[str, Any]:
    question = str(row["question"]).strip()
    get_tool_state().reset_for_question(question)

    findings: list[str] = []
    turns = []
    tool_traces = []
    final_answer = ""

    for turn_idx in range(args.max_planner_turns):
        force_answer = turn_idx == args.max_planner_turns - 1
        kind, payload, raw = planner_turn(
            client,
            model,
            question,
            findings,
            force_answer=force_answer,
            max_tokens=args.planner_max_tokens,
        )

        if kind == "answer":
            final_answer = payload
            turns.append(
                {
                    "turn": turn_idx,
                    "planner_output_raw": raw,
                    "parsed_kind": "answer",
                    "payload": payload,
                    "executor_finding": None,
                }
            )
            break

        # Malformed turns are kept in raw output for audit; do not waste tool calls on them.
        if kind != "subtask" or not payload:
            turns.append(
                {
                    "turn": turn_idx,
                    "planner_output_raw": raw,
                    "parsed_kind": "malformed",
                    "payload": payload,
                    "executor_finding": None,
                }
            )
            break

        subtask = payload
        finding, tool_meta = execute_subtask(
            client,
            model,
            question=question,
            subtask=subtask,
            browse_top_k=args.browse_top_k,
            finding_max_tokens=args.finding_max_tokens,
            finding_context_chars=args.finding_context_chars,
        )
        findings.append(finding)
        tool_traces.append(tool_meta)
        turns.append(
            {
                "turn": turn_idx,
                "planner_output_raw": raw,
                "parsed_kind": "subtask" if kind == "subtask" else "malformed",
                "payload": subtask,
                "executor_finding": finding,
            }
        )
        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    return {
        "question": question,
        "gt": str(row.get("gt", "")).strip(),
        "difficulty": int(row.get("difficulty", -1)),
        "source_index": str(row.get("source_index", "")),
        "answer": final_answer,
        "num_planner_turns": len(turns),
        "turns": turns,
        "tool_traces": tool_traces,
    }


def load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return data


def write_json(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/planner_sft_seed_l3_500.parquet")
    parser.add_argument("--output", default="outputs/planner_sft/minimax_rollouts.json")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows after --start")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-planner-turns", type=int, default=4)
    parser.add_argument("--browse-top-k", type=int, default=2)
    parser.add_argument("--planner-max-tokens", type=int, default=2048)
    parser.add_argument("--finding-max-tokens", type=int, default=384)
    parser.add_argument("--finding-context-chars", type=int, default=6000)
    parser.add_argument("--sleep-sec", type=float, default=0.0)
    parser.add_argument("--n-samples", type=int, default=1,
                        help="Best-of-N: generate N rollouts per question (tagged with sample_idx). "
                             "Build script picks the highest-F1 passing sample per question when --best-of-n.")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "MiniMax-M2.7"))
    args = parser.parse_args()

    config = get_config()
    get_tool_state().ensure_initialized()
    client = OpenAI(
        base_url=config.llm.base_url,
        api_key=config.llm.api_key,
        timeout=config.llm.timeout,
        max_retries=3,
    )

    df = pd.read_parquet(args.input)
    rows = df.iloc[args.start :]
    if args.limit and args.limit > 0:
        rows = rows.iloc[: args.limit]

    output = Path(args.output)
    generated = load_existing(output) if args.resume else []
    # Resume: count existing samples per (source_index, question) so --n-samples can top up.
    done_counts: dict[tuple[str, str], int] = {}
    for row in generated:
        key = (str(row.get("source_index", "")), str(row.get("question", "")))
        done_counts[key] = done_counts.get(key, 0) + 1

    for local_idx, (_, row) in enumerate(rows.iterrows(), start=1):
        key = (str(row.get("source_index", "")), str(row.get("question", "")))
        have = done_counts.get(key, 0)
        n_to_gen = args.n_samples - have
        if args.resume and n_to_gen <= 0:
            continue
        print(
            f"[{local_idx}/{len(rows)}] L{row.get('difficulty')} {row.get('source_index')}: "
            f"{str(row.get('question'))[:100]} (gen {n_to_gen}/{args.n_samples})",
            flush=True,
        )
        for sample_idx in range(have, have + n_to_gen):
            try:
                rec = generate_one(client, args.model, row.to_dict(), args)
                rec["sample_idx"] = sample_idx
                generated.append(rec)
            except Exception as exc:
                generated.append(
                    {
                        "question": str(row.get("question", "")),
                        "gt": str(row.get("gt", "")),
                        "difficulty": int(row.get("difficulty", -1)),
                        "source_index": str(row.get("source_index", "")),
                        "sample_idx": sample_idx,
                        "error": str(exc),
                        "turns": [],
                    }
                )
                print(f"  ERROR (sample {sample_idx}): {exc}", flush=True)
            write_json(output, generated)
        done_counts[key] = args.n_samples

    print(f"Wrote {len(generated)} rollout rows to {output}")


if __name__ == "__main__":
    main()
