"""DR-Venus (inclusionAI/DR-Venus-4B-SFT) interaction format, shared by the training
rollout (igpo_generation.py) and the validation rollout (generation.py).

Backbone is Qwen3-4B-Thinking-2507 (thinking-only). The native protocol (authoritative
source: DR-Venus Inference/run_demo.py) is:
  - system : DRVENUS_SYSTEM_PROMPT (tools embedded as <tools> JSON in the prompt TEXT,
             NOT via apply_chat_template(tools=...)).
  - assistant : "<think>...</think> ... <tool_call>{name,arguments}</tool_call>" or "<answer>...</answer>".
  - tool result : a USER-role message "<tool_response>\n{result}\n</tool_response>"
                  (NOT a tool-role message).
  - generation stop : ["<tool_response>"].
  - tools : search(query: list[str]) and visit(url: list[str], goal: str).

This repo executes tools in-process via research_agent.tools (web_search / browse_webpage),
so DR-Venus tool names/args are mapped onto this repo's tools here. The belief GT wrapper
for the thinking model closes </think> before <answer> (see DRVENUS_GT_ANSWER_PREFIX).
"""
import json
import re

# Authoritative system prompt copied verbatim from DR-Venus Inference/run_demo.py.
DRVENUS_SYSTEM_PROMPT = (
    "You are a deep research assistant. Your core function is to conduct thorough, "
    "multi-source investigations into any topic. You must handle both broad, open-domain "
    "inquiries and queries within specialized academic fields. For each user request, you "
    "must actively seek out and **cross-check information** from credible and diverse "
    "sources, then integrate the findings into a response that is comprehensive, accurate, "
    "well-structured, and objective. When you have gathered sufficient information and are "
    "ready to provide the definitive response, you must enclose the entire final answer in "
    "`<answer></answer>` tags.\n\n"
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    '{"type": "function", "function": {"name": "search", "description": "Perform Google web '
    'searches then returns a string of the top search results. Accepts multiple queries.", '
    '"parameters": {"type": "object", "properties": {"query": {"type": "array", "items": '
    '{"type": "string", "description": "The search query."}, "minItems": 1, "description": '
    '"The list of search queries."}}, "required": ["query"]}}}\n'
    '{"type": "function", "function": {"name": "visit", "description": "Visit webpage(s) and '
    'return the summary of the content.", "parameters": {"type": "object", "properties": '
    '{"url": {"type": "array", "items": {"type": "string"}, "description": "The URL(s) of the '
    'webpage(s) to visit. Can be a single URL or an array of URLs."}, "goal": {"type": '
    '"string", "description": "The specific information goal for visiting webpage(s)."}}, '
    '"required": ["url", "goal"]}}}\n'
    "</tools>\n\n"
    "For each function call, return a json object with function name and arguments within "
    "<tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>"
)

# Stop string for generation: the model must stop before fabricating a tool response.
DRVENUS_STOP = ["<tool_response>"]

# Belief GT wrapper for the thinking model: close the open <think> then emit the answer
# (matches DR-Venus RL generation.py PREFIX/SUFFIX).
DRVENUS_GT_ANSWER_PREFIX = "Now there's enough information to answer\n</think>\n<answer>\n"
DRVENUS_GT_ANSWER_SUFFIX = "\n</answer>"

# Final-turn force-answer: user nudge that *replaces* the last tool_response message
# (DR-Venus run_demo.py max-steps variant). NOTE: deliberately describes the tag by name
# ("answer tags") rather than embedding a literal "<answer>" — a bare open tag here would be
# counted by the scorer's check_tags_balance and unbalance the whole trajectory (-> 0 score).
DRVENUS_FORCE_ANSWER_USER = (
    "<tool_response>\n"
    "You've reached the maximum number of tool calls. "
    "Based on the information and knowledge you have gathered, "
    "please provide the entire final answer wrapped in answer tags now.\n"
    "</tool_response>"
)

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def map_tool_call(name: str, arguments: dict):
    """Map a DR-Venus tool call (search/visit) onto this repo's tools (web_search/browse_webpage).

    Returns (mapped_name, mapped_args) or raises ValueError on an unknown/invalid call.
      search(query: list[str])            -> web_search(query: list[str])
      visit(url: list[str], goal: str)    -> browse_webpage(url_list: list[str])
    """
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be a dict")
    if name == "search":
        query = arguments.get("query")
        if isinstance(query, str):
            query = [query]
        if not isinstance(query, list) or not query:
            raise ValueError("search.query must be a non-empty list")
        return "web_search", {"query": query}
    if name == "visit":
        url = arguments.get("url")
        if isinstance(url, str):
            url = [url]
        if not isinstance(url, list) or not url:
            raise ValueError("visit.url must be a non-empty list")
        goal = arguments.get("goal", "")
        return "browse_webpage", {"url_list": url, "goal": goal if isinstance(goal, str) else ""}
    raise ValueError(f"unknown DR-Venus tool: {name}")


def tool_response_message(result: str) -> dict:
    """Build the DR-Venus tool-result message: a USER-role <tool_response> block."""
    return {"role": "user", "content": f"<tool_response>\n{result}\n</tool_response>"}


def _unmap_tool_call(mapped_name: str, mapped_args: dict):
    """Reverse map this repo's tool call back to DR-Venus names for in-distribution history.
      web_search(query)        -> search(query)
      browse_webpage(url_list) -> visit(url, goal="")
    """
    if mapped_name == "web_search":
        return {"name": "search", "arguments": {"query": mapped_args.get("query", [])}}
    if mapped_name == "browse_webpage":
        return {"name": "visit", "arguments": {"url": mapped_args.get("url_list", []),
                                               "goal": mapped_args.get("goal", "")}}
    return {"name": mapped_name, "arguments": mapped_args}


def render_assistant_toolcall(reasoning: str, mapped_name: str, mapped_args: dict) -> str:
    """Reconstruct the assistant turn text in DR-Venus format for conversation history:
    "<think>\n{reasoning}\n</think>\n<tool_call>\n{json}\n</tool_call>" (tool name reverse-mapped)."""
    call = _unmap_tool_call(mapped_name, mapped_args)
    return (
        f"<think>\n{reasoning}\n</think>\n"
        f"<tool_call>\n{json.dumps(call, ensure_ascii=False)}\n</tool_call>"
    )


def parse_assistant_output(text: str):
    """Parse a DR-Venus assistant turn.

    Returns (is_stop, reasoning, payload):
      - answer present      -> (True,  reasoning, answer_str)
      - tool_call present   -> (False, reasoning, {"name","arguments"})  (mapped to this repo's tools)
      - otherwise (malformed/no action) -> (True, "", "")  (treated as stop, like this repo's convention)
    `reasoning` is the <think> content if present else the text before the first action tag.
    """
    think_m = _THINK_RE.search(text)
    reasoning = think_m.group(1).strip() if think_m else ""

    ans_m = _ANSWER_RE.search(text)
    if ans_m:
        if not reasoning:
            reasoning = text.split("<answer>")[0].strip()
        return True, reasoning, ans_m.group(1).strip()

    tc_m = _TOOL_RE.search(text)
    if tc_m:
        if not reasoning:
            reasoning = text.split("<tool_call>")[0].strip()
        try:
            tc = json.loads(tc_m.group(1).strip())
            mapped_name, mapped_args = map_tool_call(tc.get("name"), tc.get("arguments", {}))
            return False, reasoning, {"name": mapped_name, "arguments": mapped_args}
        except Exception:
            return True, "", ""
    return True, "", ""
