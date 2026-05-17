"""Context for tool execution (parallel web_search in one GRPO tool batch)."""

from contextvars import ContextVar
from typing import Optional

# Index into messages_list for the current tool call (same as execute_predictions tuple[0]).
tool_rollout_message_idx: ContextVar[Optional[int]] = ContextVar(
    "tool_rollout_message_idx", default=None
)

# User question for this rollout line; avoids races on ToolState.current_question when web_search runs concurrently.
tool_rollout_user_query: ContextVar[Optional[str]] = ContextVar(
    "tool_rollout_user_query", default=None
)
