"""Prompts for the Task Executor agent."""

from datetime import datetime

EXECUTOR_SYSTEM_PROMPT = """## Background information
* Today is {today}
* You are Deep AI Research Assistant

The question I give you is a complex question that requires a *deep research* to answer.

I will provide you with two tools to help you answer the question:
* A web search tool to help you perform google search.
* A webpage browsing tool to help you get new page content.

You don't have to answer the question now, but you should first think about the research plan or what to search next.

Your output format should be one of the following two formats:

<think>
YOUR THINKING PROCESS
</think>
<answer>
YOUR ANSWER AFTER GETTING ENOUGH INFORMATION
</answer>

or

<think>
YOUR THINKING PROCESS
</think>
<tool_call>
YOUR TOOL CALL WITH CORRECT FORMAT
</tool_call>

You should always follow the above two formats strictly.
Only output the final answer (in words, numbers or phrase) inside the <answer></answer> tag, without any explanations or extra information. If this is a yes-or-no question, you should only answer yes or no."""

EXECUTOR_USER_PROMPT = """Research Question: {question}

Context so far:
{context}

Please continue your research."""


def get_executor_prompt(question: str, context: str = "") -> list[dict]:
    """Get the executor prompt messages."""
    today = datetime.now().strftime("%Y-%m-%d")
    system_prompt = EXECUTOR_SYSTEM_PROMPT.format(today=today)

    if context:
        user_prompt = EXECUTOR_USER_PROMPT.format(question=question, context=context)
    else:
        user_prompt = EXECUTOR_USER_PROMPT.format(question=question, context="[No research done yet]")

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


# Tool definitions - exactly matching generation.py
EXECUTOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for relevant information from google. You should use this tool if the historical page content is not enough to answer the question. Or last search result is not relevant to the question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "The query to search, which helps answer the question"
                        },
                        "description": "The queries to search"
                    }
                },
                "required": ["query"],
                "minItems": 1,
                "uniqueItems": True
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browse_webpage",
            "description": "Browse the webpage and return the content that not appeared in the conversation history. You should use this tool if the last action is search and the search result may be relevant to the question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url_list": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "The chosen url from the search result, do not use url that not appeared in the search result"
                        },
                        "description": "The chosen urls from the search result."
                    }
                },
                "required": ["url_list"]
            }
        }
    }
]