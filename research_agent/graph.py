"""LangGraph multi-agent orchestration for Research Agent."""

from typing import List, Dict, Any, TypedDict, Literal, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage

from research_agent.agents import create_planner_agent, create_executor_agent, create_writer_agent
from research_agent.config import get_config


class ResearchState(TypedDict):
    """State shared across all agents in the research graph."""
    question: str
    todos: List[Dict[str, Any]]
    findings: List[Dict[str, Any]]
    report: str
    status: str  # "planning", "executing", "writing", "done", "error"


def create_research_graph():
    """Create the LangGraph research orchestration graph."""

    # Create agents
    planner = create_planner_agent()
    executor = create_executor_agent()
    writer = create_writer_agent()

    # Define nodes
    def planning_node(state: ResearchState) -> ResearchState:
        """Run the planner to break down the question."""
        question = state["question"]
        todos = planner.plan(question)
        return {
            "question": question,
            "todos": todos,
            "findings": [],
            "report": "",
            "status": "executing" if todos else "error"
        }

    def executing_node(state: ResearchState) -> ResearchState:
        """Run the executor to perform research."""
        question = state["question"]
        todos = state.get("todos", [])
        findings = state.get("findings", [])

        answer, trajectory = executor.execute(question, "", todos)

        # Add trajectory to findings
        findings = findings + trajectory

        return {
            "question": question,
            "todos": todos,
            "findings": findings,
            "report": "",
            "status": "writing"
        }

    def writing_node(state: ResearchState) -> ResearchState:
        """Run the writer to synthesize findings into a report."""
        question = state["question"]
        findings = state.get("findings", [])

        if not findings:
            report = "No research findings available."
        else:
            report = writer.write(question, findings)

        return {
            "question": question,
            "todos": state["todos"],
            "findings": findings,
            "report": report,
            "status": "done"
        }

    def should_continue(state: ResearchState) -> Literal["executing", "writing", "end"]:
        """Determine if we should continue to next stage."""
        status = state.get("status", "")
        if status == "error":
            return "end"
        elif status == "executing":
            return "executing"
        elif status == "writing":
            return "writing"
        else:
            return "end"

    # Build the graph
    workflow = StateGraph(ResearchState)

    workflow.add_node("planner", planning_node)
    workflow.add_node("executor", executing_node)
    workflow.add_node("writer", writing_node)

    # Define edges: planning -> executing -> writing -> END
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "writer")
    workflow.add_edge("writer", END)

    # Set entry point
    workflow.set_entry_point("planner")

    return workflow.compile()


class ResearchAssistant:
    """Multi-agent research assistant using LangGraph."""

    def __init__(self):
        self.graph = create_research_graph()

    def research(self, question: str) -> Dict[str, Any]:
        """
        Run multi-agent research on a question.

        Args:
            question: The research question.

        Returns:
            Dictionary with findings and report.
        """
        initial_state = {
            "question": question,
            "todos": [],
            "findings": [],
            "report": "",
            "status": "planning"
        }

        result = self.graph.invoke(initial_state)
        return result

    def research_stream(self, question: str):
        """Run research with streaming updates."""
        initial_state = {
            "question": question,
            "todos": [],
            "findings": [],
            "report": "",
            "status": "planning"
        }

        for state in self.graph.stream(initial_state):
            yield state


# Convenience function
def create_research_assistant() -> ResearchAssistant:
    """Create a research assistant instance."""
    return ResearchAssistant()


def research(question: str) -> Dict[str, Any]:
    """
    Quick research function using the multi-agent system.

    Args:
        question: The research question.

    Returns:
        Dictionary with findings and report.
    """
    assistant = create_research_assistant()
    return assistant.research(question)