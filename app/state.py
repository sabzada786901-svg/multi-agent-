"""Strongly typed LangGraph state shared by every agent."""
from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages

AgentName = Literal["rag", "github", "calendar", "email"]


class PlanStep(TypedDict):
    agent: AgentName
    task: str


class AgentState(TypedDict, total=False):
    user_input: str
    intent: str | None
    greeting_prefix: str | None
    messages: Annotated[list[AnyMessage], add_messages]

    # orchestration
    plan: list[PlanStep]
    step: int
    agent_outputs: list[dict]

    # RAG
    retrieved_context: list[str]
    retrieval_metadata: list[dict]
    grounding_result: dict | None

    # MCP agents
    github_result: dict | None
    calendar_result: dict | None
    email_result: dict | None

    # human-in-the-loop
    requires_confirmation: bool
    pending_action: dict | None

    final_response: str | None


def initial_turn_state(user_input: str) -> AgentState:
    """Per-turn fields are reset each turn; `messages` accumulates via the checkpointer."""
    return {
        "user_input": user_input,
        "intent": None,
        "greeting_prefix": None,
        "messages": [HumanMessage(content=user_input)],
        "plan": [],
        "step": 0,
        "agent_outputs": [],
        "retrieved_context": [],
        "retrieval_metadata": [],
        "grounding_result": None,
        "github_result": None,
        "calendar_result": None,
        "email_result": None,
        "requires_confirmation": False,
        "pending_action": None,
        "final_response": None,
    }
