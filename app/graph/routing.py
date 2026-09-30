"""Conditional edges."""
from __future__ import annotations

from app.state import AgentState

NODE_FOR_AGENT = {"rag": "rag_agent", "github": "github_agent", "calendar": "calendar_agent", "email": "email_agent"}


def route_after_input(state: AgentState) -> str:
    if state.get("final_response") or not state.get("plan"):
        return "finalize"
    return "orchestrator"


def route_next_step(state: AgentState) -> str:
    plan, step = state.get("plan") or [], state.get("step", 0)
    if step >= len(plan):
        return "finalize"
    return NODE_FOR_AGENT[plan[step]["agent"]]
