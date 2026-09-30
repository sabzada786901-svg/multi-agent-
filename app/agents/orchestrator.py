"""Orchestrator helpers: plan normalisation, shared context between agents, final composition."""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from app.state import AgentState

AGENT_ORDER = ["rag", "github", "calendar", "email"]
LABELS = {"rag": "📄 PDF", "github": "🐙 GitHub", "calendar": "📅 Calendar", "email": "✉️ Email"}
MAX_CONTEXT = 6000


def normalize_plan(plan: list[dict]) -> list[dict]:
    """Keep only known agents, one step per agent, in a safe order (email always last)."""
    seen: dict[str, dict] = {}
    for step in plan or []:
        a = step.get("agent")
        if a in AGENT_ORDER and a not in seen:
            seen[a] = {"agent": a, "task": str(step.get("task") or "").strip()}
    return [seen[a] for a in AGENT_ORDER if a in seen]


def context_from_outputs(state: AgentState) -> str:
    """Results of earlier agents, handed to later ones (state is preserved between agents)."""
    outs = [o for o in (state.get("agent_outputs") or []) if o.get("ok", True)]
    text = "\n\n".join(f"[{o['agent']}]\n{o['text']}" for o in outs)
    return text[:MAX_CONTEXT]


def history_text(state: AgentState, max_msgs: int = 6, max_chars: int = 3000) -> str:
    msgs = (state.get("messages") or [])[:-1][-max_msgs:]  # exclude the current user turn
    lines = []
    for m in msgs:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        lines.append(f"{role}: {getattr(m, 'content', '')}")
    return "\n".join(lines)[-max_chars:]


def compose_final(state: AgentState) -> str:
    outputs = state.get("agent_outputs") or []
    if not outputs:
        body = "I couldn't complete that request."
    elif len(outputs) == 1:
        body = outputs[0]["text"]
    else:
        body = "\n\n".join(f"{LABELS.get(o['agent'], o['agent'])}\n{o['text']}" for o in outputs)
    prefix = state.get("greeting_prefix")
    return f"{prefix} {body}".strip() if prefix else body


def as_ai_message(text: str) -> AIMessage:
    return AIMessage(content=text)
