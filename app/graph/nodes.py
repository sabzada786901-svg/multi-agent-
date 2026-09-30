"""LangGraph nodes."""
from __future__ import annotations

import asyncio
from dataclasses import asdict

from langgraph.types import interrupt

from app import mcp
from app.agents import calendar_agent, email_agent, github_agent, orchestrator, rag_agent
from app.agents.base import AgentResult, format_pending
from app.config import ConfigError
from app.guardrails.input_guardrail import evaluate_input
from app.guardrails.output_guardrail import apply_output_guardrail
from app.state import AgentState
from app.utils.logging import get_logger
from app.utils.security import friendly_error, redact_secrets

LOG = get_logger("nodes")
YES = {"y", "yes", "yeah", "yep", "ok", "okay", "confirm", "send", "send it", "do it", "sure", "haan", "han", "ji"}


# ---------------------------------------------------------------- guardrails
async def input_guardrail_node(state: AgentState) -> dict:
    try:
        d = await evaluate_input(state["user_input"])
    except ConfigError as exc:
        return {"intent": "UNRELATED", "final_response": str(exc), "plan": []}
    except Exception as exc:  # noqa: BLE001
        LOG.exception("input guardrail failed")
        return {"intent": "UNRELATED", "final_response": friendly_error(exc), "plan": []}
    upd: dict = {"intent": d.intent, "greeting_prefix": d.greeting_prefix, "plan": d.plan, "step": 0,
                 "user_input": d.cleaned_input or state["user_input"]}
    if d.direct_response:
        upd["final_response"] = d.direct_response
    return upd


async def orchestrator_node(state: AgentState) -> dict:
    plan = orchestrator.normalize_plan(state.get("plan") or [])
    LOG.info("plan: %s", [p["agent"] for p in plan])
    return {"plan": plan, "step": 0}


# ---------------------------------------------------------------- agents
def _apply(state: AgentState, r: AgentResult, extra: dict | None = None) -> dict:
    outputs = list(state.get("agent_outputs") or [])
    outputs.append({"agent": r.agent, "text": r.text, "ok": r.ok})
    upd: dict = {"agent_outputs": outputs, **(extra or {})}
    if r.agent in ("github", "calendar", "email"):
        upd[f"{r.agent}_result"] = {"ok": r.ok, "text": r.text, "pending": [asdict(p) for p in r.pending]}
    if r.pending:
        upd["pending_action"] = {"agent": r.agent, "actions": [asdict(p) for p in r.pending], "summary": r.text}
        upd["requires_confirmation"] = True
    return upd


def _current_task(state: AgentState) -> str:
    return state["plan"][state["step"]]["task"] or state["user_input"]


async def rag_node(state: AgentState) -> dict:
    out = await rag_agent.answer_pdf_question(_current_task(state), orchestrator.history_text(state))
    extra = {
        "retrieved_context": [h.text for h in out.chunks],
        "retrieval_metadata": [
            {"document_name": h.document_name, "page_number": h.page_number, "chunk_id": h.chunk_id, "score": round(h.score, 3)}
            for h in out.chunks
        ],
        "grounding_result": out.grounding,
    }
    return _apply(state, AgentResult("rag", out.text, out.ok), extra)


async def _mcp_node(state: AgentState, runner) -> dict:
    try:
        r = await runner(_current_task(state), orchestrator.context_from_outputs(state), orchestrator.history_text(state))
    except ConfigError as exc:
        r = AgentResult(runner.__module__.split(".")[-1].replace("_agent", ""), str(exc), ok=False)
    except Exception as exc:  # noqa: BLE001
        LOG.exception("agent crashed")
        r = AgentResult(runner.__module__.split(".")[-1].replace("_agent", ""), friendly_error(exc), ok=False)
    return _apply(state, r)


async def github_node(state: AgentState) -> dict:
    return await _mcp_node(state, github_agent.run)


async def calendar_node(state: AgentState) -> dict:
    return await _mcp_node(state, calendar_agent.run)


async def email_node(state: AgentState) -> dict:
    return await _mcp_node(state, email_agent.run)


# ---------------------------------------------------------------- human-in-the-loop
async def execute_pending(agent: str, actions: list[dict]) -> tuple[list[str], bool]:
    conn = mcp.get_connection(agent)
    lines, ok = [], True
    for a in actions:
        try:
            out = await asyncio.wait_for(conn.call(a["tool"], a["args"]), timeout=90)
            snippet = redact_secrets(out).strip().replace("\n", " ")[:300]
            lines.append(f"✅ {a['tool']} completed." + (f" {snippet}" if snippet else ""))
        except Exception as exc:  # noqa: BLE001
            ok = False
            LOG.warning("pending action %s failed: %s", a["tool"], type(exc).__name__)
            lines.append(f"⚠️ {a['tool']} failed: {friendly_error(exc)}")
    return lines, ok


async def confirm_node(state: AgentState) -> dict:
    step = state.get("step", 0)
    pending = state.get("pending_action")
    if not pending:
        return {"step": step + 1, "requires_confirmation": False}

    # Pauses the graph; resumes with the user's answer (Command(resume=...)).
    answer = interrupt({"agent": pending["agent"], "prompt": format_pending(pending)})
    approved = answer is True or str(answer).strip().lower() in YES

    outputs = list(state.get("agent_outputs") or [])
    idx = next((i for i in range(len(outputs) - 1, -1, -1) if outputs[i]["agent"] == pending["agent"]), None)
    if approved:
        lines, ok = await execute_pending(pending["agent"], pending["actions"])
        text, next_step = "\n".join(lines), step + 1
    else:
        text = "Okay, I cancelled that - nothing was changed."
        ok = False
        next_step = len(state.get("plan") or [])  # skip dependent follow-up steps
        if next_step > step + 1:
            text += " I also skipped the remaining steps that depended on it."
    if idx is not None:
        outputs[idx] = {"agent": pending["agent"], "text": text, "ok": ok}
    return {"agent_outputs": outputs, "pending_action": None, "requires_confirmation": False, "step": next_step}


# ---------------------------------------------------------------- finalize + output guardrail
async def finalize_node(state: AgentState) -> dict:
    if state.get("final_response"):
        return {}
    return {"final_response": orchestrator.compose_final(state)}


async def output_guardrail_node(state: AgentState) -> dict:
    safe = apply_output_guardrail(state.get("final_response") or "", state.get("grounding_result"))
    return {"final_response": safe, "messages": [orchestrator.as_ai_message(safe)]}
