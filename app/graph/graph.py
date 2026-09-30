"""Graph assembly.

START -> input_guardrail -> (direct reply | orchestrator)
orchestrator -> [rag|github|calendar|email]_agent -> confirm -> next step ... -> finalize -> output_guardrail -> END
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph import nodes
from app.graph.routing import NODE_FOR_AGENT, route_after_input, route_next_step
from app.state import AgentState


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("input_guardrail", nodes.input_guardrail_node)
    g.add_node("orchestrator", nodes.orchestrator_node)
    g.add_node("rag_agent", nodes.rag_node)
    g.add_node("github_agent", nodes.github_node)
    g.add_node("calendar_agent", nodes.calendar_node)
    g.add_node("email_agent", nodes.email_node)
    g.add_node("confirm", nodes.confirm_node)
    g.add_node("finalize", nodes.finalize_node)
    g.add_node("output_guardrail", nodes.output_guardrail_node)

    g.add_edge(START, "input_guardrail")
    g.add_conditional_edges("input_guardrail", route_after_input, {"orchestrator": "orchestrator", "finalize": "finalize"})

    targets = {**{n: n for n in NODE_FOR_AGENT.values()}, "finalize": "finalize"}
    g.add_conditional_edges("orchestrator", route_next_step, targets)
    for node_name in NODE_FOR_AGENT.values():
        g.add_edge(node_name, "confirm")
    g.add_conditional_edges("confirm", route_next_step, targets)

    g.add_edge("finalize", "output_guardrail")
    g.add_edge("output_guardrail", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())
