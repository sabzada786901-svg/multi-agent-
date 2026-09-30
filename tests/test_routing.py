from app.agents.orchestrator import compose_final, normalize_plan
from app.graph.routing import route_after_input, route_next_step


def test_route_after_input():
    assert route_after_input({"final_response": "Hello!", "plan": []}) == "finalize"
    assert route_after_input({"plan": [{"agent": "rag", "task": "q"}]}) == "orchestrator"


def test_route_next_step_walks_the_plan():
    plan = [{"agent": "calendar", "task": "a"}, {"agent": "email", "task": "b"}]
    assert route_next_step({"plan": plan, "step": 0}) == "calendar_agent"
    assert route_next_step({"plan": plan, "step": 1}) == "email_agent"
    assert route_next_step({"plan": plan, "step": 2}) == "finalize"


def test_normalize_plan_orders_and_dedupes():
    plan = [{"agent": "email", "task": "e"}, {"agent": "rag", "task": "r"}, {"agent": "email", "task": "dup"}, {"agent": "bogus", "task": "x"}]
    assert [p["agent"] for p in normalize_plan(plan)] == ["rag", "email"]


def test_compose_final_prepends_greeting():
    state = {"greeting_prefix": "Hi! 👋", "agent_outputs": [{"agent": "rag", "text": "According to the uploaded PDF, ...", "ok": True}]}
    assert compose_final(state).startswith("Hi! 👋 According to the uploaded PDF")
