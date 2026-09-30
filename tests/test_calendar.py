from datetime import datetime

from langchain_core.messages import AIMessage
from zoneinfo import ZoneInfo

from app.agents import base
from app.agents.base import format_pending, is_mutating, run_tool_agent
from app.agents.calendar_agent import build_system_prompt
from app.mcp.calendar import _filter
from tests import tools_fakes as tf
from types import SimpleNamespace


def test_timezone_is_karachi_not_utc():
    p = build_system_prompt(datetime(2026, 9, 28, 10, 0, tzinfo=ZoneInfo("Asia/Karachi")))
    assert "Asia/Karachi" in p and "+05:00" in p


def test_calendar_write_tools_need_confirmation():
    assert is_mutating("create-event") and is_mutating("update-event") and is_mutating("delete-event")
    assert not is_mutating("list-events") and not is_mutating("get-freebusy")


def test_calendar_tool_filter_fails_closed():
    tools = [SimpleNamespace(name="list-events"), SimpleNamespace(name="manage-accounts"), SimpleNamespace(name="new-tool")]
    assert _filter(tools) == tools[:1]
    assert _filter(tools[1:]) == []


async def test_read_runs_create_is_queued_not_executed(monkeypatch):
    tf.EXECUTED.clear()
    chain = tf.FakeChain([
        tf.call("list_events", {}),
        tf.call("create_event", {"summary": "Project Discussion", "start": "2026-09-29T15:00:00+05:00"}, "2"),
        AIMessage(content="I'll create it tomorrow at 3 PM."),
    ])
    monkeypatch.setattr(base.llm, "llm_chain", lambda transform=None: chain)
    res = await run_tool_agent(agent="calendar", system_prompt="s", task="t", context="", history="",
                               tools=[tf.list_events, tf.create_event])
    assert tf.EXECUTED == ["list_events"]              # read executed, create NOT executed
    assert [p.tool for p in res.pending] == ["create_event"]
    text = format_pending({"actions": [{"tool": p.tool, "args": p.args} for p in res.pending]})
    assert "Would you like me to proceed?" in text and "Project Discussion" in text


async def test_delete_is_queued(monkeypatch):
    tf.EXECUTED.clear()
    chain = tf.FakeChain([tf.call("delete_event", {"event_id": "abc"}), AIMessage(content="I'll cancel it.")])
    monkeypatch.setattr(base.llm, "llm_chain", lambda transform=None: chain)
    res = await run_tool_agent(agent="calendar", system_prompt="s", task="t", context="", history="", tools=[tf.delete_event])
    assert tf.EXECUTED == [] and res.pending[0].tool == "delete_event"
