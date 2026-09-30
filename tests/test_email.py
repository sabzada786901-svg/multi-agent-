from langchain_core.messages import AIMessage

from app.agents import base
from app.agents.base import format_pending, run_tool_agent
from app.agents.email_agent import SAFE
from app.mcp.email import _filter
from app.utils.security import invalid_addresses, is_valid_email
from tests import tools_fakes as tf
from types import SimpleNamespace


def test_email_tool_filter_fails_closed():
    tools = [SimpleNamespace(name="send_email"), SimpleNamespace(name="delete_email"), SimpleNamespace(name="new_tool")]
    assert _filter(tools) == tools[:1]
    assert _filter(tools[1:]) == []


def test_email_validation():
    assert is_valid_email("john@example.com") and not is_valid_email("john@")
    assert invalid_addresses({"to": ["john@example.com", "not-an-email"]}) == ["not-an-email"]


async def test_send_requires_confirmation_and_shows_preview(monkeypatch):
    tf.EXECUTED.clear()
    args = {"to": ["john@example.com"], "subject": "Project Meeting", "body": "Hi John, see you at 3."}
    chain = tf.FakeChain([tf.call("send_email", args), AIMessage(content="I prepared this email.")])
    monkeypatch.setattr(base.llm, "llm_chain", lambda transform=None: chain)
    res = await run_tool_agent(agent="email", system_prompt="s", task="t", context="", history="",
                               tools=[tf.send_email], safe_tools=SAFE)
    assert tf.EXECUTED == []                                  # NOT sent automatically
    preview = format_pending({"actions": [{"tool": p.tool, "args": p.args} for p in res.pending]})
    assert "To: john@example.com" in preview and "Subject:\nProject Meeting" in preview
    assert preview.endswith("Would you like me to proceed? (yes/no)")


async def test_invalid_address_is_never_queued(monkeypatch):
    tf.EXECUTED.clear()
    chain = tf.FakeChain([tf.call("send_email", {"to": ["john"], "subject": "s", "body": "b"}), AIMessage(content="Need a valid address.")])
    monkeypatch.setattr(base.llm, "llm_chain", lambda transform=None: chain)
    res = await run_tool_agent(agent="email", system_prompt="s", task="t", context="", history="",
                               tools=[tf.send_email], safe_tools=SAFE)
    assert res.pending == [] and tf.EXECUTED == []


async def test_search_and_draft_run_without_confirmation(monkeypatch):
    tf.EXECUTED.clear()
    chain = tf.FakeChain([
        tf.call("search_emails", {"query": "invoice"}),
        tf.call("draft_email", {"to": ["sarah@example.com"], "subject": "Hi", "body": "Hello"}, "2"),
        AIMessage(content="Draft created."),
    ])
    monkeypatch.setattr(base.llm, "llm_chain", lambda transform=None: chain)
    res = await run_tool_agent(agent="email", system_prompt="s", task="t", context="", history="",
                               tools=[tf.search_emails, tf.draft_email], safe_tools=SAFE)
    assert tf.EXECUTED == ["search_emails", "draft_email"] and res.pending == []
