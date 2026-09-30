from langchain_core.messages import AIMessage
from types import SimpleNamespace

from app.agents import base
from app.agents.base import is_mutating, run_tool_agent
from app.mcp.github import _filter
from tests import tools_fakes as tf


def test_github_tool_filter_fails_closed():
    tools = [SimpleNamespace(name="get_me"), SimpleNamespace(name="delete_repository"), SimpleNamespace(name="unknown_tool")]
    assert _filter(tools) == tools[:1]
    assert _filter(tools[1:]) == []


def test_github_live_write_tools_require_confirmation():
    assert is_mutating("create_pull_request")
    assert is_mutating("update_pull_request")
    assert is_mutating("update_pull_request_branch")
    assert is_mutating("add_reply_to_pull_request_comment")


async def test_github_reads_run_and_issue_creation_is_gated(monkeypatch):
    tf.EXECUTED.clear()
    chain = tf.FakeChain([
        tf.call("list_issues", {"repo": "me/app"}),
        tf.call("create_issue", {"repo": "me/app", "title": "Fix login"}, "2"),
        AIMessage(content="I'll open the issue."),
    ])
    monkeypatch.setattr(base.llm, "llm_chain", lambda transform=None: chain)
    res = await run_tool_agent(agent="github", system_prompt="s", task="t", context="", history="",
                               tools=[tf.list_issues, tf.create_issue])
    assert tf.EXECUTED == ["list_issues"] and [p.tool for p in res.pending] == ["create_issue"]


async def test_tool_output_is_untrusted_and_errors_are_friendly(monkeypatch):
    from langchain_core.tools import tool

    @tool
    def get_file_contents(path: str) -> str:
        """Read a file."""
        raise RuntimeError("403 Forbidden: token lacks scope ghp_" + "a" * 30)

    chain = tf.FakeChain([tf.call("get_file_contents", {"path": "x"}), AIMessage(content="No access.")])
    monkeypatch.setattr(base.llm, "llm_chain", lambda transform=None: chain)
    res = await run_tool_agent(agent="github", system_prompt="s", task="t", context="", history="", tools=[get_file_contents])
    tool_msg = chain.seen[1][-1].content
    assert "Permission denied" in tool_msg and "ghp_" not in tool_msg and res.text == "No access."
