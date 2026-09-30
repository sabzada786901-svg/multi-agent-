"""Scripted fake chain + fake MCP tools shared by the GitHub/Calendar/Email tests."""
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

EXECUTED: list[str] = []


def call(name, args, i="1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": i, "type": "tool_call"}])


class FakeChain:
    def __init__(self, script):
        self.script = list(script)
        self.seen = []

    async def ainvoke(self, msgs):
        self.seen.append(list(msgs))
        return self.script.pop(0)


@tool
def list_events(calendar_id: str = "primary") -> str:
    """List events."""
    EXECUTED.append("list_events")
    return "10:00 Standup"


@tool
def create_event(summary: str, start: str) -> str:
    """Create an event."""
    EXECUTED.append("create_event")
    return "created"


@tool
def delete_event(event_id: str) -> str:
    """Delete an event."""
    EXECUTED.append("delete_event")
    return "deleted"


@tool
def search_emails(query: str) -> str:
    """Search emails."""
    EXECUTED.append("search_emails")
    return "1 result"


@tool
def draft_email(to: list[str], subject: str, body: str) -> str:
    """Create a draft."""
    EXECUTED.append("draft_email")
    return "draft saved"


@tool
def send_email(to: list[str], subject: str, body: str) -> str:
    """Send an email."""
    EXECUTED.append("send_email")
    return "sent"


@tool
def create_issue(repo: str, title: str) -> str:
    """Create an issue."""
    EXECUTED.append("create_issue")
    return "issue #1"


@tool
def list_issues(repo: str) -> str:
    """List issues."""
    EXECUTED.append("list_issues")
    return "#1 Login bug"
