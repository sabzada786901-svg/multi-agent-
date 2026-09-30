"""GitHub MCP: GitHub's official remote MCP server, authenticated with a fine-grained PAT."""
from __future__ import annotations

from app.config import ConfigError, get_settings
from app.mcp.base import MCPConnection

_ALLOW = ("repositor", "file_contents", "search_code", "issue", "pull_request", "commit", "branch", "get_me", "tree", "list_repos")
_DENY = ("workflow", "action", "secret", "security", "dependabot", "gist", "notification", "project", "release", "star",
         "discussion", "team", "org", "copilot", "job", "artifact", "fork", "merge", "delete", "push", "review",
         "label", "sub_issue", "branch_create", "create_repository", "create_branch")
_ISSUE_WRITES = ("create_issue", "add_issue_comment", "issue_write", "update_issue")


def _config() -> dict:
    s = get_settings()
    if not s.github_token:
        raise ConfigError("GITHUB_TOKEN is missing. Create a fine-grained personal access token and add it to .env.")
    return {"transport": "streamable_http", "url": s.github_mcp_url, "headers": {"Authorization": f"Bearer {s.github_token}"}}


def _filter(tools: list) -> list:
    keep = []
    for t in tools:
        n = t.name.lower()
        if n in _ISSUE_WRITES or (any(a in n for a in _ALLOW) and not any(d in n for d in _DENY)):
            keep.append(t)
    return keep[:30]  # fail closed if the server's tool names change


connection = MCPConnection("github", _config, _filter)
