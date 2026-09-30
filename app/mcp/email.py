"""Email MCP: Gmail server @gongrzhe/server-gmail-autoauth-mcp (stdio via npx). Credentials live in ~/.gmail-mcp/."""
from __future__ import annotations

from pathlib import Path

from app.config import ConfigError, get_settings
from app.mcp.base import MCPConnection, npx_stdio_config

_ALLOW = {"send_email", "draft_email", "read_email", "search_emails", "list_email_labels"}


def _config() -> dict:
    if not (Path.home() / ".gmail-mcp" / "credentials.json").is_file():
        raise ConfigError(
            "Gmail is not authorised yet. Put gcp-oauth.keys.json in %USERPROFILE%\\.gmail-mcp and run: "
            "npx @gongrzhe/server-gmail-autoauth-mcp auth   (see README)"
        )
    return npx_stdio_config(get_settings().email_mcp_args)


def _filter(tools: list) -> list:
    return [t for t in tools if t.name in _ALLOW]


connection = MCPConnection("email", _config, _filter)
