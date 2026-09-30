"""Google Calendar MCP: community server @cocal/google-calendar-mcp (stdio via npx, OAuth desktop client)."""
from __future__ import annotations

from pathlib import Path

from app.config import ConfigError, get_settings
from app.mcp.base import MCPConnection, npx_stdio_config

ENABLED = "list-calendars,list-events,search-events,get-event,create-event,update-event,delete-event,get-freebusy,get-current-time"
_ALLOWED_TOOLS = frozenset(ENABLED.split(","))


def _config() -> dict:
    s = get_settings()
    if not s.google_oauth_credentials or not Path(s.google_oauth_credentials).is_file():
        raise ConfigError(
            "GOOGLE_OAUTH_CREDENTIALS must point to your Google OAuth client JSON file (see README: Google Calendar MCP setup)."
        )
    return npx_stdio_config(
        s.calendar_mcp_args,
        {"GOOGLE_OAUTH_CREDENTIALS": s.google_oauth_credentials, "ENABLED_TOOLS": ENABLED},
    )


def _filter(tools: list) -> list:
    return [t for t in tools if t.name in _ALLOWED_TOOLS]


connection = MCPConnection("calendar", _config, _filter)
