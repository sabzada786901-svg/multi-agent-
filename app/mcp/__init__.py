"""MCP connection registry."""
from __future__ import annotations

import importlib

_MODULES = {"github": "app.mcp.github", "calendar": "app.mcp.calendar", "email": "app.mcp.email"}


def get_connection(agent: str):
    return importlib.import_module(_MODULES[agent]).connection


async def close_all() -> None:
    for name, mod in _MODULES.items():
        m = importlib.import_module(mod)
        await m.connection.close()
