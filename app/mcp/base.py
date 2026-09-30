"""Long-lived MCP connection owned by ONE background task (anyio cancel-scopes must exit in the task that entered them)."""
from __future__ import annotations

import asyncio
import shlex
import shutil
from typing import Callable

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from app.agents.base import stringify_tool_output
from app.config import ConfigError
from app.utils.logging import get_logger
from app.utils.security import friendly_error

LOG = get_logger("mcp")


class MCPUnavailable(RuntimeError):
    """MCP server could not be started/reached (message is user-safe)."""


def npx_stdio_config(args: str, env: dict | None = None) -> dict:
    npx = shutil.which("npx")  # resolves npx.cmd on Windows
    if not npx:
        raise ConfigError("Node.js is required for this MCP server (npx not found). Install it from https://nodejs.org")
    cfg = {"transport": "stdio", "command": npx, "args": shlex.split(args, posix=False)}
    if env:
        cfg["env"] = env
    return cfg


class MCPConnection:
    def __init__(self, name: str, config_factory: Callable[[], dict], tool_filter: Callable[[list], list] | None = None):
        self.name = name
        self._config_factory = config_factory
        self._filter = tool_filter
        self.tools: list = []
        self._task: asyncio.Task | None = None
        self._ready: asyncio.Event | None = None
        self._stop: asyncio.Event | None = None
        self._error: Exception | None = None
        self._lock = asyncio.Lock()

    async def _run(self) -> None:
        try:
            client = MultiServerMCPClient({self.name: self._config_factory()})
            async with client.session(self.name) as session:
                tools = await load_mcp_tools(session)
                self.tools = self._filter(tools) if self._filter else tools
                LOG.info("MCP %s ready with %d tools", self.name, len(self.tools))
                self._ready.set()
                await self._stop.wait()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001  (includes ExceptionGroup from anyio)
            LOG.warning("MCP %s failed: %s: %s", self.name, type(exc).__name__, exc)
            self._error = exc
        finally:
            if self._ready is not None:
                self._ready.set()

    async def start(self) -> None:
        async with self._lock:
            if self._task is None or self._task.done():
                self._error, self.tools = None, []
                self._ready, self._stop = asyncio.Event(), asyncio.Event()
                self._task = asyncio.create_task(self._run())
            try:
                await asyncio.wait_for(self._ready.wait(), timeout=120)
            except asyncio.TimeoutError as exc:
                raise MCPUnavailable(f"{self.name} MCP server did not start within 120s.") from exc
            if self._error is not None:
                err, self._task = self._error, None
                inner = getattr(err, "exceptions", [err])[0] if hasattr(err, "exceptions") else err
                raise MCPUnavailable(friendly_error(inner)) from err
            if not self.tools:
                raise MCPUnavailable(f"{self.name} MCP server exposes no usable tools.")

    async def call(self, tool_name: str, args: dict) -> str:
        tool = next((t for t in self.tools if t.name == tool_name), None)
        if tool is None:
            raise MCPUnavailable(f"Tool {tool_name!r} is not available.")
        return stringify_tool_output(await tool.ainvoke(args))

    async def close(self) -> None:
        if self._task and not self._task.done():
            self._stop.set()
            try:
                await asyncio.wait_for(self._task, timeout=10)
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                self._task.cancel()
        self._task = None
