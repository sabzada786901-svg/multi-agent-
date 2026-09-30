"""Windows-friendly CLI.   python -m app.main"""
from __future__ import annotations

import asyncio
import sys
import uuid

from langgraph.types import Command

from app import mcp
from app.config import ConfigError, get_settings
from app.graph.graph import build_graph
from app.state import initial_turn_state
from app.utils.logging import get_logger, setup_logging
from app.utils.security import friendly_error, redact_secrets

LOG = get_logger("main")
BANNER = "=" * 50 + "\n      LANGGRAPH MULTI-AGENT AI ASSISTANT\n" + "=" * 50
YES, NO = {"y", "yes", "confirm", "ok", "okay", "sure"}, {"n", "no", "cancel", "stop"}


async def pending_interrupt(graph, config) -> dict | None:
    state = await graph.aget_state(config)
    for task in state.tasks:
        for it in getattr(task, "interrupts", ()) or ():
            return it.value
    return None


async def ask(prompt: str) -> str:
    return (await asyncio.to_thread(input, prompt)).strip()


async def chat() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    print(BANNER + "\nType 'exit' to quit.\n")
    while True:
        try:
            text = await ask("You: ")
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"exit", "quit"}:
            break
        if not text:
            continue
        try:
            await graph.ainvoke(initial_turn_state(text), config)
            while (payload := await pending_interrupt(graph, config)) is not None:
                print("\nAssistant:\n" + payload["prompt"])
                ans = ""
                while ans.lower() not in YES | NO:
                    ans = await ask("\nYou (yes/no): ")
                await graph.ainvoke(Command(resume=ans.lower() in YES), config)
            final = (await graph.aget_state(config)).values.get("final_response") or ""
            print("\nAssistant:\n" + redact_secrets(final) + "\n")
        except Exception as exc:  # noqa: BLE001
            LOG.exception("turn failed")
            print("\nAssistant:\n" + friendly_error(exc) + "\n")


async def amain() -> int:
    setup_logging()
    try:
        get_settings()
    except ConfigError as exc:
        print(f"Configuration error: {exc}")
        return 2
    try:
        await chat()
    finally:
        await mcp.close_all()
    print("Goodbye!")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(amain()))
