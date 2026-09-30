"""LLM access: OpenRouter (primary) + Groq (fallback), both without any OpenAI key."""
from __future__ import annotations

import json
import re
from typing import Callable, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel

from app.config import ConfigError, get_settings
from app.utils.logging import get_logger

LOG = get_logger("llm")
T = TypeVar("T", bound=BaseModel)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _build(provider: str) -> BaseChatModel:
    s = get_settings()
    if provider == "openrouter":
        from langchain_openai import ChatOpenAI  # OpenAI-compatible client pointed at OpenRouter

        return ChatOpenAI(
            model=s.openrouter_model,
            api_key=s.openrouter_api_key,
            base_url=OPENROUTER_BASE_URL,
            temperature=0,
            timeout=60,
            max_retries=2,
            default_headers={"X-Title": "LangGraph Multi-Agent Assistant"},
        )
    from langchain_groq import ChatGroq

    return ChatGroq(model=s.groq_model, api_key=s.groq_api_key, temperature=0, timeout=60, max_retries=2)


def get_chat_models() -> list[BaseChatModel]:
    """Configured provider first, then the other one if its key exists (fallback)."""
    s = get_settings()
    order = [s.llm_provider] + [p for p in ("openrouter", "groq") if p != s.llm_provider]
    models = [_build(p) for p in order if s.has_key(p)]
    if not models:
        raise ConfigError(
            "No LLM API key found. Set OPENROUTER_API_KEY and/or GROQ_API_KEY in your .env file."
        )
    return models


def llm_chain(transform: Callable[[BaseChatModel], object] = lambda m: m):
    """Apply `transform` (e.g. bind_tools) to each model and chain them with fallbacks."""
    models = get_chat_models()
    chain = transform(models[0])
    if len(models) > 1:
        chain = chain.with_fallbacks([transform(m) for m in models[1:]])
    return chain


def text_of(msg) -> str:
    c = getattr(msg, "content", msg)
    if isinstance(c, list):
        c = "".join(
            b if isinstance(b, str) else b.get("text", "")
            for b in c
            if isinstance(b, str) or (isinstance(b, dict) and b.get("type", "text") == "text")
        )
    c = str(c)
    return re.sub(r"<think>.*?</think>", "", c, flags=re.S).strip()


async def agenerate(messages: list[BaseMessage]) -> str:
    return text_of(await llm_chain().ainvoke(messages))


def _extract_json(raw: str) -> str:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in model output")
    return raw[start : end + 1]


async def astructured(schema: type[T], messages: list[BaseMessage]) -> T:
    """Structured output; falls back to 'reply with JSON' for models without tool support."""
    try:
        chain = llm_chain(lambda m: m.with_structured_output(schema, method="function_calling"))
        result = await chain.ainvoke(messages)
        if isinstance(result, schema):
            return result
        return schema.model_validate(result)
    except ConfigError:
        raise
    except Exception as exc:  # noqa: BLE001
        LOG.info("structured output failed (%s); falling back to JSON prompt", type(exc).__name__)
    hint = HumanMessage(
        "Reply with ONLY a JSON object (no markdown) matching this JSON schema:\n"
        + json.dumps(schema.model_json_schema())
    )
    raw = await agenerate([*messages, hint])
    return schema.model_validate_json(_extract_json(raw))
