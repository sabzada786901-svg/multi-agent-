"""Central configuration. Secrets only ever come from environment variables / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DEFAULT_OPENROUTER_MODEL = "openai/gpt-oss-120b:free"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
PROVIDERS = ("openrouter", "groq")


class ConfigError(RuntimeError):
    """A user-fixable configuration problem (message is safe to show)."""


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _num(name: str, default, cast):
    raw = _env(name)
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number (got {raw!r}).") from exc


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    openrouter_api_key: str
    openrouter_model: str
    groq_api_key: str
    groq_model: str
    vector_db_url: str
    vector_db_api_key: str
    collection_name: str
    embedding_model: str
    top_k: int
    min_retrieval_score: float
    github_token: str
    github_mcp_url: str
    google_oauth_credentials: str
    calendar_mcp_args: str
    email_mcp_args: str
    timezone: str
    pdf_dir: Path

    def has_key(self, provider: str) -> bool:
        return bool({"openrouter": self.openrouter_api_key, "groq": self.groq_api_key}.get(provider))

    def secret_values(self) -> list[str]:
        vals = [self.openrouter_api_key, self.groq_api_key, self.vector_db_api_key, self.github_token]
        return [v for v in vals if v and len(v) >= 8]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    provider = _env("LLM_PROVIDER", "openrouter").lower()
    if provider not in PROVIDERS:
        raise ConfigError(f"LLM_PROVIDER must be one of {PROVIDERS} (got {provider!r}).")

    tz = _env("TIMEZONE", "Asia/Karachi")
    try:
        from zoneinfo import ZoneInfo

        ZoneInfo(tz)
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(
            f"Invalid TIMEZONE {tz!r}. On Windows run: pip install tzdata"
        ) from exc

    return Settings(
        llm_provider=provider,
        openrouter_api_key=_env("OPENROUTER_API_KEY"),
        openrouter_model=_env("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL),
        groq_api_key=_env("GROQ_API_KEY"),
        groq_model=_env("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        vector_db_url=_env("VECTOR_DB_URL"),
        vector_db_api_key=_env("VECTOR_DB_API_KEY"),
        collection_name=_env("VECTOR_COLLECTION", "pdf_chunks"),
        embedding_model=_env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        top_k=_num("TOP_K", 5, int),
        min_retrieval_score=_num("MIN_RETRIEVAL_SCORE", 0.25, float),
        github_token=_env("GITHUB_TOKEN"),
        github_mcp_url=_env("GITHUB_MCP_URL", "https://api.githubcopilot.com/mcp/"),
        google_oauth_credentials=_env("GOOGLE_OAUTH_CREDENTIALS"),
        calendar_mcp_args=_env("CALENDAR_MCP_ARGS", "-y @cocal/google-calendar-mcp"),
        email_mcp_args=_env("EMAIL_MCP_ARGS", "-y @gongrzhe/server-gmail-autoauth-mcp"),
        timezone=tz,
        pdf_dir=ROOT / "data" / "pdfs",
    )
