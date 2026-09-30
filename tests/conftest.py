"""Offline test fixtures: fake embeddings, in-memory Qdrant, scripted fake LLM. No network / API keys."""
import os
import re
import zlib

os.environ["OPENROUTER_API_KEY"] = "test-openrouter-key-1234567890"
os.environ["GROQ_API_KEY"] = ""
os.environ["MIN_RETRIEVAL_SCORE"] = "0.15"
os.environ["TIMEZONE"] = "Asia/Karachi"
os.environ["VECTOR_DB_URL"] = ":memory:"

import pytest  # noqa: E402

from app.config import get_settings  # noqa: E402

get_settings.cache_clear()

STOP = set("what does the pdf say about is are a an of by how do in on to for with and or it this that".split())
DIM = 512


def _vec(text: str) -> list[float]:
    v = [0.0] * DIM
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w not in STOP:
            v[zlib.crc32(w.encode()) % DIM] += 1.0
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / n for x in v]


class FakeLLM:
    def __init__(self):
        from app.guardrails.grounding import GroundingCheck

        self.answer = "According to the uploaded PDF, authentication uses OAuth 2.0 tokens. [S1]"
        self.grounding = GroundingCheck(grounded=True, supported=True, reason="ok")
        self.classification = None
        self.generate_calls = []
        self.structured_calls = []

    async def agenerate(self, messages):
        self.generate_calls.append(messages)
        return self.answer

    async def astructured(self, schema, messages):
        self.structured_calls.append(schema)
        if schema.__name__ == "GroundingCheck":
            if isinstance(self.grounding, Exception):
                raise self.grounding
            return self.grounding
        if self.classification is None:
            raise RuntimeError("no classification scripted")
        return self.classification


@pytest.fixture
def fake_llm(monkeypatch):
    f = FakeLLM()
    from app import llm

    monkeypatch.setattr(llm, "agenerate", f.agenerate)
    monkeypatch.setattr(llm, "astructured", f.astructured)
    return f


@pytest.fixture
def store(monkeypatch):
    """In-memory Qdrant with one small 'handbook.pdf' indexed via the real vectorstore code."""
    from qdrant_client import QdrantClient

    from app.rag import chunker, embeddings, vectorstore

    monkeypatch.setattr(embeddings, "embed_texts", lambda texts: [_vec(t) for t in texts])
    monkeypatch.setattr(embeddings, "embed_query", lambda t: _vec(t))
    vectorstore.set_client(QdrantClient(":memory:"))

    def index(pages):
        from app.rag.loader import PageText

        chunks = chunker.chunk_pages([PageText("handbook.pdf", n, t) for n, t in pages])
        vectors = [_vec(c.text) for c in chunks]
        vectorstore.ensure_collection(DIM)
        vectorstore.upsert_chunks(chunks, vectors)

    yield index
    vectorstore.set_client(None)
