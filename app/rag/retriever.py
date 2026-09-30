"""Retriever: embed question locally, search Qdrant, return scored chunks."""
from __future__ import annotations

import asyncio

from app.config import get_settings
from app.rag import embeddings, vectorstore
from app.rag.vectorstore import Hit


async def retrieve(question: str) -> list[Hit]:
    s = get_settings()

    def _work() -> list[Hit]:
        vec = embeddings.embed_query(question)
        return vectorstore.search(vec, s.top_k, s.min_retrieval_score)

    return await asyncio.to_thread(_work)
