"""Qdrant Cloud (free tier) wrapper. Also supports VECTOR_DB_URL=:memory: for offline tests."""
from __future__ import annotations

from dataclasses import dataclass

from app.config import ConfigError, get_settings
from app.rag.chunker import Chunk
from app.utils.logging import get_logger

LOG = get_logger("vectorstore")
_client = None


class VectorStoreError(RuntimeError):
    """User-safe vector DB problem."""


class NoIndexError(VectorStoreError):
    """Collection does not exist / is empty."""


@dataclass
class Hit:
    text: str
    document_name: str
    page_number: int
    chunk_id: str
    score: float


def set_client(client) -> None:  # used by tests
    global _client
    _client = client


def get_client():
    global _client
    if _client is not None:
        return _client
    from qdrant_client import QdrantClient

    s = get_settings()
    if not s.vector_db_url:
        raise ConfigError("VECTOR_DB_URL is missing. Create a free Qdrant Cloud cluster and add its URL + API key to .env.")
    if s.vector_db_url == ":memory:":
        _client = QdrantClient(":memory:")
    else:
        _client = QdrantClient(url=s.vector_db_url, api_key=s.vector_db_api_key or None, timeout=30)
    return _client


def _wrap(exc: Exception) -> VectorStoreError:
    LOG.warning("vector store error: %s: %s", type(exc).__name__, exc)
    msg = str(exc).lower()
    if "401" in msg or "403" in msg or "forbidden" in msg or "unauthorized" in msg:
        return VectorStoreError("The vector database rejected the API key. Check VECTOR_DB_URL / VECTOR_DB_API_KEY.")
    return VectorStoreError(
        "The vector database is unreachable. Free Qdrant clusters are suspended after a week of inactivity - "
        "check the cluster in the Qdrant Cloud console."
    )


def ensure_collection(dim: int) -> None:
    from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

    name = get_settings().collection_name
    try:
        c = get_client()
        if not c.collection_exists(name):
            c.create_collection(name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        c.create_payload_index(name, field_name="document_name", field_schema=PayloadSchemaType.KEYWORD)
    except ConfigError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc) from exc


def delete_document(document_name: str) -> None:
    from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

    try:
        get_client().delete(
            get_settings().collection_name,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="document_name", match=MatchValue(value=document_name))])
            ),
        )
    except ConfigError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc) from exc


def upsert_chunks(chunks: list[Chunk], vectors: list[list[float]], batch: int = 64) -> None:
    from qdrant_client.models import PointStruct

    try:
        c, name = get_client(), get_settings().collection_name
        for i in range(0, len(chunks), batch):
            pts = [
                PointStruct(
                    id=ch.point_id,
                    vector=vec,
                    payload={
                        "text": ch.text,
                        "document_name": ch.document_name,
                        "page_number": ch.page_number,
                        "chunk_id": ch.chunk_id,
                    },
                )
                for ch, vec in zip(chunks[i : i + batch], vectors[i : i + batch])
            ]
            c.upsert(name, points=pts, wait=True)
    except ConfigError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc) from exc


def search(vector: list[float], k: int, min_score: float) -> list[Hit]:
    try:
        c, name = get_client(), get_settings().collection_name
        if not c.collection_exists(name):
            raise NoIndexError("no collection")
        res = c.query_points(name, query=vector, limit=k, with_payload=True, score_threshold=min_score)
        return [
            Hit(p.payload["text"], p.payload["document_name"], int(p.payload["page_number"]), p.payload["chunk_id"], float(p.score))
            for p in res.points
        ]
    except (ConfigError, NoIndexError):
        raise
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc) from exc
