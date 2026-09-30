"""Page-aware chunking. Each chunk keeps document_name / page_number / chunk_id."""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.rag.loader import PageText

_NAMESPACE = uuid.UUID("6f1c5f0e-6d0c-4c55-9a55-0f7c2b0e9a11")


@dataclass
class Chunk:
    chunk_id: str
    document_name: str
    page_number: int
    chunk_index: int
    text: str

    @property
    def point_id(self) -> str:
        """Deterministic UUID -> re-ingesting the same PDF overwrites instead of duplicating."""
        return str(uuid.uuid5(_NAMESPACE, self.chunk_id))


def chunk_pages(pages: list[PageText], chunk_size: int = 1000, overlap: int = 150) -> list[Chunk]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks: list[Chunk] = []
    for page in pages:
        for i, piece in enumerate(splitter.split_text(page.text)):
            piece = piece.strip()
            if piece:
                chunks.append(
                    Chunk(f"{page.document_name}:p{page.page_number}:c{i}", page.document_name, page.page_number, i, piece)
                )
    return chunks
