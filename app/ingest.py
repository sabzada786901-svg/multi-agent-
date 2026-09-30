"""CLI: index PDFs.   python -m app.ingest "data\\pdfs\\document.pdf"   (or a folder; default data\\pdfs)"""
from __future__ import annotations

import sys
from pathlib import Path

from app.config import ConfigError, get_settings
from app.rag import chunker, embeddings, loader, vectorstore
from app.utils.security import friendly_error


def ingest_pdf(path: Path) -> int:
    pages = loader.load_pdf(path)
    chunks = chunker.chunk_pages(pages)
    if not chunks:
        raise loader.PDFLoadError(f"{path.name} produced no text chunks.")
    vectors = embeddings.embed_texts([c.text for c in chunks])
    vectorstore.ensure_collection(len(vectors[0]))
    vectorstore.delete_document(path.name)  # replace any older version of the same file
    vectorstore.upsert_chunks(chunks, vectors)
    return len(chunks)


def main(argv: list[str]) -> int:
    try:
        targets = [Path(a) for a in argv] or [get_settings().pdf_dir]
        files: list[Path] = []
        for t in targets:
            files += sorted(t.glob("*.pdf")) if t.is_dir() else [t]
        if not files:
            print("No PDF files found. Put PDFs in data\\pdfs and run again.")
            return 1
        code = 0
        for f in files:
            try:
                n = ingest_pdf(f)
                print(f"OK   {f.name}: indexed {n} chunks")
            except (loader.PDFLoadError, vectorstore.VectorStoreError) as exc:
                print(f"FAIL {f.name}: {exc}")
                code = 1
        return code
    except ConfigError as exc:
        print(f"Configuration error: {exc}")
        return 2
    except Exception as exc:  # noqa: BLE001
        print(friendly_error(exc))
        return 3


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
