"""PDF loading + text extraction (page by page, so page numbers are real)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class PDFLoadError(RuntimeError):
    """User-safe PDF problem."""


@dataclass
class PageText:
    document_name: str
    page_number: int  # 1-based
    text: str


def load_pdf(path: str | Path) -> list[PageText]:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    p = Path(path)
    if not p.is_file():
        raise PDFLoadError(f"File not found: {p}")
    if p.suffix.lower() != ".pdf":
        raise PDFLoadError(f"Not a PDF file: {p.name}")
    try:
        reader = PdfReader(str(p))
        if reader.is_encrypted and not reader.decrypt(""):
            raise PDFLoadError(f"{p.name} is password-protected.")
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(PageText(p.name, i, text))
    except PDFLoadError:
        raise
    except (PdfReadError, ValueError, OSError, KeyError) as exc:
        raise PDFLoadError(f"Could not read {p.name}: the PDF looks corrupted or unsupported.") from exc
    if not pages:
        raise PDFLoadError(
            f"No extractable text in {p.name}. It may be a scanned/image-only PDF (OCR is not included)."
        )
    return pages
