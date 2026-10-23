"""Load a novel from .txt or text-based .pdf and normalise it for NLP.

Normalisation matters for NER quality: hard-wrapped lines (common in Project
Gutenberg files) split names and sentences, so we re-join them into paragraphs.
"""
from __future__ import annotations

import re
from pathlib import Path

_GUTENBERG_START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\n", re.I)
_GUTENBERG_END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG", re.I)


def read_file(path: str | Path) -> str:
    """Read .txt or .pdf and return cleaned text."""
    path = Path(path)
    data = path.read_bytes()
    return read_upload(path.name, data)


def read_upload(name: str, data: bytes) -> str:
    """Same as read_file but for an in-memory upload (e.g. Streamlit)."""
    raw = _read_pdf(data) if name.lower().endswith(".pdf") else _decode(data)
    return clean_text(raw)


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _read_pdf(data: bytes) -> str:
    try:
        import pymupdf  # PyMuPDF >= 1.24
    except ImportError:  # pragma: no cover
        import fitz as pymupdf  # type: ignore
    pages = []
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            pages.append(page.get_text("text"))
    text = "\n".join(pages)
    if len(text.strip()) < 200:
        raise ValueError(
            "This PDF has almost no selectable text (it is probably scanned). "
            "Run OCR first, or upload a .txt version."
        )
    return text


def clean_text(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    # Remove Project Gutenberg licence header/footer when present.
    start = _GUTENBERG_START.search(text)
    if start:
        text = text[start.end():]
    end = _GUTENBERG_END.search(text)
    if end:
        text = text[: end.start()]
    # De-hyphenate words broken across lines ("Hel-\nsing" -> "Helsing").
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Re-join hard-wrapped lines: single newline -> space, blank line -> paragraph.
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [re.sub(r"\s*\n\s*", " ", p).strip() for p in paragraphs]
    paragraphs = [re.sub(r"[ \t]{2,}", " ", p) for p in paragraphs if p]
    return "\n\n".join(paragraphs)
