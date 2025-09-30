from __future__ import annotations

from pathlib import Path
from typing import Iterable

from docx import Document
from PyPDF2 import PdfReader


class TranscriptReadError(RuntimeError):
    """Raised when a transcript cannot be read."""


def load_transcripts(paths: Iterable[Path]) -> list[tuple[str, str]]:
    transcripts: list[tuple[str, str]] = []

    for path in paths:
        text = _read_file(path)
        if text.strip():
            transcripts.append((path.stem, text.strip()))

    if not transcripts:
        raise TranscriptReadError("No readable transcripts were provided.")

    return transcripts


def _read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return _read_docx(path)
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    raise TranscriptReadError(f"Unsupported file type: {path.name}")


def _read_docx(path: Path) -> str:
    document = Document(path)
    paragraphs = [para.text for para in document.paragraphs if para.text]
    return "\n".join(paragraphs)


def _read_pdf(path: Path) -> str:
    reader = PdfReader(path)
    texts: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        texts.append(page_text)
    return "\n".join(texts)


__all__ = ["load_transcripts", "TranscriptReadError"]
