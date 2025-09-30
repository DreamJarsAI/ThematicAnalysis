from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter
from nltk.tokenize import word_tokenize


@dataclass(slots=True)
class ChunkingConfig:
    chunk_size: int
    chunk_overlap: int


def chunk_transcript(text: str, config: ChunkingConfig) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", ".", "?", "!"],
        length_function=_token_length,
    )
    chunks = [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]
    return chunks or [text.strip()]


def _token_length(text: str) -> int:
    return len(_safe_tokenize(text))


def _safe_tokenize(text: str) -> list[str]:
    try:
        return word_tokenize(text)
    except LookupError:
        return text.split()


__all__ = ["ChunkingConfig", "chunk_transcript"]
