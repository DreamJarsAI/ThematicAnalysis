from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

load_dotenv()

_BASE_DIR = Path(__file__).resolve().parent.parent
_ALLOWED_MODELS: Final[list[str]] = [
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-5-small",
    "gpt-5-medium",
    "gpt-5-large",
    "gpt-5.1",
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4o-mini",
    "gpt-4o",
]
_CHUNK_OPTIONS: Final[list[int]] = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
_MAX_CONTEXT_TOKENS: Final[int] = 500
_MAX_RESEARCH_TOKENS: Final[int] = 500
_MAX_SCRIPT_TOKENS: Final[int] = 2000


class Config:
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "autotheme-dev-key")
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 64 * 1024 * 1024))
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", str(_BASE_DIR / "uploads"))
    RESULTS_FOLDER = os.getenv("RESULTS_FOLDER", str(_BASE_DIR / "results"))
    ALLOWED_EXTENSIONS = {"txt", "pdf", "docx"}
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-5-nano")
    AVAILABLE_MODELS = _ALLOWED_MODELS
    DEFAULT_CHUNK_SIZE = 4000
    CHUNK_SIZE_CHOICES = _CHUNK_OPTIONS
    DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", 200))
    MAX_CONTEXT_TOKENS = _MAX_CONTEXT_TOKENS
    MAX_RESEARCH_TOKENS = _MAX_RESEARCH_TOKENS
    MAX_SCRIPT_TOKENS = _MAX_SCRIPT_TOKENS
    MAX_METADATA_TOKENS = (
        _MAX_CONTEXT_TOKENS + _MAX_RESEARCH_TOKENS + _MAX_SCRIPT_TOKENS
    )


__all__ = ["Config"]
