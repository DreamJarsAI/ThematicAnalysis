from __future__ import annotations

import nltk


def ensure_nltk_data() -> None:
    """Download punkt tokenizer once per environment."""

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


__all__ = ["ensure_nltk_data"]
