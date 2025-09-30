import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def _stub_word_tokenize(monkeypatch):
    from AutoThemeGenerator.core import agent_pipeline, chunking, prompt_builder
    from nltk import tokenize as nltk_tokenize
    from tests import test_chunking as test_chunking_module

    def _simple_tokenizer(text: str) -> list[str]:
        return text.split()

    monkeypatch.setattr(agent_pipeline, "word_tokenize", _simple_tokenizer)
    monkeypatch.setattr(chunking, "word_tokenize", _simple_tokenizer)
    monkeypatch.setattr(prompt_builder, "word_tokenize", _simple_tokenizer)
    monkeypatch.setattr(nltk_tokenize, "word_tokenize", _simple_tokenizer)
    test_chunking_module.word_tokenize = _simple_tokenizer
