from nltk.tokenize import word_tokenize

from AutoThemeGenerator.core.chunking import ChunkingConfig, chunk_transcript


def test_chunk_transcript_token_lengths_respected():
    text = (
        "Sentence one is here. Sentence two follows closely. "
        "Sentence three adds more detail. Sentence four concludes."
    )
    config = ChunkingConfig(chunk_size=12, chunk_overlap=2)
    chunks = chunk_transcript(text, config)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(word_tokenize(chunk)) <= config.chunk_size + 1
