import pytest

from AutoThemeGenerator.core.agent_pipeline import (
    AutoThemeAgentPipeline,
    ProgressUpdate,
    ThematicAnalysisConfig,
    TokenLimitError,
)


def test_pipeline_runs_with_mocked_batches(monkeypatch):
    transcripts = [
        ("participant_one", "This is sentence one. This is sentence two."),
        ("participant_two", "Another person speaks here. They add more content."),
    ]

    async def fake_execute_batch(self, prompts, *, stage=None):
        return [f"response-{i}" for i, _ in enumerate(prompts, start=1)]

    monkeypatch.setattr(
        AutoThemeAgentPipeline,
        "_execute_batch",
        fake_execute_batch,
    )

    config = ThematicAnalysisConfig(
        model="gpt-5-nano",
        chunk_size=10,
        chunk_overlap=2,
        context="Background",
        research_questions="Research question",
        script=None,
        transcripts=transcripts,
        combine_tokens_individual=50,
        combine_tokens_overall=50,
    )

    pipeline = AutoThemeAgentPipeline(api_key="test-key", config=config)
    result = pipeline.run()

    assert len(result.chunk_outputs) == len(transcripts)
    assert all(result.chunk_outputs)
    assert len(result.participant_summaries) == len(transcripts)
    assert result.overall_summaries


def test_pipeline_reports_progress(monkeypatch):
    transcripts = [("participant_one", "Sentence one. Sentence two.")]

    async def fake_execute_batch(self, prompts, *, stage=None):
        return [f"response-{i}" for i, _ in enumerate(prompts, start=1)]

    monkeypatch.setattr(
        AutoThemeAgentPipeline,
        "_execute_batch",
        fake_execute_batch,
    )

    updates: list[ProgressUpdate] = []

    config = ThematicAnalysisConfig(
        model="gpt-5-nano",
        chunk_size=10,
        chunk_overlap=2,
        context="Background",
        research_questions="Research question",
        script=None,
        transcripts=transcripts,
        combine_tokens_individual=50,
        combine_tokens_overall=50,
    )

    pipeline = AutoThemeAgentPipeline(
        api_key="test-key",
        config=config,
        progress_callback=updates.append,
    )
    pipeline.run()

    assert updates
    assert any(update.stage == "chunk_analysis" for update in updates)
    assert updates[-1].progress <= 1.0


def _build_config(**overrides):
    base = dict(
        model="gpt-5-nano",
        chunk_size=10,
        chunk_overlap=2,
        context="Background",
        research_questions="Research question",
        script=None,
        transcripts=[("p1", "Text")],
        combine_tokens_individual=50,
        combine_tokens_overall=50,
    )
    base.update(overrides)
    return ThematicAnalysisConfig(**base)


@pytest.mark.parametrize(
    "override,expected",
    [
        ({"context": "word " * 501}, "background"),
        ({"research_questions": "word " * 501}, "questions"),
        ({"script": "word " * 2001}, "script"),
    ],
)
def test_token_limit_error_triggered_per_field(override, expected):
    config = _build_config(**override)
    pipeline = AutoThemeAgentPipeline(api_key="test-key", config=config)

    with pytest.raises(TokenLimitError, match=expected):
        pipeline._validate_prompt_lengths()
