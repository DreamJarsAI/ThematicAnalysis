from __future__ import annotations

import asyncio
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Sequence

from agents import Agent, Runner
from agents.models.openai_provider import OpenAIProvider
from agents.run import RunConfig
from nltk.tokenize import word_tokenize

from .chunking import ChunkingConfig, chunk_transcript
from .prompt_builder import count_prompt_tokens, create_prompt

MAX_CONTEXT_TOKENS = 500
MAX_RESEARCH_TOKENS = 500
MAX_SCRIPT_TOKENS = 2000
MAX_METADATA_TOKENS = MAX_CONTEXT_TOKENS + MAX_RESEARCH_TOKENS + MAX_SCRIPT_TOKENS


class TokenLimitError(ValueError):
    """Raised when optional context exceeds supported token length."""


@dataclass(slots=True)
class ThematicAnalysisConfig:
    model: str
    chunk_size: int
    chunk_overlap: int
    context: str
    research_questions: str
    script: str | None
    transcripts: Sequence[tuple[str, str]]
    combine_tokens_individual: int = 5000
    combine_tokens_overall: int = 4000


@dataclass(slots=True)
class ThematicAnalysisResult:
    chunk_texts: list[list[str]]
    chunk_outputs: list[list[str]]
    participant_summaries: list[list[str]]
    overall_summaries: list[str]


@dataclass(slots=True)
class ProgressUpdate:
    stage: str
    message: str
    progress: float
    current: int
    total: int


class _ProgressTracker:
    """Utility for translating pipeline progress into UI friendly updates."""

    STAGE_ORDER = [
        "chunking",
        "chunk_analysis",
        "participant_summaries",
        "overall_summary",
    ]
    STAGE_WEIGHTS = {
        "chunking": 0.15,
        "chunk_analysis": 0.55,
        "participant_summaries": 0.2,
        "overall_summary": 0.1,
    }

    DEFAULT_MESSAGES = {
        "chunking": "Processed transcript {current} of {total}",
        "chunk_analysis": "Generated themes for chunk {current} of {total}",
        "participant_summaries": "Synthesized participant themes {current} of {total}",
        "overall_summary": "Created overall study themes",
    }

    def __init__(self, callback: Callable[[ProgressUpdate], None] | None) -> None:
        self._callback = callback
        self._totals = {stage: 0 for stage in self.STAGE_ORDER}
        self._completed = {stage: 0 for stage in self.STAGE_ORDER}

    def set_stage_total(self, stage: str, total: int) -> None:
        total = max(0, total)
        self._totals[stage] = total
        self._completed[stage] = min(self._completed.get(stage, 0), total)

    def increment(
        self,
        stage: str,
        *,
        message: str | None = None,
    ) -> tuple[int, int]:
        total = self._totals.get(stage, 0)
        completed = min(total, self._completed.get(stage, 0) + 1)
        self._completed[stage] = completed
        if message is None:
            template = self.DEFAULT_MESSAGES.get(stage, "{current}/{total}")
            message = template.format(current=completed, total=max(total, 1))
        self._emit(stage, message, completed, total)
        return completed, total

    def update(self, stage: str, *, message: str) -> None:
        total = self._totals.get(stage, 0)
        completed = self._completed.get(stage, 0)
        self._emit(stage, message, completed, total)

    def complete(self, stage: str, *, message: str | None = None) -> None:
        total = self._totals.get(stage, 0)
        self._completed[stage] = total
        if message is None:
            template = self.DEFAULT_MESSAGES.get(stage, "{current}/{total}")
            message = template.format(current=total, total=max(total, 1))
        self._emit(stage, message, total, total)

    def _emit(self, stage: str, message: str, current: int, total: int) -> None:
        if not self._callback:
            return
        progress = self._total_progress()
        update = ProgressUpdate(
            stage=stage,
            message=message,
            progress=min(progress, 1.0),
            current=current,
            total=total,
        )
        self._callback(update)

    def _total_progress(self) -> float:
        progress = 0.0
        for stage in self.STAGE_ORDER:
            weight = self.STAGE_WEIGHTS[stage]
            total = self._totals.get(stage, 0)
            if total <= 0:
                progress += weight
                continue
            completed = min(self._completed.get(stage, 0), total)
            progress += weight * (completed / total)
        return min(progress, 1.0)


def _safe_tokenize(text: str) -> list[str]:
    try:
        return word_tokenize(text)
    except LookupError:
        return text.split()


class AutoThemeAgentPipeline:
    def __init__(
        self,
        api_key: str,
        config: ThematicAnalysisConfig,
        *,
        concurrency_limit: int = 4,
        progress_callback: Callable[[ProgressUpdate], None] | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAI API key is required for analysis.")
        self.api_key = api_key
        self.config = config
        self._chunking_config = ChunkingConfig(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        self._concurrency_limit = max(1, concurrency_limit)
        self._progress_tracker = _ProgressTracker(progress_callback)
        self._model_provider = OpenAIProvider(api_key=api_key)

    def run(self) -> ThematicAnalysisResult:
        self._validate_prompt_lengths()
        with _temporary_openai_key(self.api_key):
            return asyncio.run(self._run())

    async def _run(self) -> ThematicAnalysisResult:
        tracker = self._progress_tracker
        if tracker:
            tracker.set_stage_total("chunking", len(self.config.transcripts))
            tracker.update(
                "chunking",
                message="Preparing transcripts for chunking...",
            )

        chunk_texts: list[list[str]] = []
        for index, (_, transcript) in enumerate(self.config.transcripts, start=1):
            chunks = chunk_transcript(transcript, self._chunking_config)
            chunk_texts.append(chunks)
            if tracker:
                tracker.increment(
                    "chunking",
                    message=(
                        f"Processed transcript {index} of "
                        f"{max(len(self.config.transcripts), 1)}"
                    ),
                )

        chunk_outputs: list[list[str]] = []
        total_chunks = sum(len(chunks) for chunks in chunk_texts)
        if tracker:
            tracker.set_stage_total("chunk_analysis", total_chunks)
            if total_chunks:
                tracker.update(
                    "chunk_analysis",
                    message="Generating themes for each transcript chunk...",
                )
            else:
                tracker.complete(
                    "chunk_analysis",
                    message="No transcript chunks required theme generation.",
                )
        for chunks in chunk_texts:
            prompts = [
                create_prompt(
                    context=self.config.context,
                    research_questions=self.config.research_questions,
                    script=self.config.script,
                    text_chunk=chunk,
                    prompt_type="transcript",
                )
                for chunk in chunks
            ]
            responses = await self._execute_batch(
                prompts,
                stage="chunk_analysis",
            )
            chunk_outputs.append(responses)

        participant_summaries: list[list[str]] = []
        participant_totals = sum(1 for participant_chunks in chunk_outputs if participant_chunks)
        if tracker:
            tracker.set_stage_total("participant_summaries", participant_totals)
            if participant_totals:
                tracker.update(
                    "participant_summaries",
                    message="Synthesizing participant-level themes...",
                )
            else:
                tracker.complete(
                    "participant_summaries",
                    message="No participant summaries required.",
                )

        completed_participants = 0
        for index, participant_chunks in enumerate(chunk_outputs, start=1):
            if not participant_chunks:
                participant_summaries.append([])
                continue
            summaries = await self._recursive_synthesis(
                participant_chunks,
                prompt_type="themes_same_id",
                token_threshold=self.config.combine_tokens_individual,
            )
            participant_summaries.append(summaries)
            if tracker and participant_chunks:
                completed_participants += 1
                tracker.increment(
                    "participant_summaries",
                    message=(
                        f"Synthesized participant themes {completed_participants}"
                        f" of {max(participant_totals, 1)}"
                    ),
                )

        flattened = [theme for summary in participant_summaries for theme in summary]
        if tracker:
            tracker.set_stage_total("overall_summary", 1 if flattened else 0)
            if flattened:
                tracker.update(
                    "overall_summary",
                    message="Creating overall study themes...",
                )
            else:
                tracker.complete(
                    "overall_summary",
                    message="No overall study themes were generated.",
                )
        if flattened:
            overall_summaries = await self._recursive_synthesis(
                flattened,
                prompt_type="themes_diff_id",
                token_threshold=self.config.combine_tokens_overall,
            )
            if tracker:
                tracker.complete(
                    "overall_summary",
                    message="Created overall study themes.",
                )
        else:
            overall_summaries = []

        return ThematicAnalysisResult(
            chunk_texts=chunk_texts,
            chunk_outputs=chunk_outputs,
            participant_summaries=participant_summaries,
            overall_summaries=overall_summaries,
        )

    async def _execute_batch(
        self,
        prompts: Sequence[str],
        *,
        stage: str | None = None,
    ) -> list[str]:
        semaphore = asyncio.Semaphore(self._concurrency_limit)

        async def _run_prompt(index: int, prompt: str) -> tuple[int, str]:
            async with semaphore:
                agent = self._create_agent()
                run_config = RunConfig(model_provider=self._model_provider)
                result = await Runner.run(
                    agent,
                    input=prompt,
                    run_config=run_config,
                )
                output = result.final_output or ""
                return index, output.strip()

        tasks = [
            asyncio.create_task(_run_prompt(index, prompt))
            for index, prompt in enumerate(prompts)
        ]
        responses = [""] * len(prompts)

        for task in asyncio.as_completed(tasks):
            index, value = await task
            responses[index] = value
            if stage and self._progress_tracker and len(prompts) > 0:
                self._progress_tracker.increment(stage)

        return responses

    async def _recursive_synthesis(
        self,
        themes: Sequence[str],
        *,
        prompt_type: str,
        token_threshold: int,
    ) -> list[str]:
        combined = _combine_themes(themes, token_threshold)
        prompts = [
            create_prompt(
                context=self.config.context,
                research_questions=self.config.research_questions,
                script=self.config.script,
                text_chunk=theme,
                prompt_type=prompt_type,
            )
            for theme in combined
        ]
        responses = await self._execute_batch(prompts)
        if len(responses) == 1:
            return responses
        return await self._recursive_synthesis(
            responses,
            prompt_type=prompt_type,
            token_threshold=token_threshold,
        )

    def _create_agent(self) -> Agent:
        return Agent(
            name="AutoTheme Assistant",
            instructions="You are a helpful research assistant.",
            model=self.config.model,
        )

    def _validate_prompt_lengths(self) -> None:
        context_tokens = len(_safe_tokenize(self.config.context or ""))
        research_tokens = len(_safe_tokenize(self.config.research_questions or ""))
        script_tokens = len(_safe_tokenize(self.config.script or ""))

        if context_tokens > MAX_CONTEXT_TOKENS:
            raise TokenLimitError(
                "Study background exceeds the 500-token limit. Please shorten it."
            )
        if research_tokens > MAX_RESEARCH_TOKENS:
            raise TokenLimitError(
                "Research questions exceed the 500-token limit. Please shorten them."
            )
        if script_tokens > MAX_SCRIPT_TOKENS:
            raise TokenLimitError(
                "Interview script exceeds the 2000-token limit. Please shorten it."
            )

        metadata_tokens = count_prompt_tokens(
            self.config.context,
            self.config.research_questions,
            self.config.script or "",
        )
        if metadata_tokens > MAX_METADATA_TOKENS:
            raise TokenLimitError(
                "Combined study background, research questions, and script exceed the 3000-token limit."
            )


def _combine_themes(themes: Sequence[str], token_threshold: int) -> list[str]:
    combined_texts: list[str] = []
    current_text: list[str] = []
    current_tokens = 0

    for theme in themes:
        tokens = len(_safe_tokenize(theme))
        if not current_text:
            current_text.append(theme)
            current_tokens = tokens
            continue

        if current_tokens + tokens > token_threshold:
            combined_texts.append("\n\n".join(current_text))
            current_text = [theme]
            current_tokens = tokens
        else:
            current_text.append(theme)
            current_tokens += tokens

    if current_text:
        combined_texts.append("\n\n".join(current_text))

    return combined_texts


@contextmanager
def _temporary_openai_key(key: str):
    previous = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = key
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = previous


__all__ = [
    "AutoThemeAgentPipeline",
    "ThematicAnalysisConfig",
    "ThematicAnalysisResult",
    "TokenLimitError",
    "ProgressUpdate",
]
