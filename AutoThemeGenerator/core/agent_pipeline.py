from __future__ import annotations

import asyncio
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Sequence

from agents import Agent, Runner
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


class AutoThemeAgentPipeline:
    def __init__(
        self,
        api_key: str,
        config: ThematicAnalysisConfig,
        *,
        concurrency_limit: int = 4,
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

    def run(self) -> ThematicAnalysisResult:
        self._validate_prompt_lengths()
        with _temporary_openai_key(self.api_key):
            return asyncio.run(self._run())

    async def _run(self) -> ThematicAnalysisResult:
        chunk_texts: list[list[str]] = []
        for _, transcript in self.config.transcripts:
            chunks = chunk_transcript(transcript, self._chunking_config)
            chunk_texts.append(chunks)

        chunk_outputs: list[list[str]] = []
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
            responses = await self._execute_batch(prompts)
            chunk_outputs.append(responses)

        participant_summaries: list[list[str]] = []
        for participant_chunks in chunk_outputs:
            if not participant_chunks:
                participant_summaries.append([])
                continue
            summaries = await self._recursive_synthesis(
                participant_chunks,
                prompt_type="themes_same_id",
                token_threshold=self.config.combine_tokens_individual,
            )
            participant_summaries.append(summaries)

        flattened = [theme for summary in participant_summaries for theme in summary]
        if flattened:
            overall_summaries = await self._recursive_synthesis(
                flattened,
                prompt_type="themes_diff_id",
                token_threshold=self.config.combine_tokens_overall,
            )
        else:
            overall_summaries = []

        return ThematicAnalysisResult(
            chunk_texts=chunk_texts,
            chunk_outputs=chunk_outputs,
            participant_summaries=participant_summaries,
            overall_summaries=overall_summaries,
        )

    async def _execute_batch(self, prompts: Sequence[str]) -> list[str]:
        semaphore = asyncio.Semaphore(self._concurrency_limit)

        async def _run_prompt(prompt: str) -> str:
            async with semaphore:
                agent = self._create_agent()
                result = await Runner.run(agent, input=prompt)
                output = result.final_output or ""
                return output.strip()

        return await asyncio.gather(*(_run_prompt(prompt) for prompt in prompts))

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
        context_tokens = len(word_tokenize(self.config.context or ""))
        research_tokens = len(word_tokenize(self.config.research_questions or ""))
        script_tokens = len(word_tokenize(self.config.script or ""))

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
        tokens = len(word_tokenize(theme))
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
]
