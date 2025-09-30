from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from agents import Agent, Runner
from nltk.tokenize import word_tokenize

from .core.agent_pipeline import AutoThemeAgentPipeline, ThematicAnalysisConfig
from .core.chunking import ChunkingConfig, chunk_transcript
from .core.file_ingest import load_transcripts
from .core.prompt_builder import count_prompt_tokens, create_prompt

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


CITATION = """
If you find this package useful, please cite: 
Yuyi Yang, Charles Alba, Chenyu Wang, Xi Wang, Jami Anderson, and Ruopeng An. 
"GPT Models Can Perform Thematic Analysis in Public Health Studies, Akin to Qualitative Researchers." 
Journal of Social Computing, vol. 5, no. 4, (2024): 293-312. 
doi: https://doi.org/10.23919/JSC.2024.0024
""".strip()

SUPPORTED_SUFFIXES = {".txt", ".pdf", ".docx"}


@dataclass(slots=True)
class AnalysisOutputs:
    initial_themes: list[list[str]]
    participant_themes: list[list[str]]
    overall_themes: list[str]
    artifacts_dir: Path | None = None
    archive_path: Path | None = None


def read_transcripts(directory_path: str | Path) -> list[str]:
    directory = Path(directory_path)
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    candidates = [
        p for p in directory.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    transcripts = load_transcripts(candidates)
    total_tokens = sum(len(word_tokenize(text)) for _, text in transcripts)

    print(f"Total number of transcripts: {len(transcripts)}")
    print(f"Total number of tokens in the transcripts: {total_tokens}")

    return [text for _, text in transcripts]


def chunk_text(
    text: str, max_tokens_chunk: int = 4000, overlap_tokens: int = 200
) -> list[str]:
    config = ChunkingConfig(chunk_size=max_tokens_chunk, chunk_overlap=overlap_tokens)
    return chunk_transcript(text, config)


def process_transcripts(
    directory_path: str | Path, max_tokens_chunk: int = 4000, overlap_tokens: int = 200
) -> list[list[str]]:
    transcripts = read_transcripts(directory_path)
    chunks: list[list[str]] = []
    source = (
        tqdm(transcripts, desc="Chunking transcripts", unit="file")
        if tqdm is not None
        else transcripts
    )
    for transcript in source:
        chunks.append(
            chunk_text(
                transcript,
                max_tokens_chunk=max_tokens_chunk,
                overlap_tokens=overlap_tokens,
            )
        )
    return chunks


def generate_response(prompt: str, api_key: str, model: str = "gpt-5-nano") -> str:
    async def _run() -> str:
        agent = Agent(
            name="AutoTheme Assistant",
            instructions="You are a helpful research assistant.",
            model=model,
        )
        result = await Runner.run(agent, input=prompt)
        return (result.final_output or "").strip()

    previous = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = api_key
    try:
        return asyncio.run(_run())
    finally:
        if previous is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = previous


def synthesize_themes(
    themes_list: Sequence[str],
    context: str,
    research_questions: str,
    script: str | None,
    api_key: str,
    prompt_type: str = "themes_same_id",
    model: str = "gpt-5-nano",
    token_threshold: int = 5000,
) -> list[str]:
    config = ThematicAnalysisConfig(
        model=model,
        chunk_size=token_threshold,
        chunk_overlap=0,
        context=context,
        research_questions=research_questions,
        script=script,
        transcripts=[("themes", "\n\n".join(themes_list))],
        combine_tokens_individual=token_threshold,
        combine_tokens_overall=token_threshold,
    )
    pipeline = AutoThemeAgentPipeline(
        api_key=api_key, config=config, concurrency_limit=1
    )
    result = pipeline.run()
    if prompt_type == "themes_diff_id":
        return result.overall_summaries
    return result.participant_summaries[0]


def save_results_to_json(results: Sequence, file_path: str | Path) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)


def load_results_from_json(file_path: str | Path):
    with Path(file_path).open("r", encoding="utf-8") as file:
        return json.load(file)


def analyze_and_synthesize_transcripts(
    directory_path: str | Path,
    context: str,
    research_questions: str,
    script: str | None,
    api_key: str,
    save_results_path: str | Path,
    *,
    model: str = "gpt-5-nano",
    max_tokens_chunk: int = 4000,
    overlap_tokens: int = 200,
    max_tokens_combine_ind_themes: int = 5000,
    max_tokens_combine_all_themes: int = 4000,
) -> tuple[list[list[str]], list[list[str]], list[str]]:
    print(CITATION)
    start_time = time.time()

    directory = Path(directory_path)
    files = [p for p in directory.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES]
    if not files:
        raise FileNotFoundError("No supported transcript files found in directory.")

    transcripts = load_transcripts(files)
    config = ThematicAnalysisConfig(
        model=model,
        chunk_size=max_tokens_chunk,
        chunk_overlap=overlap_tokens,
        context=context,
        research_questions=research_questions,
        script=script,
        transcripts=transcripts,
        combine_tokens_individual=max_tokens_combine_ind_themes,
        combine_tokens_overall=max_tokens_combine_all_themes,
    )
    pipeline = AutoThemeAgentPipeline(api_key=api_key, config=config)
    result = pipeline.run()

    save_path = Path(save_results_path)
    save_path.mkdir(parents=True, exist_ok=True)

    raw_path = save_path / "themes_raw.json"
    per_person_path = save_path / "themes_per_person.json"
    overall_path = save_path / "themes_overall.json"

    save_results_to_json(result.chunk_outputs, raw_path)
    save_results_to_json(result.participant_summaries, per_person_path)
    save_results_to_json(result.overall_summaries, overall_path)

    elapsed = time.time() - start_time
    print(f"Function execution time: {elapsed:.2f} seconds")

    return result.chunk_outputs, result.participant_summaries, result.overall_summaries


def download_example(
    folder_name: str = "example_transcripts",
    url: str = "https://sites.wustl.edu/alba/files/2024/04/book_empathy_club-02119c68e92058fe.zip",
) -> None:
    import shutil
    import zipfile
    from io import BytesIO

    import requests

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    folder = Path(folder_name)
    folder.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(BytesIO(response.content)) as zip_file:
        zip_file.extractall(folder)

    source_folder = folder / "book_empathy_club"
    if source_folder.exists():
        for item in source_folder.iterdir():
            target = folder / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            item.rename(target)
        source_folder.rmdir()


__all__ = [
    "AnalysisOutputs",
    "read_transcripts",
    "chunk_text",
    "process_transcripts",
    "generate_response",
    "synthesize_themes",
    "save_results_to_json",
    "load_results_from_json",
    "analyze_and_synthesize_transcripts",
    "download_example",
    "create_prompt",
    "count_prompt_tokens",
]
