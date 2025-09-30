from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .agent_pipeline import ThematicAnalysisResult


@dataclass(slots=True)
class StoredFile:
    label: str
    name: str
    path: Path


@dataclass(slots=True)
class AnalysisStorageRecord:
    job_id: str
    directory: Path
    files: list[StoredFile]


def write_analysis_outputs(
    *,
    result: ThematicAnalysisResult,
    participant_names: Sequence[str],
    output_dir: Path,
) -> AnalysisStorageRecord:
    output_dir.mkdir(parents=True, exist_ok=True)

    files: list[StoredFile] = []
    for index, name in enumerate(participant_names):
        safe_name = _slugify(name) or f"participant_{index + 1}"

        chunks_file = output_dir / f"{safe_name}_chunks.txt"
        chunks_file.write_text(
            _format_chunks(result.chunk_texts[index]),
            encoding="utf-8",
        )
        files.append(
            StoredFile(
                label=f"Transcript chunks — {name or safe_name}",
                name=chunks_file.name,
                path=chunks_file,
            )
        )

        themes_file = output_dir / f"{safe_name}_themes.txt"
        themes_file.write_text(
            _format_participant_themes(
                chunk_outputs=result.chunk_outputs[index],
                participant_summaries=result.participant_summaries[index],
            ),
            encoding="utf-8",
        )
        files.append(
            StoredFile(
                label=f"Themes — {name or safe_name}",
                name=themes_file.name,
                path=themes_file,
            )
        )

    overall_file = output_dir / "study_level_themes.txt"
    overall_file.write_text(
        _format_overall_themes(result.overall_summaries),
        encoding="utf-8",
    )
    files.append(
        StoredFile(
            label="Overall study themes",
            name=overall_file.name,
            path=overall_file,
        )
    )

    return AnalysisStorageRecord(
        job_id=output_dir.name,
        directory=output_dir,
        files=files,
    )


def create_archive(record: AnalysisStorageRecord) -> Path:
    archive_base = record.directory / f"{record.job_id}_results"
    archive_path = shutil.make_archive(
        str(archive_base), "zip", root_dir=record.directory
    )
    return Path(archive_path)


def _format_chunks(chunk_texts: Sequence[str]) -> str:
    if not chunk_texts:
        return "No transcript chunks were produced."

    sections = []
    for index, chunk in enumerate(chunk_texts, start=1):
        sections.append(f"Chunk {index}\n{chunk.strip()}")
    return "\n\n".join(sections)


def _format_participant_themes(
    *,
    chunk_outputs: Sequence[str],
    participant_summaries: Sequence[str],
) -> str:
    sections = [
        _format_numbered_section(
            heading="Chunk-specific themes",
            items=chunk_outputs,
            item_label="Chunk",
            empty_message="No chunk-specific themes were produced.",
        ),
        _format_numbered_section(
            heading="Participant themes",
            items=participant_summaries,
            item_label="Theme block",
            empty_message="No participant-level themes were produced.",
        ),
    ]
    return "\n\n".join(sections)


def _format_overall_themes(themes: Sequence[str]) -> str:
    return _format_numbered_section(
        heading="Study-level themes",
        items=themes,
        item_label="Theme block",
        empty_message="No study-level themes were produced.",
    )


def _format_numbered_section(
    *,
    heading: str,
    items: Sequence[str],
    item_label: str,
    empty_message: str,
) -> str:
    if not items:
        return f"{heading}\n{empty_message}"

    lines = [heading]
    for index, item in enumerate(items, start=1):
        lines.append("")
        lines.append(f"{item_label} {index}")
        lines.append(item.strip())
    return "\n".join(lines)


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", value.strip())


__all__ = [
    "AnalysisStorageRecord",
    "StoredFile",
    "create_archive",
    "write_analysis_outputs",
]
