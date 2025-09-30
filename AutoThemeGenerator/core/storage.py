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
        chunk_file = output_dir / f"{safe_name}_chunk_themes.txt"
        chunk_file.write_text(
            _format_chunk_outputs(
                chunk_texts=result.chunk_texts[index],
                chunk_outputs=result.chunk_outputs[index],
            ),
            encoding="utf-8",
        )
        files.append(
            StoredFile(
                label=f"Chunk-level themes — {name or safe_name}",
                name=chunk_file.name,
                path=chunk_file,
            )
        )

        summary_file = output_dir / f"{safe_name}_participant_summary.txt"
        summary_file.write_text(
            _format_list(
                result.participant_summaries[index], heading="Participant themes"
            ),
            encoding="utf-8",
        )
        files.append(
            StoredFile(
                label=f"Synthesized participant themes — {name or safe_name}",
                name=summary_file.name,
                path=summary_file,
            )
        )

    overall_file = output_dir / "overall_study_themes.txt"
    overall_file.write_text(
        _format_list(result.overall_summaries, heading="Overall study themes"),
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


def _format_chunk_outputs(
    *,
    chunk_texts: Sequence[str],
    chunk_outputs: Sequence[str],
) -> str:
    sections: list[str] = []
    for index, (chunk_text, output) in enumerate(
        zip(chunk_texts, chunk_outputs, strict=True), start=1
    ):
        sections.append(
            "\n".join(
                [
                    f"Chunk {index} — original excerpt:",
                    chunk_text.strip(),
                    "",
                    "Generated themes:",
                    output.strip(),
                ]
            )
        )
    return "\n\n".join(sections)


def _format_list(items: Sequence[str], *, heading: str) -> str:
    if not items:
        return f"{heading}\nNo themes were produced."

    sections = [heading]
    for index, item in enumerate(items, start=1):
        sections.append(f"\nTheme block {index}\n{item.strip()}")
    return "\n".join(sections)


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", value.strip())


__all__ = [
    "AnalysisStorageRecord",
    "StoredFile",
    "create_archive",
    "write_analysis_outputs",
]
