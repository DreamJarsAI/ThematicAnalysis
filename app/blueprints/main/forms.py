from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from flask import Request, current_app
from nltk.tokenize import word_tokenize
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename


@dataclass
class AnalysisFormData:
    api_key: str
    model: str
    chunk_size: int
    study_context: str
    research_questions: str
    script: str | None
    transcript_paths: list[Path]


class FormValidationError(ValueError):
    """Raised when the user submission is invalid."""


def parse_analysis_form(request: Request, upload_dir: Path) -> AnalysisFormData:
    api_key = _clean_field(request.form.get("api_key"))
    if not api_key:
        raise FormValidationError("OpenAI API key is required.")

    model = (
        _clean_field(request.form.get("model")) or current_app.config["DEFAULT_MODEL"]
    )
    if model not in current_app.config["AVAILABLE_MODELS"]:
        raise FormValidationError("Selected model is not supported.")

    chunk_size_raw = _clean_field(request.form.get("chunk_size"))
    if chunk_size_raw:
        try:
            chunk_size = int(chunk_size_raw)
        except ValueError as exc:
            raise FormValidationError("Chunk size must be a number.") from exc
    else:
        chunk_size = current_app.config["DEFAULT_CHUNK_SIZE"]

    if chunk_size not in current_app.config["CHUNK_SIZE_CHOICES"]:
        raise FormValidationError("Chunk size is not supported.")

    files = request.files.getlist("transcripts")
    if not files:
        raise FormValidationError("Upload at least one transcript file.")

    transcript_paths = _save_files(files, upload_dir)
    if not transcript_paths:
        raise FormValidationError("No valid transcript files were uploaded.")

    study_context = _clean_field(request.form.get("study_context"))
    research_questions = _clean_field(request.form.get("research_questions"))
    script = _clean_field(request.form.get("script")) or None

    _validate_token_limits(study_context, research_questions, script)

    return AnalysisFormData(
        api_key=api_key,
        model=model,
        chunk_size=chunk_size,
        study_context=study_context or "",
        research_questions=research_questions or "",
        script=script,
        transcript_paths=transcript_paths,
    )


def _save_files(files: Iterable[FileStorage], upload_dir: Path) -> list[Path]:
    allowed = current_app.config["ALLOWED_EXTENSIONS"]
    saved_paths: list[Path] = []
    upload_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        if not file or not file.filename:
            continue
        filename = secure_filename(file.filename)
        if not filename:
            continue
        suffix = Path(filename).suffix.lower().lstrip(".")
        if suffix not in allowed:
            continue
        destination = upload_dir / filename
        file.save(destination)
        saved_paths.append(destination)

    return saved_paths


def _clean_field(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value.strip())


def _validate_token_limits(
    context: str, research_questions: str, script: str | None
) -> None:
    config = current_app.config
    context_tokens = len(word_tokenize(context)) if context else 0
    research_tokens = (
        len(word_tokenize(research_questions)) if research_questions else 0
    )
    script_tokens = len(word_tokenize(script)) if script else 0

    if context_tokens > config["MAX_CONTEXT_TOKENS"]:
        raise FormValidationError(
            "Study background exceeds the 500-token limit. Please shorten it."
        )
    if research_tokens > config["MAX_RESEARCH_TOKENS"]:
        raise FormValidationError(
            "Research questions exceed the 500-token limit. Please shorten them."
        )
    if script_tokens > config["MAX_SCRIPT_TOKENS"]:
        raise FormValidationError(
            "Interview script exceeds the 2000-token limit. Please shorten it."
        )

    total_tokens = context_tokens + research_tokens + script_tokens
    if total_tokens > config["MAX_METADATA_TOKENS"]:
        raise FormValidationError(
            "Combined study background, research questions, and script exceed the "
            "3000-token limit."
        )


__all__ = ["AnalysisFormData", "FormValidationError", "parse_analysis_form"]
