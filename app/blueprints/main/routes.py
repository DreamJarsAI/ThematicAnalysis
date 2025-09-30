from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from flask import (
    Blueprint,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)

from AutoThemeGenerator.core.agent_pipeline import (
    AutoThemeAgentPipeline,
    ThematicAnalysisConfig,
    TokenLimitError,
)
from AutoThemeGenerator.core.file_ingest import load_transcripts
from AutoThemeGenerator.core.storage import create_archive, write_analysis_outputs

from .forms import FormValidationError, parse_analysis_form

main_bp = Blueprint("main", __name__)


@main_bp.route("/", methods=["GET", "POST"])
def index() -> str:
    if request.method == "POST":
        job_id = uuid4().hex
        upload_dir = Path(current_app.config["UPLOAD_FOLDER"]) / job_id
        results_dir = Path(current_app.config["RESULTS_FOLDER"]) / job_id

        try:
            form_data = parse_analysis_form(request, upload_dir)
            transcripts = load_transcripts(form_data.transcript_paths)

            config = ThematicAnalysisConfig(
                model=form_data.model,
                chunk_size=form_data.chunk_size,
                chunk_overlap=current_app.config["DEFAULT_CHUNK_OVERLAP"],
                context=form_data.study_context,
                research_questions=form_data.research_questions,
                script=form_data.script,
                transcripts=transcripts,
            )

            pipeline = AutoThemeAgentPipeline(
                api_key=form_data.api_key,
                config=config,
            )
            result = pipeline.run()

            storage_record = write_analysis_outputs(
                result=result,
                participant_names=[name for name, _ in transcripts],
                output_dir=results_dir,
            )
            archive_path = create_archive(storage_record)
        except FormValidationError as exc:
            _cleanup_directory(upload_dir)
            flash(str(exc), "danger")
            return redirect(url_for("main.index"))
        except TokenLimitError as exc:
            _cleanup_directory(upload_dir)
            flash(str(exc), "warning")
            return redirect(url_for("main.index"))
        except Exception as exc:  # noqa: BLE001
            _cleanup_directory(upload_dir)
            flash("An unexpected error occurred during analysis.", "danger")
            current_app.logger.exception("Analysis failure: %s", exc)
            return redirect(url_for("main.index"))
        finally:
            _cleanup_directory(upload_dir)

        return render_template(
            "results.html",
            files=storage_record.files,
            archive_path=archive_path.name,
            job_id=storage_record.job_id,
        )

    return render_template(
        "index.html",
        models=current_app.config["AVAILABLE_MODELS"],
        default_model=current_app.config["DEFAULT_MODEL"],
        chunk_sizes=current_app.config["CHUNK_SIZE_CHOICES"],
        default_chunk=current_app.config["DEFAULT_CHUNK_SIZE"],
    )


@main_bp.route("/results/<job_id>/<path:filename>")
def download(job_id: str, filename: str) -> Any:
    results_dir = Path(current_app.config["RESULTS_FOLDER"]) / job_id
    if not results_dir.exists():
        flash("Requested file is no longer available.", "warning")
        return redirect(url_for("main.index"))

    return send_from_directory(results_dir, filename, as_attachment=True)


def _cleanup_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


__all__ = ["main_bp"]
