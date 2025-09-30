from __future__ import annotations

import shutil
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

from flask import (
    Blueprint,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)

from AutoThemeGenerator.core.agent_pipeline import (
    AutoThemeAgentPipeline,
    ProgressUpdate,
    ThematicAnalysisConfig,
    TokenLimitError,
)
from AutoThemeGenerator.core.file_ingest import load_transcripts
from AutoThemeGenerator.core.storage import create_archive, write_analysis_outputs

from .forms import FormValidationError, parse_analysis_form
from ...services.analysis_jobs import AnalysisJobStore

main_bp = Blueprint("main", __name__)


@main_bp.route("/", methods=["GET", "POST"])
def index() -> str:
    if request.method == "POST":
        job_id = uuid4().hex
        upload_dir = Path(current_app.config["UPLOAD_FOLDER"]) / job_id
        results_dir = Path(current_app.config["RESULTS_FOLDER"]) / job_id
        job_store = _get_job_store()

        try:
            form_data = parse_analysis_form(request, upload_dir)
            transcripts = load_transcripts(form_data.transcript_paths)
            _cleanup_directory(upload_dir)

            config = ThematicAnalysisConfig(
                model=form_data.model,
                chunk_size=form_data.chunk_size,
                chunk_overlap=current_app.config["DEFAULT_CHUNK_OVERLAP"],
                context=form_data.study_context,
                research_questions=form_data.research_questions,
                script=form_data.script,
                transcripts=transcripts,
            )

        except FormValidationError as exc:
            _cleanup_directory(upload_dir)
            flash(str(exc), "danger")
            return redirect(url_for("main.index"))
        except TokenLimitError as exc:
            _cleanup_directory(upload_dir)
            flash(str(exc), "warning")
            return redirect(url_for("main.index"))
        except Exception as exc:  # noqa: BLE001
            flash("An unexpected error occurred during analysis.", "danger")
            current_app.logger.exception("Analysis preparation failure: %s", exc)
            _cleanup_directory(upload_dir)
            return redirect(url_for("main.index"))

        job_store.create(job_id)

        app = current_app._get_current_object()
        thread = threading.Thread(
            target=_run_analysis_job,
            args=(
                app,
                job_id,
                form_data,
                transcripts,
                results_dir,
                config,
            ),
            daemon=True,
        )
        thread.start()

        return redirect(url_for("main.progress", job_id=job_id))

    return render_template(
        "index.html",
        models=current_app.config["AVAILABLE_MODELS"],
        default_model=current_app.config["DEFAULT_MODEL"],
        chunk_sizes=current_app.config["CHUNK_SIZE_CHOICES"],
        default_chunk=current_app.config["DEFAULT_CHUNK_SIZE"],
    )


@main_bp.route("/progress/<job_id>")
def progress(job_id: str) -> str:
    job_store = _get_job_store()
    if not job_store.exists(job_id):
        flash("We couldn't find that analysis job. Please start a new analysis.", "warning")
        return redirect(url_for("main.index"))

    return render_template("progress.html", job_id=job_id)


@main_bp.route("/progress/<job_id>/status")
def job_status(job_id: str) -> Any:
    job_store = _get_job_store()
    state = job_store.get(job_id)
    if state is None:
        return jsonify({"error": "Job not found."}), 404

    data = state.to_dict()
    data["job_id"] = job_id
    return jsonify(data)


@main_bp.route("/results/<job_id>")
def results(job_id: str) -> str:
    job_store = _get_job_store()
    result = job_store.result(job_id)
    if result is None:
        if job_store.exists(job_id):
            flash("Analysis is still running. Please wait for it to complete.", "info")
            return redirect(url_for("main.progress", job_id=job_id))
        flash("We couldn't find results for that job.", "warning")
        return redirect(url_for("main.index"))

    return render_template(
        "results.html",
        files=result.record.files,
        archive_path=result.archive_path.name,
        job_id=result.record.job_id,
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


def _run_analysis_job(
    app,
    job_id: str,
    form_data,
    transcripts,
    results_dir: Path,
    config: ThematicAnalysisConfig,
) -> None:
    with app.app_context():
        job_store = app.extensions["analysis_jobs"]

        def _handle_progress(update: ProgressUpdate) -> None:
            job_store.update_from_progress(job_id, update)

        try:
            results_dir.mkdir(parents=True, exist_ok=True)
            pipeline = AutoThemeAgentPipeline(
                api_key=form_data.api_key,
                config=config,
                progress_callback=_handle_progress,
            )
            result = pipeline.run()

            storage_record = write_analysis_outputs(
                result=result,
                participant_names=[name for name, _ in transcripts],
                output_dir=results_dir,
            )
            archive_path = create_archive(storage_record)
            job_store.mark_completed(
                job_id,
                record=storage_record,
                archive_path=archive_path,
            )
        except TokenLimitError as exc:
            job_store.mark_failed(job_id, str(exc))
        except Exception as exc:  # noqa: BLE001
            job_store.mark_failed(
                job_id,
                "An unexpected error occurred during analysis.",
            )
            app.logger.exception("Analysis failure: %s", exc)
        finally:
            state = job_store.get(job_id)
            if (state is None or state.status != "completed") and results_dir.exists():
                shutil.rmtree(results_dir, ignore_errors=True)


def _get_job_store() -> AnalysisJobStore:
    return current_app.extensions["analysis_jobs"]


__all__ = ["main_bp"]
