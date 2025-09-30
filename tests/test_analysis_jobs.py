from AutoThemeGenerator.core.agent_pipeline import ProgressUpdate
from AutoThemeGenerator.core.storage import AnalysisStorageRecord

from app.services.analysis_jobs import AnalysisJobStore


def test_job_store_tracks_progress_and_completion(tmp_path):
    store = AnalysisJobStore(max_events=10)
    job_id = "job-123"
    store.create(job_id)

    store.update_from_progress(
        job_id,
        ProgressUpdate(
            stage="chunking",
            message="Processed transcript 1 of 2",
            progress=0.2,
            current=1,
            total=2,
        ),
    )

    state = store.get(job_id)
    assert state is not None
    assert state.status == "running"
    assert state.progress == 0.2
    assert state.events

    record = AnalysisStorageRecord(job_id=job_id, directory=tmp_path, files=[])
    archive_path = tmp_path / "archive.zip"
    store.mark_completed(job_id, record=record, archive_path=archive_path)

    result = store.result(job_id)
    assert result is not None
    assert result.archive_path == archive_path

    state = store.get(job_id)
    assert state is not None
    assert state.status == "completed"
    assert state.progress == 1.0


def test_job_store_handles_failures():
    store = AnalysisJobStore()
    job_id = "job-456"
    store.create(job_id)

    store.mark_failed(job_id, "Failed to run")
    state = store.get(job_id)
    assert state is not None
    assert state.status == "failed"
    assert state.error == "Failed to run"
    assert state.progress == 1.0
    assert state.events
