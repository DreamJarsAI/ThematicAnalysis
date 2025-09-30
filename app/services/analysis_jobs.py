from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from AutoThemeGenerator.core.storage import AnalysisStorageRecord
from AutoThemeGenerator.core.agent_pipeline import ProgressUpdate


@dataclass(slots=True)
class JobEvent:
    stage: str
    message: str
    progress: float
    current: int
    total: int


@dataclass(slots=True)
class AnalysisJobResult:
    record: AnalysisStorageRecord
    archive_path: Path


@dataclass(slots=True)
class JobState:
    status: str = "queued"
    progress: float = 0.0
    message: str = "Waiting to start..."
    events: List[JobEvent] = field(default_factory=list)
    result: Optional[AnalysisJobResult] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "progress": min(max(self.progress, 0.0), 1.0),
            "message": self.message,
            "events": [
                {
                    "stage": event.stage,
                    "message": event.message,
                    "progress": event.progress,
                    "current": event.current,
                    "total": event.total,
                }
                for event in self.events
            ],
            "error": self.error,
        }


class AnalysisJobStore:
    """Thread-safe in-memory registry for running analysis jobs."""

    def __init__(self, max_events: int = 100) -> None:
        self._jobs: Dict[str, JobState] = {}
        self._lock = threading.Lock()
        self._max_events = max_events

    def _trim_events(self, state: JobState) -> None:
        if len(state.events) > self._max_events:
            state.events = state.events[-self._max_events :]

    def create(self, job_id: str) -> None:
        with self._lock:
            self._jobs[job_id] = JobState()

    def exists(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._jobs

    def get(self, job_id: str) -> Optional[JobState]:
        with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                return None
            return JobState(
                status=state.status,
                progress=state.progress,
                message=state.message,
                events=list(state.events),
                result=state.result,
                error=state.error,
            )

    def update_from_progress(self, job_id: str, update: ProgressUpdate) -> None:
        with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                return
            if state.status == "completed":
                return
            state.status = "running"
            state.progress = max(state.progress, update.progress)
            state.message = update.message
            state.events.append(
                JobEvent(
                    stage=update.stage,
                    message=update.message,
                    progress=update.progress,
                    current=update.current,
                    total=update.total,
                )
            )
            self._trim_events(state)

    def mark_failed(self, job_id: str, message: str) -> None:
        with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                return
            state.status = "failed"
            state.progress = 1.0
            state.message = message
            state.error = message
            state.events.append(
                JobEvent(
                    stage="error",
                    message=message,
                    progress=state.progress,
                    current=0,
                    total=0,
                )
            )
            self._trim_events(state)

    def mark_completed(
        self,
        job_id: str,
        *,
        record: AnalysisStorageRecord,
        archive_path: Path,
    ) -> None:
        with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                return
            state.status = "completed"
            state.progress = 1.0
            state.message = "Analysis complete."
            state.result = AnalysisJobResult(record=record, archive_path=archive_path)
            state.events.append(
                JobEvent(
                    stage="completed",
                    message="Analysis complete.",
                    progress=1.0,
                    current=1,
                    total=1,
                )
            )
            self._trim_events(state)

    def result(self, job_id: str) -> Optional[AnalysisJobResult]:
        with self._lock:
            state = self._jobs.get(job_id)
            if not state or state.status != "completed" or not state.result:
                return None
            return state.result


__all__ = [
    "AnalysisJobStore",
    "AnalysisJobResult",
    "JobState",
]
