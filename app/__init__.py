from __future__ import annotations

from pathlib import Path

from flask import Flask

from .config import Config
from .extensions import ensure_nltk_data
from .services.analysis_jobs import AnalysisJobStore


def create_app(config_class: type[Config] | None = None) -> Flask:
    """Application factory for the thematic analysis UI."""

    app = Flask(__name__)
    app.config.from_object(config_class or Config())

    ensure_nltk_data()

    app.extensions["analysis_jobs"] = AnalysisJobStore()

    _prepare_directories(app)
    _register_blueprints(app)

    return app


def _prepare_directories(app: Flask) -> None:
    upload_path = Path(app.config["UPLOAD_FOLDER"])
    results_path = Path(app.config["RESULTS_FOLDER"])

    upload_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)


def _register_blueprints(app: Flask) -> None:
    from .blueprints.main.routes import main_bp

    app.register_blueprint(main_bp)


__all__ = ["create_app", "Config"]
