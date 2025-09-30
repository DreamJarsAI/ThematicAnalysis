# Developer Setup Guide

## Prerequisites
- Python 3.11+ (3.12 tested)
- `pip`
- OpenAI API key with GPT-4/5 access

## Bootstrap
```bash
git clone <your-fork-url>
cd thematic_analysis
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

## Environment
Set `OPENAI_API_KEY` via shell export or `.env`:
```bash
export OPENAI_API_KEY="sk-..."
```
Optional overrides: `FLASK_SECRET_KEY`, `UPLOAD_FOLDER`, `RESULTS_FOLDER`, `DEFAULT_MODEL`, `DEFAULT_CHUNK_OVERLAP`.

## NLTK Data
```bash
python - <<'PY'
import nltk
nltk.download("punkt")
PY
```

## Run the UI
```bash
python run.py
```
Visit `http://127.0.0.1:5000`.

## Quality Gates
```bash
ruff check .
black --check .
isort --check-only --profile black .
pytest
```

## Handy Commands
- Reformat: `black .`
- Sort imports: `isort --profile black .`
- Targeted tests: `pytest tests/test_agent_pipeline.py -k mocked`
- Clean temp folders: `rm -rf uploads results`
