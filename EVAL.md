# Evaluation Guide

## Smoke Test
1. Install deps: `pip install -r requirements.txt`
2. Set `OPENAI_API_KEY`.
3. Launch UI: `python run.py`
4. Upload 1–2 transcripts (one file per interviewee) and optional metadata.
5. Run analysis with default settings.
6. Verify generated downloads:
   - `<participant>_chunk_themes.txt`
   - `<participant>_participant_summary.txt`
   - `overall_study_themes.txt`
   - `<job_id>_results.zip`
7. Confirm temp upload folder is removed post-run.

## Automated Checks
```bash
pytest
```
Covers chunking, token guards, recursive synthesis, and prompt fidelity via mocked agent calls.

## Quality Bar
```bash
ruff check .
black --check .
isort --check-only --profile black .
```

## Acceptance Criteria
- Metadata token limits enforced (≤500 background, ≤500 questions, ≤2000 script, ≤3000 combined).
- Chunk sizes up to 8000 operate without exceeding model context when combined with metadata.
- Outputs remain UTF-8 `.txt` files containing both excerpts and synthesized themes.
- No secrets or PII committed; `.env` is ignored by git.
