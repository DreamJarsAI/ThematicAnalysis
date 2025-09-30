# Changelog

## [Unreleased]
- None yet.

## [2025-02-14]
- Replaced the legacy pipeline with an OpenAI Agents SDK workflow and sentence-aware chunking.
- Added a Flask UI for uploading transcripts, selecting models/chunk sizes, and downloading `.txt`/`.zip` outputs.
- Enforced metadata token caps (500 background, 500 questions, 2000 script, 3000 combined).
- Consolidated dependencies into `requirements*.txt`; removed packaging artifacts and setup script.
- Added developer documentation (`SETUP.md`, `EVAL.md`, `TODO.md`) and tightened git hygiene (.gitignore, temp cleanup).
