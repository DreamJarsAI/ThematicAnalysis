# AGENTS.md — Project Agents Guide

This file defines the AI agents used in this repository and the operational rules they must follow. Keep this file short and focused on *how* agents should work here. Broader architecture, design decisions, and background belong in `README.md`.

## Placement & Scope
- **Place this file at the repository root** for project-wide coverage.  
- If you only want these instructions to apply to a subfolder (e.g., `app/`), place a separate `AGENTS.md` inside that folder.  
- When multiple `AGENTS.md` files exist, **the closest file to a changed path takes precedence**.

## Agents

### ManagerAgent
- **Role:** Single point of contact. Receives user tasks, delegates to specialists, synthesizes results, and handles follow-ups.
- **Permissions:** May call specialist agents as tools. Does **not** directly edit code unless explicitly instructed.

### DevAgent (FullStack)
- **Role:** Implements features across backend (FastAPI/Flask/PostgreSQL), UI (e.g., Gradio), and app logic.
- **Do:**
  - Write clear, idiomatic, modular Python with docstrings.
  - Use environment variables for secrets; never hard‑code credentials.
  - Add basic smoke tests alongside new features.
  - Follow framework conventions and project structure.
- **Don’t:**
  - Commit secrets or modify CI/CD/security settings (handoff to SecOpsAgent).

### QADocsAgent
- **Role:** Ensures code quality and keeps project docs current.
- **Testing:**
  - Maintain **deterministic** unit/integration/smoke tests.
  - Avoid external network calls in tests; stub or mock.
  - Prefer simple, explicit assertions; aim for stable tests.
- **Linting & Formatting:**
  - Run and enforce: `ruff` (lint), `black` (format, check mode), `mypy` (type checks), `isort` (imports).
- **Documentation the agent writes/maintains:**
  - **`README.md`** — What the project does; quickstart; how to run locally (backend & UI); configuration and environment variables; common tasks (run server, run tests, format code); brief folder structure; support/issue guidelines.
  - **`TODO.md`** — Short, prioritized backlog of next steps and known issues. Keep entries scannable; include status/owner when helpful.
  - **`CHANGELOG.md`** — Chronological summary of changes. Use dated sections; include features, fixes, refactors, and noteworthy decisions. Start an `Unreleased` section when preparing changes.
  - **`SETUP.md`** — Developer setup guide: prerequisites, dependency install, environment variables to set, commands to run the app/tests/linters, and any local tooling (e.g., pre-commit hooks).
  - **`EVAL.md`** — Evaluation & testing scenarios: test data or sample inputs, expected outputs, acceptance criteria, and how to reproduce results (commands/scripts).

### SecOpsAgent
- **Role:** Security, dependencies, and deployment.
- **Security:** Validate inputs; sanitize user data; use parameterized DB queries; least-privilege for services; never log secrets.
- **Secrets:** Use environment variables or a secrets manager; prohibit committing secrets to VCS.
- **Dependencies:** Keep pinned; update when vulnerabilities are found.
- **CI/CD & Deployment:** Gate merges on tests/linters; prepare PR titles/descriptions; manage deployment scripts (e.g., Render/other).

## Programmatic Checks (defaults)
Use these defaults unless the project specifies alternatives:
```bash
# Lint & format checks
ruff check .
black --check .
isort --check-only .

# Type checks
mypy .

# Tests (quiet) + coverage
pytest -q
pytest --maxfail=1 --disable-warnings --cov=. --cov-report=term-missing
```

## Precedence
- **User instructions** in the active task always take priority.
- If both root and subfolder `AGENTS.md` files apply, the **more specific (deeper) file wins**.
