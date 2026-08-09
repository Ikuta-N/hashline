# hashline

## What this is
A local-first micro-note app. Capture a thought in one line, tag it with
inline `#hashtags`, retrieve it later by tag or full-text search.

## Design constraints
- Core logic (`store.py`, `tags.py`) must be pure Python with no web/CLI
  dependency. All tests target the core.
- Storage: SQLite via stdlib `sqlite3`, with FTS5 for search. No ORM.
- Web layer is a thin adapter over the core. Same for the CLI.
- Full type hints. `mypy src` must pass.
- Python 3.12, managed with uv.

## Commit rules
- One commit per logical change. Never squash a whole feature into one commit.
- Conventional Commits style: feat:, fix:, test:, docs:, ci:, refactor:
- Do not commit `*.db`, `.env`, or anything under `.venv/`.

## Testing
- pytest. Core logic aims for high coverage; web layer gets smoke tests only.
- Every new public function ships with its test in the same commit.

## Roadmap
The app will gain semantic search: retrieve notes by meaning, not just
keyword, complementing the FTS5 index. Design the schema and store API
so this can be added without migrating data.

## Semantic search layer rules
- Lives entirely under `src/hashline/ml/`.
- `sentence-transformers` is an optional extra named `ml`. The app must
  start and work fully without it; only semantic search is disabled.
  Import it inside functions, never at module top level.
- `search.py` is pure numpy. It takes arrays and rank lists in and
  returns ranked ids out. It must NOT import torch or
  sentence-transformers. Its tests run in CI without the ml extra.
- Never download a model in a test that runs in CI. Model-dependent
  tests are marked `@pytest.mark.slow` and excluded by default.
