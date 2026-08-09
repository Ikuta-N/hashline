# hashline

[![CI](https://github.com/Ikuta-N/hashline/actions/workflows/ci.yml/badge.svg)](https://github.com/Ikuta-N/hashline/actions/workflows/ci.yml)

Local-first micro-notes. Capture a thought in one line, tag it with inline
`#hashtags`, retrieve it later by tag or full-text search.

Everything lives in one SQLite file on your machine. Nothing is uploaded.

- **One line in, one note out.** No titles, no folders, no editor.
- **Tags come from the text.** Write `#sqlite` in the note and it is tagged.
- **Search that works in Japanese.** The FTS5 index uses the trigram
  tokenizer, so `全文検索` matches without any word segmentation.
- **Bulk import.** Point it at a directory of `.md` / `.txt` files and every
  line, or every Markdown section, becomes a note.

## Install

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Ikuta-N/hashline.git
cd hashline
uv sync
```

## Quick start

```bash
# capture
uv run hashline add "FTS5 の bm25 を調べた #sqlite #検索"

# read back, newest first
uv run hashline list
uv run hashline list --tag sqlite

# full-text search, best match first
uv run hashline search "bm25"

# what tags do I use?
uv run hashline tags

# import a pile of files
uv run hashline import ~/notes --mode heading --tag imported
```

Every command takes `--db PATH`. Without it the database is `$HASHLINE_DB`,
and failing that `~/.local/share/hashline/hashline.db`.

## Web UI

```bash
uv run uvicorn hashline.web.app:app --reload
# http://127.0.0.1:8000
```

Capture, tag filtering and search-as-you-type over the same database the CLI
uses; it honours `$HASHLINE_DB`. HTMX is vendored under
`src/hashline/web/static/`, so the page needs no CDN and works offline.

### `import`

```
hashline import PATH... [--mode line|heading] [--tag NAME] [--dry-run]
```

- `--mode line` (default) makes one note per non-blank line.
- `--mode heading` makes one note per Markdown section — a heading plus the
  lines under it. Text before the first heading is kept as its own note, and a
  `#` inside a fenced code block does not start a section.
- `--tag NAME` is repeatable and tags every note from that run. It is stored as
  a tag only: **the note body is left exactly as written**, so `--tag` names do
  not turn up in full-text search results.
- Directories are walked recursively for `.md`, `.markdown` and `.txt`. A file
  named directly on the command line is read whatever it is called.
- `--dry-run` reports what would be imported and writes nothing.

### Things worth knowing about search

- Results are ranked by BM25 and printed best-first. The score shown is
  `-bm25()`, so **higher is better**. On a very small database the score can
  legitimately be `0.00`: BM25 gives no weight to a term that appears in about
  half the notes.
- A trigram index cannot answer queries shorter than three characters, so those
  fall back to a substring scan, come back newest-first, and score `0.00`.
- The whole query is treated as literal text. `#`, `-`, `*` and `"` are
  searched for, not interpreted as operators.

## Development

```bash
uv sync --dev          # dev tools; note this does NOT install the ml extra
uv run ruff check .
uv run mypy src
uv run pytest
```

### Tests

```bash
uv run pytest                                     # the default suite
uv run pytest --cov=src --cov-report=term-missing # with coverage
uv run pytest -m slow                             # model-dependent tests only
```

Tests marked `slow` need an embedding model downloaded and are excluded by
default; CI never runs them. Import fixtures under `tests/fixtures/` are
synthetic — no real note directory is referenced anywhere in the test suite.

CI runs `ruff check .`, `mypy src` and `pytest --cov` on every push and pull
request.

## Layout

```
src/hashline/
  models.py     dataclasses shared by every layer
  tags.py       #tag extraction (pure functions)
  store.py      SQLite repository; no web or CLI dependency
  importer.py   documents -> note drafts (pure functions; no file I/O)
  cli.py        Typer adapter; owns all filesystem I/O
  schema.sql    tables, indexes, FTS5 index and its sync triggers
  web/app.py    FastAPI + HTMX adapter
  ml/search.py  ranking maths for semantic search (pure numpy)
tests/
```

The core (`models`, `tags`, `store`, `importer`) is plain Python over the
standard library plus numpy. The CLI and the web UI are thin adapters over it.

## Roadmap

Semantic search: retrieve notes by meaning, not just keyword, alongside the
FTS5 index.

The groundwork is in place. The `embeddings` table is already in the schema —
`(note_id, model)` keyed so several models can coexist — so adding it needs no
migration and no reimport. `hashline.ml.search` holds the ranking maths
(cosine similarity plus reciprocal rank fusion for blending the keyword and
semantic rankings); it is pure numpy, imports no model runtime, and is covered
by the default test suite.

What is left is the embedding backend. `sentence-transformers` will be an
optional `ml` extra imported inside functions, so the app keeps running fully
without it, with only semantic search disabled.

## License

MIT. See [LICENSE](LICENSE).
