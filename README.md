# hashline

[![CI](https://github.com/Ikuta-N/hashline/actions/workflows/ci.yml/badge.svg)](https://github.com/Ikuta-N/hashline/actions/workflows/ci.yml)

Local-first micro-notes. Capture a thought in one line, tag it with inline
`#hashtags`, retrieve it later by tag or full-text search.

Everything lives in one SQLite file on your machine. Nothing is uploaded.

![A hashline session: capture, list, keyword search, then a semantic search
finding a note that shares no characters with the query](docs/cli-session.png)

- **One line in, one note out.** No titles, no folders, no editor.
- **Tags come from the text.** Write `#sqlite` in the note and it is tagged.
- **Search that works in Japanese.** The FTS5 index uses the trigram
  tokenizer, so `全文検索` matches without any word segmentation.
- **Bulk import.** Point it at a directory of `.md` / `.txt` files and every
  line, or every Markdown section, becomes a note.
- **Reading mode.** Pin a BibTeX reference while reading; captured notes automatically carry the citekey tag and page number.
- **Replies and threads.** Reply to notes to build threads and structured note trees.
- **Markdown round-trip.** Export note trees as indented Markdown outlines, or import outline files back into notes.

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

## Working with notes

Beyond the commands in the quick start, the CLI can bulk-import files, run a
citation-aware reading workflow, keep tags pinned across captures, and build
reply threads.

### `import`

```
hashline import PATH... [--mode line|heading|outline] [--tag NAME] [--dry-run]
```

- `--mode line` (default) makes one note per non-blank line.
- `--mode heading` makes one note per Markdown section — a heading plus the
  lines under it. Text before the first heading is kept as its own note, and a
  `#` inside a fenced code block does not start a section.
- `--mode outline` parses indented bullet lists into note trees, matching the format produced by `hashline export`.
- `--tag NAME` is repeatable and tags every note from that run. It is stored as
  a tag only: **the note body is left exactly as written**, so `--tag` names do
  not turn up in full-text search results.
- Directories are walked recursively for `.md`, `.markdown` and `.txt`. A file
  named directly on the command line is read whatever it is called.
- `--dry-run` reports what would be imported and writes nothing.

### Reading notes

Pin a bibliography entry while reading to automatically tag and cite captured notes:

```bash
# import bibliography entries from a .bib file
uv run hashline bib import library.bib [--replace]
uv run hashline bib list
uv run hashline bib show smith2020

# pin a work to start reading mode
uv run hashline read start smith2020 [--tag NAME]
uv run hashline read status

# capture notes with page numbers (carried into context automatically)
uv run hashline add "Trigram indexing is fast for short queries #notes" --page 12-15

# filter notes by citation key
uv run hashline list --citekey smith2020

# stop reading mode
uv run hashline read stop
```

Things worth knowing:
- `read status` reports the pinned **work**. If only plain tags are pinned via `pin`, `read status` says nothing is pinned — use `pin --show` for those.
- A note captured under a reading context receives both the reading tag (`#reading` by default, or custom `--tag NAME`) and the work's citekey tag (e.g. `#smith2020`).
- `--page` is a free-form string (`"42"`, `"12-15"`, `"xii"`, `"第3章"`) and there is therefore no page ordering or range search.
- `--page` without a pinned work is an error, not a silent no-op.
- LaTeX escapes in `.bib` values are stored as written; `{\"o}` is not turned into `ö`.
- The BibTeX parser skips an entry it cannot read and reports it, rather than failing the whole import.

### Pinned tags

Pin tags across multiple `add` invocations without repeating them:

```bash
# pin tags for subsequent notes
uv run hashline pin research sqlite

# check currently pinned tags
uv run hashline pin --show

# capture a note (receives #research and #sqlite tags automatically)
uv run hashline add "Investigating indexing performance"

# clear pinned tags
uv run hashline pin --clear
```

Like `import --tag`, `hashline pin` tags without touching the body, so pinned tag names do not turn up in full-text search results.

### Replies and outlines

Notes can reply to existing notes, forming threads and tree structures:

```bash
# reply to a note ID to build a thread
uv run hashline reply 1 "Sub-point about implementation"

# view a thread indented by depth
uv run hashline thread 1

# list timeline hiding replies
uv run hashline list --roots-only

# delete a note (refuses if it has replies unless --recursive is passed)
uv run hashline rm 1 [--recursive]

# export notes as a Markdown outline
uv run hashline export [--tag X] [--citekey Y] [--root ID] [-o FILE]

# import an outline file back into note trees
uv run hashline import PATH --mode outline
```

Things worth knowing:
- `rm` refuses a note that has replies; `--recursive` removes the thread.
- `export` promotes a note whose parent is outside the selection to a root, so filtering never makes replies disappear.
- There is no reparenting: restructuring means deleting and re-entering.

### Search

- Results are ranked by BM25 and printed best-first. The score shown is
  `-bm25()`, so **higher is better**. On a very small database the score can
  legitimately be `0.00`: BM25 gives no weight to a term that appears in about
  half the notes.
- A trigram index cannot answer queries shorter than three characters, so those
  fall back to a substring scan, come back newest-first, and score `0.00`.
- The whole query is treated as literal text. `#`, `-`, `*` and `"` are
  searched for, not interpreted as operators.
- `--semantic` blends in a ranking by meaning; see
  [Semantic search](#semantic-search). It needs the `ml` extra and a
  `hashline index` pass, and says so when either is missing.

## Web UI

```bash
uv run uvicorn hashline.web.app:app --reload
# http://127.0.0.1:8000
```

![The hashline web UI: the pinned-context strip, the composer, tag chips, and
a timeline with a reply nested under its parent](docs/web-ui.png)

Capture, tag filtering and search-as-you-type over the same database the CLI
uses; it honours `$HASHLINE_DB`. HTMX is vendored under
`src/hashline/web/static/`, so the page needs no CDN and works offline. The
vendored build is htmx 2.0.4 (`htmx.min.js`, ~50 KB), licensed 0BSD, which
carries no attribution requirement — update the file by hand when a new
version is needed.

### Feature Parity

The Web UI implements equivalent functionality to the CLI commands:

| CLI command | Web UI route |
|---|---|
| `hashline list` | `GET /` |
| `hashline add` | `POST /notes` |
| `hashline rm` | `POST /notes/{id}/delete` |
| `hashline reply` | `POST /notes` (with parent_id) |
| `hashline thread` | `GET /` (with root filter) |
| `hashline search` | `GET /` (with q) |
| `hashline pin` | `POST /context/pin`, `POST /context/clear_tags` |
| `hashline read` | `POST /context/read`, `POST /context/clear_read` |
| `hashline bib list` | `GET /bib` |
| `hashline bib show` | `GET /bib/{citekey}` |
| `hashline import` | `POST /import` (also supports browser uploads) |
| `hashline bib import` | `POST /bib/import` (also supports browser uploads) |
| `hashline export` | `GET /export`, `GET /export/download` |
| `hashline index`, `search --semantic` | *CLI only* |

One deliberate difference: `hashline read start` replaces the pinned tags, while `POST /context/read` adds the reading tag to them. The context strip shows the pinned tags and the pinned work side by side with a clear button each, so starting a read there should not empty the column next to it.

> **Security Note:** The `path` field in both `/import` and `/bib/import` reads files directly from the local filesystem on the machine running the server. Do not expose this web app to a network. All state-changing routes also reject a POST whose `Origin` header does not match the server's own host, so a form on another site left open in the same browser cannot delete notes or replace the bibliography; a request with no `Origin` header (curl, the CLI) is still allowed through.

## Semantic search

Retrieve notes by meaning, not only by the characters in them. Optional: the
app installs, starts and works fully without it.

```bash
uv sync --extra ml            # torch (CPU build) and sentence-transformers
uv run hashline index         # embed every note not embedded yet
uv run hashline search 睡眠 --semantic
```

```
0.0164      1  2026-08-09 22:18  昨日は寝不足で、朝から頭が回らなかった  [日記]
0.0161      3  2026-08-09 22:18  夜ふかしをやめたい                      [日記]
```

Neither note contains 睡眠, so `hashline search 睡眠` on its own finds nothing.

`hashline index` walks `notes_without_embedding`, so re-running it costs
nothing; `--rebuild` re-embeds everything and `--limit` stops early. A search
that finds notes added since the last index says how many are missing rather
than leaving a short result list as the only clue.

### How the two rankings are combined

BM25 and cosine similarity are on unrelated scales, so the **ranks** are fused,
not the scores: each list contributes `1 / (60 + rank)` per note
(reciprocal rank fusion). No normalization constant to tune, and a third
ranker could be added without revisiting the first two. Both sides contribute
their top 100 regardless of `--limit`, so a note that both rankers place 25th
is not beaten by one that a single ranker happened to put 20th.

### How vectors are stored

The `embeddings` table has been in the schema since the first release, so this needed no migration and no reimport.

- Little-endian `float32`, fixed explicitly, so a `.db` file stays portable between machines.
- `dim` lives in its own column, not a header inside the BLOB.
- Vectors are L2-normalized on write.
- `embeddings.model` records the prefix convention (`intfloat/multilingual-e5-small+query`), not just the model name — e5-small is 384-wide exactly like the MiniLM model it replaced, so no dimension check could catch the two being mixed.

### What runs where

- `ml/search.py` is pure numpy and imports no model runtime.
- `ml/embed.py` imports the backend inside functions.
- `ml/protocols.py` defines the `Embedder` protocol both sides share.
- The CLI imports both only inside the two commands that need them; `--semantic` is still CLI-only, not in the web UI.

## Upgrading

This release carries the project's first schema migrations. An existing database is upgraded in place the next time it is opened, and going back to an older version of `hashline` afterwards will not work.

Before upgrading, it is recommended to copy your database file to back it up. The database location is specified by `$HASHLINE_DB`, or defaults to `~/.local/share/hashline/hashline.db`:

```bash
cp "${HASHLINE_DB:-$HOME/.local/share/hashline/hashline.db}" ~/hashline.db.bak
```

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
  bib.py        BibTeX parsing (pure functions; no file I/O)
  outline.py    Markdown outline building and rendering (pure functions)
  cli.py        Typer adapter; owns all filesystem I/O
  schema.sql    tables, indexes, FTS5 index and its sync triggers
  web/app.py       FastAPI + HTMX adapter
  ml/search.py     ranking maths for semantic search (pure numpy)
  ml/embed.py      embedding backend behind the optional `ml` extra
  ml/protocols.py  the Embedder protocol; numpy and nothing else
tests/
```

The core (`models`, `tags`, `store`, `importer`, `bib`, `outline`) is plain Python
over the standard library plus numpy. The CLI and the web UI are thin adapters over it.

## License

MIT. See [LICENSE](LICENSE).

