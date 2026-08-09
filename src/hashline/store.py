"""SQLite repository layer.

Depends only on the standard library plus ``models`` and ``tags``. It must not
import FastAPI, Typer, or anything else from an adapter layer.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Final

from hashline.models import BibEntry, Context, Note, NoteDraft, SearchHit, TagCount
from hashline.tags import extract_tags, normalize_tag

SCHEMA_VERSION: Final = 3

_MIGRATIONS: Final[Mapping[int, str]] = {
    2: (
        "CREATE TABLE IF NOT EXISTS bib_entries ("
        "  citekey TEXT PRIMARY KEY,"
        "  tag TEXT NOT NULL,"
        "  entry_type TEXT NOT NULL,"
        "  title TEXT,"
        "  author TEXT,"
        "  year TEXT,"
        "  doi TEXT,"
        "  raw TEXT NOT NULL,"
        "  updated_at TEXT NOT NULL"
        ");"
        "CREATE TABLE IF NOT EXISTS app_state ("
        "  key TEXT PRIMARY KEY,"
        "  value TEXT NOT NULL"
        ");"
        "ALTER TABLE notes ADD COLUMN page TEXT;"
        "ALTER TABLE notes ADD COLUMN citekey TEXT REFERENCES bib_entries(citekey);"
        "CREATE INDEX IF NOT EXISTS idx_bib_entries_tag ON bib_entries(tag);"
        "CREATE INDEX IF NOT EXISTS idx_notes_citekey ON notes(citekey);"
    ),
    3: (
        "ALTER TABLE notes ADD COLUMN parent_id INTEGER "
        "REFERENCES notes(id) ON DELETE CASCADE;"
        "CREATE INDEX IF NOT EXISTS idx_notes_parent ON notes(parent_id);"
    ),
}


class SchemaVersionError(Exception):
    """The database was created by a newer version of the application."""

_SCHEMA_PATH: Final = Path(__file__).with_name("schema.sql")

_DB_ENV_VAR: Final = "HASHLINE_DB"


def default_db_path() -> Path:
    """Where notes live unless an adapter is told otherwise.

    ``$HASHLINE_DB`` wins; otherwise the XDG data directory. Every adapter
    resolves the database the same way because they all call this.
    """
    override = os.environ.get(_DB_ENV_VAR)
    if override:
        return Path(override)
    data_home = os.environ.get("XDG_DATA_HOME")
    root = Path(data_home) if data_home else Path.home() / ".local" / "share"
    return root / "hashline" / "hashline.db"

#: The trigram tokenizer indexes three-character sequences, so it cannot match
#: anything shorter than that.
_MIN_TRIGRAM_QUERY: Final = 3

#: The app_state row the pinned Context is stored under, as one JSON blob.
_CONTEXT_KEY: Final = "context"


def _as_phrase(text: str) -> str:
    """Wrap user input as a single FTS5 phrase.

    Quoting the whole query keeps characters like ``#``, ``-`` and ``*`` from
    being read as FTS5 operators and turning a search into a syntax error. With
    the trigram tokenizer a phrase query is a substring query, which is exactly
    what a note search should mean.
    """
    return '"' + text.replace('"', '""') + '"'


def _as_like_pattern(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _to_text(value: datetime) -> str:
    """Serialize a timestamp so that lexicographic order equals chronological order.

    Always UTC, always the same width, so ``ORDER BY created_at`` on the TEXT
    column is a correct time ordering.
    """
    aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return aware.astimezone(UTC).isoformat(timespec="microseconds")


def _filter_tag(tag: str) -> str | None:
    """Normalize a tag used as a read filter, or ``None`` if it cannot be one.

    Read paths stay lenient: a tag no note could ever carry simply matches
    nothing, which saves every caller from guarding the query.
    """
    try:
        return normalize_tag(tag)
    except ValueError:
        return None


def _to_note(row: sqlite3.Row) -> Note:
    return Note(
        id=row["id"],
        body=row["body"],
        created_at=datetime.fromisoformat(row["created_at"]),
        source=row["source"],
        page=row["page"],
        citekey=row["citekey"],
        parent_id=row["parent_id"],
    )


class Store:
    """A note repository backed by a single SQLite database."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._conn.row_factory = sqlite3.Row

    @classmethod
    def open(cls, path: str | Path) -> Store:
        """Open (creating if needed) the database at ``path`` and apply the schema.

        ``":memory:"`` is accepted and is what the tests use.
        """
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        # Per-connection and off by default: without it the ON DELETE CASCADE
        # clauses in the schema do nothing at all.
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        store = cls(conn)
        store.init_schema()
        return store

    def init_schema(self) -> None:
        """Create or migrate the schema.

        Fresh databases get everything from ``schema.sql`` and are stamped with
        the current :data:`SCHEMA_VERSION`.  Existing databases whose version
        is behind get each intermediate migration applied in order, each in its
        own transaction.  A database from a *newer* version raises
        :class:`SchemaVersionError` so we never silently damage it.

        ``schema.sql`` is only run on databases that have never been versioned
        (``user_version == 0``), because it references the latest column set
        and would fail on an older schema that is missing columns.
        """
        (current,) = self._conn.execute("PRAGMA user_version").fetchone()

        if current == 0:
            # Brand-new database: schema.sql contains the full current schema.
            self._conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            return

        if current > SCHEMA_VERSION:
            raise SchemaVersionError(
                f"database is version {current}, but this build only "
                f"knows up to version {SCHEMA_VERSION}"
            )

        for version in range(current + 1, SCHEMA_VERSION + 1):
            sql = _MIGRATIONS[version]
            self._conn.executescript(sql)
            self._conn.execute(f"PRAGMA user_version = {version}")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # --- writing ---------------------------------------------------------

    def add_note(
        self,
        body: str,
        *,
        created_at: datetime | None = None,
        source: str | None = None,
        extra_tags: Sequence[str] = (),
        page: str | None = None,
        citekey: str | None = None,
        parent_id: int | None = None,
    ) -> Note:
        """Store one note, linking both its inline #tags and ``extra_tags``."""
        draft = NoteDraft(
            body=body,
            created_at=created_at,
            source=source,
            extra_tags=tuple(extra_tags),
            page=page,
            citekey=citekey,
        )
        with self._conn:
            return self._insert(draft, parent_id=parent_id)

    def add_notes(self, drafts: Iterable[NoteDraft]) -> list[Note]:
        """Store many notes in a single transaction.

        Bodies are stripped of surrounding whitespace; a blank body is rejected.
        """
        items = list(drafts)
        if not items:
            return []
        notes: list[Note] = []
        ids: list[int] = []
        with self._conn:
            for i, draft in enumerate(items):
                parent_id = None
                if draft.parent_index is not None:
                    if draft.parent_index >= i:
                        raise ValueError(
                            f"forward parent_index {draft.parent_index} at position {i}"
                        )
                    parent_id = ids[draft.parent_index]
                note = self._insert(draft, parent_id=parent_id)
                ids.append(note.id)
                notes.append(note)
        return notes

    def _insert(self, draft: NoteDraft, parent_id: int | None = None) -> Note:
        if parent_id is not None:
            (exists,) = self._conn.execute(
                "SELECT count(*) FROM notes WHERE id = ?", (parent_id,)
            ).fetchone()
            if not exists:
                raise ValueError(f"parent_id {parent_id} does not exist")
        body = draft.body.strip()
        if not body:
            raise ValueError("note body must not be blank")
        created_at = draft.created_at if draft.created_at is not None else _utc_now()
        created_text = _to_text(created_at)
        # Blank/whitespace-only page normalises to None.
        page = draft.page.strip() if draft.page else None
        page = page if page else None
        cursor = self._conn.execute(
            "INSERT INTO notes (body, created_at, source, page, citekey, parent_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (body, created_text, draft.source, page, draft.citekey, parent_id),
        )
        note_id = cursor.lastrowid
        if note_id is None:  # pragma: no cover - sqlite always reports it here
            raise RuntimeError("sqlite did not report a row id for the new note")
        self._link_tags(note_id, self._tag_names(body, draft.extra_tags))
        return Note(
            id=note_id,
            body=body,
            created_at=datetime.fromisoformat(created_text),
            source=draft.source,
            page=page,
            citekey=draft.citekey,
            parent_id=parent_id,
        )

    @staticmethod
    def _tag_names(body: str, extra_tags: Sequence[str]) -> list[str]:
        names = extract_tags(body)
        for raw in extra_tags:
            name = normalize_tag(raw)
            if name not in names:
                names.append(name)
        return names

    def _link_tags(self, note_id: int, names: Sequence[str]) -> None:
        for name in names:
            self._conn.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (name,))
            self._conn.execute(
                "INSERT OR IGNORE INTO note_tags (note_id, tag_id) "
                "VALUES (?, (SELECT id FROM tags WHERE name = ?))",
                (note_id, name),
            )

    def delete_note(self, note_id: int) -> bool:
        """Delete a note. Returns whether it existed."""
        with self._conn:
            cursor = self._conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        return cursor.rowcount > 0

    # --- reading ---------------------------------------------------------

    def get_note(self, note_id: int) -> Note | None:
        row = self._conn.execute(
            "SELECT id, body, created_at, source, page, citekey, parent_id "
            "FROM notes WHERE id = ?",
            (note_id,),
        ).fetchone()
        return _to_note(row) if row is not None else None

    def list_notes(
        self,
        *,
        roots_only: bool = False,
        tag: str | None = None,
        citekey: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Note]:
        """Return the timeline, newest first.

        Optionally narrowed to one tag or citekey.
        """
        where_clauses = []
        params: list[object] = []
        
        if roots_only:
            where_clauses.append("n.parent_id IS NULL")

        if citekey is not None:
            where_clauses.append("n.citekey = ?")
            params.append(citekey)

        tag_name: str | None = None
        if tag is not None:
            tag_name = _filter_tag(tag)
            if tag_name is None:
                return []
            where_clauses.append("t.name = ?")
            params.append(tag_name)

        sql = (
            "SELECT n.id, n.body, n.created_at, n.source, n.page, n.citekey, "
            "n.parent_id FROM notes n "
        )
        if tag_name is not None:
            sql += "JOIN note_tags nt ON nt.note_id = n.id "
            sql += "JOIN tags t ON t.id = nt.tag_id "
            
        if where_clauses:
            sql += "WHERE " + " AND ".join(where_clauses) + " "
            
        sql += "ORDER BY n.created_at DESC, n.id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self._conn.execute(sql, params).fetchall()
        return [_to_note(row) for row in rows]

    def count_notes(self, *, tag: str | None = None) -> int:
        """Count notes, optionally narrowed to one tag."""
        if tag is None:
            (count,) = self._conn.execute("SELECT count(*) FROM notes").fetchone()
            return int(count)

        name = _filter_tag(tag)
        if name is None:
            return 0
        (count,) = self._conn.execute(
            "SELECT count(*) FROM note_tags nt "
            "JOIN tags t ON t.id = nt.tag_id WHERE t.name = ?",
            (name,),
        ).fetchone()
        return int(count)

    def list_tags(self, *, limit: int | None = None) -> list[TagCount]:
        """Return tags that are in use, most used first, ties broken by name."""
        sql = (
            "SELECT t.name AS name, count(nt.note_id) AS count FROM tags t "
            "JOIN note_tags nt ON nt.tag_id = t.id "
            "GROUP BY t.id ORDER BY count DESC, t.name ASC"
        )
        params: tuple[int, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        rows = self._conn.execute(sql, params).fetchall()
        return [TagCount(name=row["name"], count=row["count"]) for row in rows]

    def replies_to(self, note_id: int) -> list[Note]:
        """Return direct children of note_id, ordered by created_at then id."""
        rows = self._conn.execute(
            "SELECT id, body, created_at, source, page, citekey, parent_id "
            "FROM notes WHERE parent_id = ? ORDER BY created_at, id",
            (note_id,)
        ).fetchall()
        return [_to_note(row) for row in rows]

    def thread(self, note_id: int) -> list[Note]:
        """Return the note and all its descendants, depth-first.
        Siblings ordered by created_at then id.
        """
        # Recursive CTE for depth-first traversal.
        # SQLite recursive CTE processes depth-first if we sort properly.
        # But native depth-first is easier by maintaining a sort key.
        sql = """
            WITH RECURSIVE
              thread_tree(id, body, created_at, source, page, citekey, parent_id,
                          sort_key) AS (
                SELECT id, body, created_at, source, page, citekey, parent_id,
                       printf('%s-%08X', created_at, id)
                FROM notes WHERE id = ?
                UNION ALL
                SELECT n.id, n.body, n.created_at, n.source, n.page, n.citekey,
                       n.parent_id,
                       t.sort_key || '/' || printf('%s-%08X', n.created_at, n.id)
                FROM notes n
                JOIN thread_tree t ON n.parent_id = t.id
              )
            SELECT * FROM thread_tree ORDER BY sort_key;
        """
        rows = self._conn.execute(sql, (note_id,)).fetchall()
        if not rows:
            raise ValueError(f"note_id {note_id} does not exist")
        return [_to_note(row) for row in rows]

    # --- embeddings (semantic search) ------------------------------------

    def upsert_embedding(
        self,
        note_id: int,
        *,
        model: str,
        vector: bytes,
        dim: int,
        updated_at: datetime | None = None,
    ) -> None:
        """Store or replace one note's vector for one model."""
        stamp = _to_text(updated_at if updated_at is not None else _utc_now())
        with self._conn:
            self._conn.execute(
                "INSERT INTO embeddings (note_id, model, dim, vec, updated_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(note_id, model) DO UPDATE SET "
                "dim = excluded.dim, vec = excluded.vec, "
                "updated_at = excluded.updated_at",
                (note_id, model, dim, vector, stamp),
            )

    def notes_without_embedding(
        self, model: str, *, limit: int | None = None
    ) -> list[Note]:
        """Return notes this model has not embedded yet, oldest id first."""
        sql = (
            "SELECT n.id, n.body, n.created_at, n.source, n.page, n.citekey, "
            "n.parent_id "
            "FROM notes n "
            "LEFT JOIN embeddings e ON e.note_id = n.id AND e.model = ? "
            "WHERE e.note_id IS NULL ORDER BY n.id"
        )
        params: list[object] = [model]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [_to_note(row) for row in self._conn.execute(sql, params).fetchall()]

    def iter_embeddings(self, model: str) -> Iterator[tuple[int, bytes]]:
        """Yield ``(note_id, vector)`` for one model, ordered by note id."""
        cursor = self._conn.execute(
            "SELECT note_id, vec FROM embeddings WHERE model = ? ORDER BY note_id",
            (model,),
        )
        for row in cursor:
            yield row["note_id"], row["vec"]

    def search_notes(
        self, query: str, *, tag: str | None = None, limit: int = 50
    ) -> list[SearchHit]:
        """Full-text search, best match first.

        ``SearchHit.score`` is ``-bm25()``: SQLite returns a negative relevance
        where smaller is better, so the sign is flipped and **higher means a
        better match**.

        The index uses the trigram tokenizer, which cannot answer queries
        shorter than three characters. Those fall back to a substring scan and
        come back newest-first with a score of ``0.0``.
        """
        text = query.strip()
        if not text:
            return []
        name: str | None = None
        if tag is not None:
            name = _filter_tag(tag)
            if name is None:
                return []
        if len(text) < _MIN_TRIGRAM_QUERY:
            return self._search_by_substring(text, name, limit)
        return self._search_by_rank(text, name, limit)

    def _search_by_rank(
        self, text: str, tag: str | None, limit: int
    ) -> list[SearchHit]:
        sql = (
            "SELECT n.id, n.body, n.created_at, n.source, n.page, n.citekey, "
            "n.parent_id, "
            "-bm25(notes_fts) AS score "
            "FROM notes_fts JOIN notes n ON n.id = notes_fts.rowid "
            "WHERE notes_fts MATCH ?"
        )
        params: list[object] = [_as_phrase(text)]
        if tag is not None:
            sql += " AND n.id IN (SELECT nt.note_id FROM note_tags nt "
            sql += "JOIN tags t ON t.id = nt.tag_id WHERE t.name = ?)"
            params.append(tag)
        sql += " ORDER BY score DESC, n.created_at DESC, n.id DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [SearchHit(note=_to_note(row), score=row["score"]) for row in rows]

    def _search_by_substring(
        self, text: str, tag: str | None, limit: int
    ) -> list[SearchHit]:
        sql = (
            "SELECT n.id, n.body, n.created_at, n.source, n.page, n.citekey, "
            "n.parent_id "
            "FROM notes n "
            "WHERE n.body LIKE ? ESCAPE '\\'"
        )
        params: list[object] = [_as_like_pattern(text)]
        if tag is not None:
            sql += " AND n.id IN (SELECT nt.note_id FROM note_tags nt "
            sql += "JOIN tags t ON t.id = nt.tag_id WHERE t.name = ?)"
            params.append(tag)
        sql += " ORDER BY n.created_at DESC, n.id DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [SearchHit(note=_to_note(row), score=0.0) for row in rows]

    def tags_for_note(self, note_id: int) -> list[str]:
        """Return the tags linked to one note, in alphabetical order."""
        rows = self._conn.execute(
            "SELECT t.name FROM tags t "
            "JOIN note_tags nt ON nt.tag_id = t.id "
            "WHERE nt.note_id = ? ORDER BY t.name",
            (note_id,),
        ).fetchall()
        return [row["name"] for row in rows]

    # --- bibliography ----------------------------------------------------

    def upsert_bib_entries(
        self, entries: Iterable[BibEntry], *, replace: bool = False
    ) -> int:
        """Insert or update bibliography entries.

        Uses ``ON CONFLICT(citekey) DO UPDATE`` so re-importing a library
        refreshes rather than fails.  With ``replace=True`` the existing
        library is cleared first, in the same transaction as the insert, so a
        re-import that dropped entries does not leave the old ones behind.
        Returns how many entries were written.
        """
        stamp = _to_text(_utc_now())
        count = 0
        with self._conn:
            if replace:
                self._conn.execute("DELETE FROM bib_entries")
            for entry in entries:
                self._conn.execute(
                    "INSERT INTO bib_entries "
                    "(citekey, tag, entry_type, title, author, year, doi, "
                    "raw, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(citekey) DO UPDATE SET "
                    "tag = excluded.tag, "
                    "entry_type = excluded.entry_type, "
                    "title = excluded.title, "
                    "author = excluded.author, "
                    "year = excluded.year, "
                    "doi = excluded.doi, "
                    "raw = excluded.raw, "
                    "updated_at = excluded.updated_at",
                    (
                        entry.citekey,
                        entry.tag,
                        entry.entry_type,
                        entry.title,
                        entry.author,
                        entry.year,
                        entry.doi,
                        entry.raw,
                        stamp,
                    ),
                )
                count += 1
        return count

    def get_bib_entry(self, citekey: str) -> BibEntry | None:
        """Return a single bibliography entry, or ``None`` if unknown."""
        row = self._conn.execute(
            "SELECT citekey, tag, entry_type, title, author, year, doi, raw "
            "FROM bib_entries WHERE citekey = ?",
            (citekey,),
        ).fetchone()
        if row is None:
            return None
        return BibEntry(
            citekey=row["citekey"],
            tag=row["tag"],
            entry_type=row["entry_type"],
            title=row["title"],
            author=row["author"],
            year=row["year"],
            doi=row["doi"],
            raw=row["raw"],
        )

    def list_bib_entries(
        self, *, limit: int | None = None
    ) -> list[BibEntry]:
        """Return all bibliography entries, ordered by citekey."""
        sql = (
            "SELECT citekey, tag, entry_type, title, author, year, doi, raw "
            "FROM bib_entries ORDER BY citekey"
        )
        params: tuple[int, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        rows = self._conn.execute(sql, params).fetchall()
        return [
            BibEntry(
                citekey=row["citekey"],
                tag=row["tag"],
                entry_type=row["entry_type"],
                title=row["title"],
                author=row["author"],
                year=row["year"],
                doi=row["doi"],
                raw=row["raw"],
            )
            for row in rows
        ]

    # --- context -----------------------------------------------------------

    def get_context(self) -> Context:
        """Return the pinned context, or an empty ``Context`` when none is set."""
        row = self._conn.execute(
            "SELECT value FROM app_state WHERE key = ?", (_CONTEXT_KEY,)
        ).fetchone()
        if row is None:
            return Context()
        data = json.loads(row["value"])
        return Context(tags=tuple(data["tags"]), citekey=data["citekey"])

    def set_context(self, context: Context) -> None:
        """Persist ``context`` as the pinned context, replacing any previous one.

        Tags are normalized on the way in through ``tags.normalize_tag``, so a
        tag that could never be a valid ``#tag`` fails here -- where the user
        typed it -- rather than later, silently, when a note tries to use it.
        """
        payload = json.dumps(
            {
                "tags": [normalize_tag(tag) for tag in context.tags],
                "citekey": context.citekey,
            }
        )
        with self._conn:
            self._conn.execute(
                "INSERT INTO app_state (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (_CONTEXT_KEY, payload),
            )

    def clear_context(self) -> None:
        """Unpin the context, if one is set."""
        with self._conn:
            self._conn.execute(
                "DELETE FROM app_state WHERE key = ?", (_CONTEXT_KEY,)
            )

    def add_note_with_context(
        self,
        body: str,
        *,
        page: str | None = None,
        extra_tags: Sequence[str] = (),
    ) -> Note:
        """Store one note under the pinned context.

        Composes the tag list from three sources: the body's own inline
        ``#tags`` (``add_note`` -> ``_insert`` already handles those), the
        pinned context's tags, and, when the context has a citekey, that
        entry's ``bib_entries.tag`` -- so a note written while "reading" a
        work is tagged with that work automatically. ``page`` and the
        citekey come from the context too.

        ``add_note`` itself deliberately never reads the context. Keeping
        implicit state out of the repository's core write path means every
        other caller -- ``add_notes``, the importer -- stays predictable and
        testable without a context to set up first. This method is the one
        place the composition happens, and both the CLI and the web adapter
        call it so pinning behaves the same from either.
        """
        context = self.get_context()
        tags = list(context.tags)
        tags.extend(extra_tags)
        if context.citekey is not None:
            entry = self.get_bib_entry(context.citekey)
            if entry is not None:
                tags.append(entry.tag)
        return self.add_note(
            body, page=page, citekey=context.citekey, extra_tags=tags
        )
