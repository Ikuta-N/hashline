"""SQLite repository layer.

Depends only on the standard library plus ``models`` and ``tags``. It must not
import FastAPI, Typer, or anything else from an adapter layer.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Final

from hashline.models import Note, NoteDraft, SearchHit, TagCount
from hashline.tags import extract_tags, normalize_tag

SCHEMA_VERSION: Final = 1

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
        """Create the schema if it is missing. Safe to call on an existing database."""
        self._conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
        (current,) = self._conn.execute("PRAGMA user_version").fetchone()
        if current == 0:
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

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
    ) -> Note:
        """Store one note, linking both its inline #tags and ``extra_tags``."""
        draft = NoteDraft(
            body=body,
            created_at=created_at,
            source=source,
            extra_tags=tuple(extra_tags),
        )
        return self.add_notes([draft])[0]

    def add_notes(self, drafts: Iterable[NoteDraft]) -> list[Note]:
        """Store many notes in a single transaction.

        Bodies are stripped of surrounding whitespace; a blank body is rejected.
        """
        items = list(drafts)
        if not items:
            return []
        notes: list[Note] = []
        with self._conn:
            for draft in items:
                notes.append(self._insert(draft))
        return notes

    def _insert(self, draft: NoteDraft) -> Note:
        body = draft.body.strip()
        if not body:
            raise ValueError("note body must not be blank")
        created_at = draft.created_at if draft.created_at is not None else _utc_now()
        created_text = _to_text(created_at)
        cursor = self._conn.execute(
            "INSERT INTO notes (body, created_at, source) VALUES (?, ?, ?)",
            (body, created_text, draft.source),
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
            "SELECT id, body, created_at, source FROM notes WHERE id = ?",
            (note_id,),
        ).fetchone()
        return _to_note(row) if row is not None else None

    def list_notes(
        self, *, tag: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[Note]:
        """Return the timeline, newest first, optionally narrowed to one tag."""
        if tag is None:
            rows = self._conn.execute(
                "SELECT id, body, created_at, source FROM notes "
                "ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            return [_to_note(row) for row in rows]

        name = _filter_tag(tag)
        if name is None:
            return []
        rows = self._conn.execute(
            "SELECT n.id, n.body, n.created_at, n.source FROM notes n "
            "JOIN note_tags nt ON nt.note_id = n.id "
            "JOIN tags t ON t.id = nt.tag_id "
            "WHERE t.name = ? "
            "ORDER BY n.created_at DESC, n.id DESC LIMIT ? OFFSET ?",
            (name, limit, offset),
        ).fetchall()
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
            "SELECT n.id, n.body, n.created_at, n.source, "
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
            "SELECT n.id, n.body, n.created_at, n.source FROM notes n "
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
