"""SQLite repository layer.

Depends only on the standard library plus ``models`` and ``tags``. It must not
import FastAPI, Typer, or anything else from an adapter layer.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Final

from hashline.models import Note, NoteDraft
from hashline.tags import extract_tags, normalize_tag

SCHEMA_VERSION: Final = 1

_SCHEMA_PATH: Final = Path(__file__).with_name("schema.sql")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _to_text(value: datetime) -> str:
    """Serialize a timestamp so that lexicographic order equals chronological order.

    Always UTC, always the same width, so ``ORDER BY created_at`` on the TEXT
    column is a correct time ordering.
    """
    aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return aware.astimezone(UTC).isoformat(timespec="microseconds")


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

    def list_notes(self, *, limit: int = 50, offset: int = 0) -> list[Note]:
        """Return the timeline, newest first."""
        rows = self._conn.execute(
            "SELECT id, body, created_at, source FROM notes "
            "ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [_to_note(row) for row in rows]
