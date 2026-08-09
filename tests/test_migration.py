"""Tests for schema migration."""

import sqlite3

import pytest

from hashline.store import SCHEMA_VERSION, SchemaVersionError, Store

#: The v1 DDL — the schema as it was before the page/citekey/bib_entries
#: additions.  Used to build a realistic v1 database inside tests.
_V1_DDL = """\
CREATE TABLE IF NOT EXISTS notes (
  id         INTEGER PRIMARY KEY,
  body       TEXT NOT NULL,
  created_at TEXT NOT NULL,
  source     TEXT
);

CREATE TABLE IF NOT EXISTS tags (
  id   INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS note_tags (
  note_id INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
  tag_id  INTEGER NOT NULL REFERENCES tags(id)  ON DELETE CASCADE,
  PRIMARY KEY (note_id, tag_id)
);

CREATE TABLE IF NOT EXISTS embeddings (
  note_id    INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
  model      TEXT    NOT NULL,
  dim        INTEGER NOT NULL,
  vec        BLOB    NOT NULL,
  updated_at TEXT    NOT NULL,
  PRIMARY KEY (note_id, model)
);

CREATE INDEX IF NOT EXISTS idx_notes_created_at ON notes(created_at DESC, id DESC);
CREATE INDEX IF NOT EXISTS idx_note_tags_tag    ON note_tags(tag_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model);

CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
  body,
  content='notes',
  content_rowid='id',
  tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN
  INSERT INTO notes_fts(rowid, body) VALUES (new.id, new.body);
END;

CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN
  INSERT INTO notes_fts(notes_fts, rowid, body)
    VALUES ('delete', old.id, old.body);
END;

CREATE TRIGGER IF NOT EXISTS notes_au AFTER UPDATE ON notes BEGIN
  INSERT INTO notes_fts(notes_fts, rowid, body)
    VALUES ('delete', old.id, old.body);
  INSERT INTO notes_fts(rowid, body) VALUES (new.id, new.body);
END;
"""


def _build_v1_db(conn: sqlite3.Connection) -> None:
    """Set up a v1 database with one note, stamp it, and close cleanly."""
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_V1_DDL)
    conn.execute("PRAGMA user_version = 1")
    conn.execute(
        "INSERT INTO notes (body, created_at) "
        "VALUES ('hello #test', '2026-01-01T00:00:00.000000+00:00')"
    )
    # The trigger should have populated the FTS index.
    conn.commit()


class TestMigration:
    def test_v1_database_gains_new_columns(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        db = tmp_path / "test.db"  # type: ignore[union-attr]
        conn = sqlite3.connect(db)
        _build_v1_db(conn)
        conn.close()

        with Store.open(db) as store:
            cols = {
                row[1]
                for row in store._conn.execute(
                    "PRAGMA table_info(notes)"
                ).fetchall()
            }
            assert "page" in cols
            assert "citekey" in cols
            assert "parent_id" in cols

    def test_pre_existing_note_survives_migration(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        db = tmp_path / "test.db"  # type: ignore[union-attr]
        conn = sqlite3.connect(db)
        _build_v1_db(conn)
        conn.close()

        with Store.open(db) as store:
            notes = store.list_notes()
            assert len(notes) == 1
            assert notes[0].body == "hello #test"

    def test_pre_existing_note_is_still_searchable(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        db = tmp_path / "test.db"  # type: ignore[union-attr]
        conn = sqlite3.connect(db)
        _build_v1_db(conn)
        conn.close()

        with Store.open(db) as store:
            hits = store.search_notes("hello")
            assert len(hits) == 1

    def test_user_version_is_updated(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        db = tmp_path / "test.db"  # type: ignore[union-attr]
        conn = sqlite3.connect(db)
        _build_v1_db(conn)
        conn.close()

        with Store.open(db) as store:
            (version,) = store._conn.execute(
                "PRAGMA user_version"
            ).fetchone()
            assert version == SCHEMA_VERSION

    def test_reopening_after_migration_is_idempotent(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        db = tmp_path / "test.db"  # type: ignore[union-attr]
        conn = sqlite3.connect(db)
        _build_v1_db(conn)
        conn.close()

        with Store.open(db) as store:
            store.add_note("after migration")
        # Second open should change nothing.
        with Store.open(db) as store:
            notes = store.list_notes()
            assert len(notes) == 2
            (version,) = store._conn.execute(
                "PRAGMA user_version"
            ).fetchone()
            assert version == SCHEMA_VERSION

    def test_future_version_raises_schema_version_error(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        db = tmp_path / "test.db"  # type: ignore[union-attr]
        conn = sqlite3.connect(db)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(_V1_DDL)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 99}")
        conn.commit()
        conn.close()

        with pytest.raises(SchemaVersionError):
            Store.open(db)

    def test_fresh_database_has_all_columns(self) -> None:
        with Store.open(":memory:") as store:
            cols = {
                row[1]
                for row in store._conn.execute(
                    "PRAGMA table_info(notes)"
                ).fetchall()
            }
            assert "page" in cols
            assert "citekey" in cols
            assert "parent_id" in cols
