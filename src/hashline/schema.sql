-- src/hashline/schema.sql
--
-- Everything is IF NOT EXISTS so that init_schema() is idempotent and safe to
-- call on an existing database. The embeddings table is here from the start so
-- that adding semantic search later needs no migration.

CREATE TABLE IF NOT EXISTS notes (
  id         INTEGER PRIMARY KEY,
  body       TEXT NOT NULL,
  created_at TEXT NOT NULL,          -- UTC ISO 8601, microsecond precision
  source     TEXT                    -- import origin; NULL for notes typed by hand
);

CREATE TABLE IF NOT EXISTS tags (
  id   INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE          -- normalized, see tags.normalize_tag
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
  updated_at TEXT    NOT NULL,       -- lets us tell a stale embedding from a fresh one
  PRIMARY KEY (note_id, model)       -- several models can coexist for one note
);

CREATE INDEX IF NOT EXISTS idx_notes_created_at ON notes(created_at DESC, id DESC);
CREATE INDEX IF NOT EXISTS idx_note_tags_tag    ON note_tags(tag_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model);

-- trigram so that Japanese text is searchable; unicode61 cannot segment it.
CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
  body,
  content='notes',
  content_rowid='id',
  tokenize='trigram'
);

-- notes への変更を notes_fts に同期する
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
