-- src/hashline/schema.sql
CREATE TABLE notes (
  id         INTEGER PRIMARY KEY,
  body       TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE tags (
  id   INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE
);

CREATE TABLE note_tags (
  note_id INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
  tag_id  INTEGER NOT NULL REFERENCES tags(id)  ON DELETE CASCADE,
  PRIMARY KEY (note_id, tag_id)
);

CREATE TABLE embeddings (
  note_id INTEGER PRIMARY KEY REFERENCES notes(id) ON DELETE CASCADE,
  model   TEXT    NOT NULL,
  dim     INTEGER NOT NULL,
  vec     BLOB    NOT NULL
);

CREATE INDEX idx_notes_created_at ON notes(created_at DESC);
CREATE INDEX idx_note_tags_tag    ON note_tags(tag_id);

CREATE VIRTUAL TABLE notes_fts USING fts5(
  body,
  content='notes',
  content_rowid='id',
  tokenize='trigram'
);

-- notes への変更を notes_fts に同期する
CREATE TRIGGER notes_ai AFTER INSERT ON notes BEGIN
  INSERT INTO notes_fts(rowid, body) VALUES (new.id, new.body);
END;

CREATE TRIGGER notes_ad AFTER DELETE ON notes BEGIN
  INSERT INTO notes_fts(notes_fts, rowid, body)
    VALUES ('delete', old.id, old.body);
END;

CREATE TRIGGER notes_au AFTER UPDATE ON notes BEGIN
  INSERT INTO notes_fts(notes_fts, rowid, body)
    VALUES ('delete', old.id, old.body);
  INSERT INTO notes_fts(rowid, body) VALUES (new.id, new.body);
END;
