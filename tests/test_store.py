from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from hashline.models import NoteDraft
from hashline.store import SCHEMA_VERSION, NoteHasReplies, Store, default_db_path


@pytest.fixture
def store() -> Iterator[Store]:
    with Store.open(":memory:") as opened:
        yield opened


class TestOpen:
    def test_creates_file_and_parent_directory(self, tmp_path: Path) -> None:
        db = tmp_path / "nested" / "hashline.db"
        with Store.open(db):
            pass
        assert db.exists()

    def test_reopening_keeps_the_notes(self, tmp_path: Path) -> None:
        db = tmp_path / "hashline.db"
        with Store.open(db) as first:
            first.add_note("persisted #sqlite")
        with Store.open(db) as second:
            assert [note.body for note in second.list_notes()] == ["persisted #sqlite"]

    def test_stamps_the_schema_version(self, store: Store) -> None:
        (version,) = store._conn.execute("PRAGMA user_version").fetchone()
        assert version == SCHEMA_VERSION

    def test_init_schema_is_idempotent(self, store: Store) -> None:
        store.add_note("kept #sqlite")
        store.init_schema()
        assert len(store.list_notes()) == 1

    def test_foreign_keys_are_enforced(self, store: Store) -> None:
        (enabled,) = store._conn.execute("PRAGMA foreign_keys").fetchone()
        assert enabled == 1


class TestDefaultDbPath:
    def test_prefers_the_environment_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HASHLINE_DB", "/tmp/somewhere/hl.db")
        assert default_db_path() == Path("/tmp/somewhere/hl.db")

    def test_falls_back_to_the_data_directory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("HASHLINE_DB", raising=False)
        monkeypatch.setenv("XDG_DATA_HOME", "/tmp/xdg")
        assert default_db_path() == Path("/tmp/xdg/hashline/hashline.db")


class TestAddNote:
    def test_returns_a_note_with_an_id(self, store: Store) -> None:
        note = store.add_note("first thought #sqlite")
        assert note.id > 0
        assert note.body == "first thought #sqlite"
        assert note.source is None

    def test_defaults_created_at_to_now_in_utc(self, store: Store) -> None:
        before = datetime.now(UTC)
        note = store.add_note("now")
        assert before <= note.created_at <= datetime.now(UTC)
        assert note.created_at.tzinfo is not None

    def test_honours_an_explicit_created_at(self, store: Store) -> None:
        when = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
        note = store.add_note("backdated", created_at=when)
        assert note.created_at == when

    def test_converts_created_at_to_utc(self, store: Store) -> None:
        jst = timezone(timedelta(hours=9))
        note = store.add_note("tokyo", created_at=datetime(2026, 1, 2, 12, tzinfo=jst))
        stored = store.get_note(note.id)
        assert stored is not None
        assert stored.created_at == datetime(2026, 1, 2, 3, tzinfo=UTC)

    def test_strips_surrounding_whitespace(self, store: Store) -> None:
        assert store.add_note("  padded  ").body == "padded"

    @pytest.mark.parametrize("body", ["", "   ", "\n\t"])
    def test_rejects_a_blank_body(self, store: Store, body: str) -> None:
        with pytest.raises(ValueError):
            store.add_note(body)

    def test_records_the_source(self, store: Store) -> None:
        note = store.add_note("imported", source="notes/2026-08.md")
        assert note.source == "notes/2026-08.md"

    def test_links_inline_tags(self, store: Store) -> None:
        note = store.add_note("bm25 の話 #sqlite #検索")
        assert _tags_of(store, note.id) == ["sqlite", "検索"]

    def test_links_extra_tags_without_touching_the_body(self, store: Store) -> None:
        note = store.add_note("plain body", extra_tags=["Imported"])
        assert note.body == "plain body"
        assert _tags_of(store, note.id) == ["imported"]

    def test_merges_inline_and_extra_tags(self, store: Store) -> None:
        note = store.add_note("body #sqlite", extra_tags=["sqlite", "imported"])
        assert _tags_of(store, note.id) == ["imported", "sqlite"]

    def test_rejects_an_unusable_extra_tag(self, store: Store) -> None:
        with pytest.raises(ValueError):
            store.add_note("body", extra_tags=["two words"])

    def test_reuses_an_existing_tag_row(self, store: Store) -> None:
        store.add_note("one #sqlite")
        store.add_note("two #SQLite")
        (count,) = store._conn.execute("SELECT count(*) FROM tags").fetchone()
        assert count == 1

    def test_rejects_unknown_parent_id(self, store: Store) -> None:
        with pytest.raises(ValueError):
            store.add_note("child", parent_id=999)


class TestAddNotes:
    def test_stores_every_draft(self, store: Store) -> None:
        notes = store.add_notes(
            [NoteDraft(body="a #x"), NoteDraft(body="b", extra_tags=("y",))]
        )
        assert [note.body for note in notes] == ["a #x", "b"]
        assert _tags_of(store, notes[1].id) == ["y"]

    def test_empty_input_is_a_no_op(self, store: Store) -> None:
        assert store.add_notes([]) == []

    def test_rolls_back_the_whole_batch_on_failure(self, store: Store) -> None:
        with pytest.raises(ValueError):
            store.add_notes([NoteDraft(body="good"), NoteDraft(body="  ")])
        assert store.list_notes() == []

    def test_resolves_parent_index(self, store: Store) -> None:
        notes = store.add_notes(
            [
                NoteDraft(body="parent"),
                NoteDraft(body="child", parent_index=0),
            ]
        )
        assert notes[0].id > 0
        assert notes[1].parent_id == notes[0].id

    def test_rejects_forward_parent_index(self, store: Store) -> None:
        with pytest.raises(ValueError, match="forward parent_index 1"):
            store.add_notes(
                [
                    NoteDraft(body="child", parent_index=1),
                    NoteDraft(body="parent"),
                ]
            )


class TestGetAndDelete:
    def test_get_returns_the_note(self, store: Store) -> None:
        note = store.add_note("findable")
        assert store.get_note(note.id) == note

    def test_get_returns_none_when_missing(self, store: Store) -> None:
        assert store.get_note(404) is None

    def test_delete_reports_whether_it_existed(self, store: Store) -> None:
        note = store.add_note("doomed")
        assert store.delete_note(note.id) == 1
        assert store.delete_note(note.id) == 0

    def test_delete_cascades_to_note_tags(self, store: Store) -> None:
        note = store.add_note("doomed #sqlite")
        store.delete_note(note.id)
        (count,) = store._conn.execute("SELECT count(*) FROM note_tags").fetchone()
        assert count == 0

    def test_delete_keeps_the_fts_index_in_sync(self, store: Store) -> None:
        note = store.add_note("doomed body")
        store.delete_note(note.id)
        rows = store._conn.execute(
            "SELECT rowid FROM notes_fts WHERE notes_fts MATCH '\"doomed\"'"
        ).fetchall()
        assert rows == []

    def test_delete_raises_if_replies_exist_and_not_recursive(
        self, store: Store
    ) -> None:
        parent = store.add_note("parent")
        store.add_note("child", parent_id=parent.id)

        with pytest.raises(NoteHasReplies) as excinfo:
            store.delete_note(parent.id)
        assert excinfo.value.note_id == parent.id
        assert excinfo.value.reply_count == 1

    def test_recursive_delete_removes_thread_and_returns_count(
        self, store: Store
    ) -> None:
        parent = store.add_note("parent")
        child = store.add_note("child", parent_id=parent.id)
        store.add_note("grandchild", parent_id=child.id)

        assert store.delete_note(parent.id, recursive=True) == 3
        assert store.list_notes() == []

    def test_recursive_delete_removes_subtree_note_tags_and_embeddings(
        self, store: Store
    ) -> None:
        parent = store.add_note("parent #sqlite")
        child = store.add_note("child #python", parent_id=parent.id)
        store.upsert_embedding(parent.id, model="test", vector=b"123", dim=1)
        store.upsert_embedding(child.id, model="test", vector=b"456", dim=1)

        store.delete_note(parent.id, recursive=True)

        (tags_count,) = store._conn.execute("SELECT count(*) FROM note_tags").fetchone()
        assert tags_count == 0
        (embed_count,) = store._conn.execute(
            "SELECT count(*) FROM embeddings"
        ).fetchone()
        assert embed_count == 0

    def test_recursive_delete_keeps_fts_integrity(self, store: Store) -> None:
        survivor = store.add_note("survivor node")
        parent = store.add_note("parent thread")
        child = store.add_note("child thread", parent_id=parent.id)
        store.add_note("grandchild thread", parent_id=child.id)

        store.delete_note(parent.id, recursive=True)

        # FTS integrity check
        store._conn.execute(
            "INSERT INTO notes_fts(notes_fts) VALUES('integrity-check')"
        )

        # Surviving note should still be searchable
        hits = store.search_notes("survivor")
        assert len(hits) == 1
        assert hits[0].note.id == survivor.id


class TestListNotes:
    def test_orders_newest_first(self, store: Store) -> None:
        base = datetime(2026, 8, 9, tzinfo=UTC)
        store.add_note("older", created_at=base)
        store.add_note("newer", created_at=base + timedelta(minutes=1))
        assert [note.body for note in store.list_notes()] == ["newer", "older"]

    def test_breaks_ties_by_id_descending(self, store: Store) -> None:
        same = datetime(2026, 8, 9, tzinfo=UTC)
        store.add_note("first", created_at=same)
        store.add_note("second", created_at=same)
        assert [note.body for note in store.list_notes()] == ["second", "first"]

    def test_honours_limit_and_offset(self, store: Store) -> None:
        base = datetime(2026, 8, 9, tzinfo=UTC)
        for index in range(5):
            store.add_note(f"note {index}", created_at=base + timedelta(minutes=index))
        page = store.list_notes(limit=2, offset=1)
        assert [note.body for note in page] == ["note 3", "note 2"]

    def test_empty_store(self, store: Store) -> None:
        assert store.list_notes() == []

    def test_roots_only_excludes_replies(self, store: Store) -> None:
        parent = store.add_note("parent")
        store.add_note("child", parent_id=parent.id)
        roots = store.list_notes(roots_only=True)
        assert [n.body for n in roots] == ["parent"]


class TestReplies:
    def test_replies_to_returns_direct_children(self, store: Store) -> None:
        parent = store.add_note("parent")
        child1 = store.add_note("child1", parent_id=parent.id)
        child2 = store.add_note("child2", parent_id=parent.id)
        # Add a grandchild to make sure it's not included
        store.add_note("grandchild", parent_id=child1.id)

        replies = store.replies_to(parent.id)
        assert [n.id for n in replies] == [child1.id, child2.id]

    def test_thread_orders_depth_first(self, store: Store) -> None:
        parent = store.add_note("parent")
        child1 = store.add_note("child1", parent_id=parent.id)
        child2 = store.add_note("child2", parent_id=parent.id)
        gc1 = store.add_note("gc1", parent_id=child1.id)
        gc2 = store.add_note("gc2", parent_id=child1.id)

        thread = store.thread(parent.id)
        assert [n.id for n in thread] == [
            parent.id,
            child1.id,
            gc1.id,
            gc2.id,
            child2.id,
        ]

    def test_thread_raises_on_unknown_id(self, store: Store) -> None:
        with pytest.raises(ValueError):
            store.thread(999)


def _tags_of(store: Store, note_id: int) -> list[str]:
    rows = store._conn.execute(
        "SELECT t.name FROM tags t "
        "JOIN note_tags nt ON nt.tag_id = t.id WHERE nt.note_id = ? "
        "ORDER BY t.name",
        (note_id,),
    ).fetchall()
    return [row["name"] for row in rows]
