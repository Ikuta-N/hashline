from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from hashline.models import NoteDraft
from hashline.store import SCHEMA_VERSION, Store


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


class TestGetAndDelete:
    def test_get_returns_the_note(self, store: Store) -> None:
        note = store.add_note("findable")
        assert store.get_note(note.id) == note

    def test_get_returns_none_when_missing(self, store: Store) -> None:
        assert store.get_note(404) is None

    def test_delete_reports_whether_it_existed(self, store: Store) -> None:
        note = store.add_note("doomed")
        assert store.delete_note(note.id) is True
        assert store.delete_note(note.id) is False

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


def _tags_of(store: Store, note_id: int) -> list[str]:
    rows = store._conn.execute(
        "SELECT t.name FROM tags t "
        "JOIN note_tags nt ON nt.tag_id = t.id WHERE nt.note_id = ? "
        "ORDER BY t.name",
        (note_id,),
    ).fetchall()
    return [row["name"] for row in rows]
