from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest

from hashline.models import TagCount
from hashline.store import Store


@pytest.fixture
def store() -> Iterator[Store]:
    with Store.open(":memory:") as opened:
        yield opened


@pytest.fixture
def populated(store: Store) -> Store:
    base = datetime(2026, 8, 9, tzinfo=UTC)
    store.add_note("bm25 のメモ #sqlite #検索", created_at=base)
    store.add_note("trigram のメモ #sqlite", created_at=base + timedelta(minutes=1))
    store.add_note("no tags here", created_at=base + timedelta(minutes=2))
    store.add_note("from a file", created_at=base, extra_tags=["imported"])
    return store


class TestListNotesByTag:
    def test_returns_only_notes_with_that_tag(self, populated: Store) -> None:
        bodies = [note.body for note in populated.list_notes(tag="sqlite")]
        assert bodies == ["trigram のメモ #sqlite", "bm25 のメモ #sqlite #検索"]

    def test_is_case_insensitive_and_accepts_a_leading_hash(
        self, populated: Store
    ) -> None:
        assert len(populated.list_notes(tag="#SQLite")) == 2

    def test_matches_a_japanese_tag(self, populated: Store) -> None:
        assert [note.body for note in populated.list_notes(tag="検索")] == [
            "bm25 のメモ #sqlite #検索"
        ]

    def test_matches_a_tag_added_out_of_band(self, populated: Store) -> None:
        assert [note.body for note in populated.list_notes(tag="imported")] == [
            "from a file"
        ]

    def test_unknown_tag_returns_nothing(self, populated: Store) -> None:
        assert populated.list_notes(tag="nonexistent") == []

    def test_unusable_tag_returns_nothing_instead_of_raising(
        self, populated: Store
    ) -> None:
        assert populated.list_notes(tag="two words") == []

    def test_honours_limit_and_offset(self, populated: Store) -> None:
        page = populated.list_notes(tag="sqlite", limit=1, offset=1)
        assert [note.body for note in page] == ["bm25 のメモ #sqlite #検索"]


class TestCountNotes:
    def test_counts_everything_by_default(self, populated: Store) -> None:
        assert populated.count_notes() == 4

    def test_counts_one_tag(self, populated: Store) -> None:
        assert populated.count_notes(tag="sqlite") == 2

    def test_unknown_tag_counts_zero(self, populated: Store) -> None:
        assert populated.count_notes(tag="nonexistent") == 0

    def test_unusable_tag_counts_zero(self, populated: Store) -> None:
        assert populated.count_notes(tag="two words") == 0

    def test_empty_store(self, store: Store) -> None:
        assert store.count_notes() == 0


class TestListTags:
    def test_orders_by_use_then_name(self, populated: Store) -> None:
        assert populated.list_tags() == [
            TagCount(name="sqlite", count=2),
            TagCount(name="imported", count=1),
            TagCount(name="検索", count=1),
        ]

    def test_honours_limit(self, populated: Store) -> None:
        assert populated.list_tags(limit=1) == [TagCount(name="sqlite", count=2)]

    def test_omits_tags_left_behind_by_deleted_notes(self, store: Store) -> None:
        note = store.add_note("only user of the tag #lonely")
        store.delete_note(note.id)
        assert store.list_tags() == []

    def test_empty_store(self, store: Store) -> None:
        assert store.list_tags() == []


class TestTagsForNote:
    def test_returns_inline_and_extra_tags_alphabetically(self, store: Store) -> None:
        note = store.add_note("body #sqlite #fts5", extra_tags=["imported"])
        assert store.tags_for_note(note.id) == ["fts5", "imported", "sqlite"]

    def test_untagged_note(self, store: Store) -> None:
        note = store.add_note("nothing to see")
        assert store.tags_for_note(note.id) == []

    def test_missing_note(self, store: Store) -> None:
        assert store.tags_for_note(404) == []
