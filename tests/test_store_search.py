from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest

from hashline.store import Store


@pytest.fixture
def store() -> Iterator[Store]:
    with Store.open(":memory:") as opened:
        yield opened


class TestSearchNotes:
    def test_finds_a_note_by_substring(self, store: Store) -> None:
        store.add_note("looked into bm25 today")
        store.add_note("unrelated thought")
        hits = store.search_notes("bm25")
        assert [hit.note.body for hit in hits] == ["looked into bm25 today"]

    def test_finds_a_note_by_japanese_text(self, store: Store) -> None:
        store.add_note("全文検索の話 #検索")
        store.add_note("別の話題")
        hits = store.search_notes("全文検索")
        assert [hit.note.body for hit in hits] == ["全文検索の話 #検索"]

    def test_matches_inside_a_word(self, store: Store) -> None:
        store.add_note("tokenizer notes")
        assert len(store.search_notes("kenize")) == 1

    def test_scores_are_positive_and_descending(self, store: Store) -> None:
        store.add_note("bm25")
        store.add_note("a longer note that mentions bm25 once among other words")
        hits = store.search_notes("bm25")
        assert len(hits) == 2
        assert hits[0].score > 0
        assert hits[0].score >= hits[1].score

    def test_ranks_a_focused_note_above_a_rambling_one(self, store: Store) -> None:
        base = datetime(2026, 8, 9, tzinfo=UTC)
        store.add_note("bm25", created_at=base)
        store.add_note(
            "a much longer note that happens to mention bm25 exactly once "
            "while going on at length about entirely unrelated subjects",
            created_at=base + timedelta(minutes=1),
        )
        hits = store.search_notes("bm25")
        assert hits[0].note.body == "bm25"

    def test_no_match_returns_nothing(self, store: Store) -> None:
        store.add_note("something else")
        assert store.search_notes("bm25") == []

    def test_honours_limit(self, store: Store) -> None:
        for index in range(5):
            store.add_note(f"bm25 note {index}")
        assert len(store.search_notes("bm25", limit=2)) == 2

    def test_reflects_a_deleted_note(self, store: Store) -> None:
        note = store.add_note("bm25 note")
        store.delete_note(note.id)
        assert store.search_notes("bm25") == []


class TestSearchQueryHandling:
    @pytest.mark.parametrize("query", ["", "   ", "\n"])
    def test_blank_query_returns_nothing(self, store: Store, query: str) -> None:
        store.add_note("anything")
        assert store.search_notes(query) == []

    @pytest.mark.parametrize(
        "query",
        ['a "quoted" phrase', "NOT AND OR", "trailing*", "-minus", "a:b", "(paren)"],
    )
    def test_fts5_operators_are_treated_as_literal_text(
        self, store: Store, query: str
    ) -> None:
        store.add_note(f"body containing {query} verbatim")
        hits = store.search_notes(query)
        assert len(hits) == 1

    def test_query_is_trimmed(self, store: Store) -> None:
        store.add_note("bm25 note")
        assert len(store.search_notes("  bm25  ")) == 1


class TestShortQueryFallback:
    def test_two_character_query_still_finds_notes(self, store: Store) -> None:
        store.add_note("メモ帳のこと")
        hits = store.search_notes("メモ")
        assert [hit.note.body for hit in hits] == ["メモ帳のこと"]

    def test_fallback_scores_zero(self, store: Store) -> None:
        store.add_note("メモ帳のこと")
        assert store.search_notes("メモ")[0].score == 0.0

    def test_fallback_orders_newest_first(self, store: Store) -> None:
        base = datetime(2026, 8, 9, tzinfo=UTC)
        store.add_note("ab older", created_at=base)
        store.add_note("ab newer", created_at=base + timedelta(minutes=1))
        assert [hit.note.body for hit in store.search_notes("ab")] == [
            "ab newer",
            "ab older",
        ]

    def test_fallback_escapes_like_wildcards(self, store: Store) -> None:
        store.add_note("literal % sign")
        store.add_note("no wildcard here")
        assert [hit.note.body for hit in store.search_notes("%")] == ["literal % sign"]

    def test_fallback_honours_limit(self, store: Store) -> None:
        for index in range(5):
            store.add_note(f"ab note {index}")
        assert len(store.search_notes("ab", limit=2)) == 2


class TestSearchWithTagFilter:
    @pytest.fixture
    def populated(self, store: Store) -> Store:
        store.add_note("bm25 in sqlite #sqlite")
        store.add_note("bm25 in another engine #elsewhere")
        return store

    def test_narrows_to_the_tag(self, populated: Store) -> None:
        hits = populated.search_notes("bm25", tag="sqlite")
        assert [hit.note.body for hit in hits] == ["bm25 in sqlite #sqlite"]

    def test_accepts_a_leading_hash_and_mixed_case(self, populated: Store) -> None:
        assert len(populated.search_notes("bm25", tag="#SQLite")) == 1

    def test_unknown_tag_returns_nothing(self, populated: Store) -> None:
        assert populated.search_notes("bm25", tag="nonexistent") == []

    def test_unusable_tag_returns_nothing(self, populated: Store) -> None:
        assert populated.search_notes("bm25", tag="two words") == []

    def test_applies_to_the_short_query_fallback(self, populated: Store) -> None:
        hits = populated.search_notes("in", tag="sqlite")
        assert [hit.note.body for hit in hits] == ["bm25 in sqlite #sqlite"]
