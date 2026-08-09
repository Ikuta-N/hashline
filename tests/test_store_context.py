"""Tests for the pinned tag/citekey context."""

from collections.abc import Iterator

import pytest

from hashline.models import BibEntry, Context
from hashline.store import Store


@pytest.fixture
def store() -> Iterator[Store]:
    with Store.open(":memory:") as opened:
        yield opened


def _smith_bib() -> BibEntry:
    return BibEntry(
        citekey="smith2020",
        tag="smith2020",
        entry_type="article",
        title="A Survey",
        raw="@article{smith2020}",
    )


class TestGetContext:
    def test_an_unset_context_reads_back_empty(self, store: Store) -> None:
        context = store.get_context()
        assert context.is_empty
        assert context.tags == ()
        assert context.citekey is None


class TestSetContext:
    def test_round_trips_tags_and_citekey(self, store: Store) -> None:
        store.upsert_bib_entries([_smith_bib()])
        store.set_context(Context(tags=("research", "reading"), citekey="smith2020"))
        got = store.get_context()
        assert got.tags == ("research", "reading")
        assert got.citekey == "smith2020"
        assert not got.is_empty

    def test_normalizes_tags_on_the_way_in(self, store: Store) -> None:
        store.set_context(Context(tags=("#Research",)))
        assert store.get_context().tags == ("research",)

    def test_an_unusable_tag_raises_value_error(self, store: Store) -> None:
        with pytest.raises(ValueError, match="invalid tag name"):
            store.set_context(Context(tags=("two words",)))

    def test_a_second_call_replaces_the_first(self, store: Store) -> None:
        store.set_context(Context(tags=("one",)))
        store.set_context(Context(tags=("two",)))
        assert store.get_context().tags == ("two",)


class TestClearContext:
    def test_empties_a_set_context(self, store: Store) -> None:
        store.set_context(Context(tags=("research",)))
        store.clear_context()
        assert store.get_context().is_empty

    def test_is_a_no_op_when_nothing_is_pinned(self, store: Store) -> None:
        store.clear_context()
        assert store.get_context().is_empty


class TestAddNoteWithContext:
    def test_a_context_tag_lands_on_the_note_without_entering_the_body(
        self, store: Store
    ) -> None:
        store.set_context(Context(tags=("research",)))
        note = store.add_note_with_context("plain body")
        assert note.body == "plain body"
        assert "research" in store.tags_for_note(note.id)

    def test_a_pinned_citekey_adds_that_entrys_tag_automatically(
        self, store: Store
    ) -> None:
        store.upsert_bib_entries([_smith_bib()])
        store.set_context(Context(citekey="smith2020"))
        note = store.add_note_with_context("thinking about this")
        assert "smith2020" in store.tags_for_note(note.id)

    def test_a_pinned_citekey_is_written_to_the_note(self, store: Store) -> None:
        store.upsert_bib_entries([_smith_bib()])
        store.set_context(Context(citekey="smith2020"))
        note = store.add_note_with_context("thinking about this")
        assert note.citekey == "smith2020"

    def test_extra_tags_merge_with_context_tags_without_duplicates(
        self, store: Store
    ) -> None:
        store.set_context(Context(tags=("research",)))
        note = store.add_note_with_context("body", extra_tags=["research", "urgent"])
        assert sorted(store.tags_for_note(note.id)) == ["research", "urgent"]

    def test_page_is_stored_when_given(self, store: Store) -> None:
        store.upsert_bib_entries([_smith_bib()])
        store.set_context(Context(citekey="smith2020"))
        note = store.add_note_with_context("body", page="12-15")
        assert note.page == "12-15"

    def test_with_no_context_pinned_it_behaves_like_a_plain_note(
        self, store: Store
    ) -> None:
        note = store.add_note_with_context("body #own")
        assert note.citekey is None
        assert store.tags_for_note(note.id) == ["own"]

    def test_fails_when_pinned_citekey_is_missing_from_bib(self, store: Store) -> None:
        store.set_context(Context(citekey="smith2020"))
        with pytest.raises(ValueError, match="no longer in the bibliography"):
            store.add_note_with_context("thinking about this")


class TestAddNoteIgnoresContext:
    def test_add_note_never_applies_a_pinned_context(self, store: Store) -> None:
        store.set_context(Context(tags=("research",)))
        note = store.add_note("plain body")
        assert store.tags_for_note(note.id) == []
        assert note.citekey is None
