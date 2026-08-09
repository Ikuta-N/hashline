"""Tests for page and citekey on notes."""

import sqlite3
from collections.abc import Iterator

import pytest

from hashline.models import BibEntry
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


class TestPageAndCitekey:
    def test_round_trips_page_and_citekey(self, store: Store) -> None:
        store.upsert_bib_entries([_smith_bib()])
        note = store.add_note("body", page="42", citekey="smith2020")
        assert note.page == "42"
        assert note.citekey == "smith2020"
        got = store.get_note(note.id)
        assert got is not None
        assert got.page == "42"
        assert got.citekey == "smith2020"

    def test_list_notes_filters_by_citekey(self, store: Store) -> None:
        store.upsert_bib_entries([_smith_bib()])
        store.add_note("with ref", citekey="smith2020")
        store.add_note("without ref")
        notes = store.list_notes(citekey="smith2020")
        assert len(notes) == 1
        assert notes[0].body == "with ref"

    def test_unknown_citekey_is_rejected_by_foreign_key(self, store: Store) -> None:
        with pytest.raises(sqlite3.IntegrityError):
            store.add_note("body", citekey="nonexistent")

    def test_page_normalisation_whitespace_only_becomes_none(
        self, store: Store
    ) -> None:
        note = store.add_note("body", page="   ")
        assert note.page is None

    def test_page_normalisation_empty_string_becomes_none(self, store: Store) -> None:
        note = store.add_note("body", page="")
        assert note.page is None

    def test_existing_notes_default_to_none(self, store: Store) -> None:
        note = store.add_note("plain body")
        assert note.page is None
        assert note.citekey is None

    def test_free_form_page_values(self, store: Store) -> None:
        """Page is free-form: numbers, ranges, roman, CJK all accepted."""
        store.upsert_bib_entries([_smith_bib()])
        for page in ["42", "12-15", "xii", "第3章"]:
            note = store.add_note(f"note for {page}", page=page, citekey="smith2020")
            assert note.page == page

    def test_search_notes_returns_page_and_citekey(self, store: Store) -> None:
        store.upsert_bib_entries([_smith_bib()])
        store.add_note("searchable body text", page="7", citekey="smith2020")
        hits = store.search_notes("searchable body")
        assert len(hits) == 1
        assert hits[0].note.page == "7"
        assert hits[0].note.citekey == "smith2020"


class TestIterNoteTags:
    def test_empty_database_yields_nothing(self, store: Store) -> None:
        assert list(store.iter_note_tags()) == []

    def test_yields_every_note_tag_pair(self, store: Store) -> None:
        first = store.add_note("one #rust #async")
        second = store.add_note("two #rust")
        assert list(store.iter_note_tags()) == [
            (first.id, "async"),
            (first.id, "rust"),
            (second.id, "rust"),
        ]

    def test_ordered_by_note_id_then_tag(self, store: Store) -> None:
        first = store.add_note("first #zebra #apple")
        second = store.add_note("second #mango")
        pairs = list(store.iter_note_tags())
        assert pairs == [
            (first.id, "apple"),
            (first.id, "zebra"),
            (second.id, "mango"),
        ]

    def test_a_note_without_tags_is_absent(self, store: Store) -> None:
        store.add_note("no tags here")
        assert list(store.iter_note_tags()) == []
