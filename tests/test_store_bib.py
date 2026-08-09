"""Tests for bibliography storage in the store."""

from collections.abc import Iterator

import pytest

from hashline.models import BibEntry
from hashline.store import Store


@pytest.fixture
def store() -> Iterator[Store]:
    with Store.open(":memory:") as opened:
        yield opened


def _smith() -> BibEntry:
    return BibEntry(
        citekey="smith2020",
        tag="smith2020",
        entry_type="article",
        title="A Survey of Trigram Indexing",
        author="Smith, John",
        year="2020",
        doi="10.1234/synth.2020.001",
        raw="@article{smith2020, ...}",
    )


def _tanaka() -> BibEntry:
    return BibEntry(
        citekey="tanaka2019",
        tag="tanaka2019",
        entry_type="book",
        title="Introduction to Full-Text Search",
        author="Tanaka, Yuki",
        year="2019",
        raw="@book{tanaka2019, ...}",
    )


class TestUpsertBibEntries:
    def test_inserts_entries(self, store: Store) -> None:
        written, kept = store.upsert_bib_entries([_smith(), _tanaka()])
        assert written == 2
        assert kept == 0

    def test_re_import_updates_in_place(self, store: Store) -> None:
        store.upsert_bib_entries([_smith()])
        updated = BibEntry(
            citekey="smith2020",
            tag="smith2020",
            entry_type="article",
            title="Updated Title",
            author="Smith, John",
            year="2020",
            raw="@article{smith2020, updated}",
        )
        store.upsert_bib_entries([updated])
        entry = store.get_bib_entry("smith2020")
        assert entry is not None
        assert entry.title == "Updated Title"

    def test_empty_input_is_a_no_op(self, store: Store) -> None:
        assert store.upsert_bib_entries([]) == (0, 0)

    def test_replace_keeps_cited_entries(self, store: Store) -> None:
        store.upsert_bib_entries([_smith(), _tanaka()])
        store.add_note("A citation", citekey="smith2020")

        # Now replace with a completely different library (empty for simplicity)
        written, kept = store.upsert_bib_entries([], replace=True)
        
        assert written == 0
        assert kept == 1
        
        # smith2020 should survive because it's cited
        assert store.get_bib_entry("smith2020") is not None
        # tanaka2019 should be gone because it's not cited
        assert store.get_bib_entry("tanaka2019") is None


class TestGetBibEntry:
    def test_returns_stored_entry(self, store: Store) -> None:
        store.upsert_bib_entries([_smith()])
        entry = store.get_bib_entry("smith2020")
        assert entry is not None
        assert entry.title == "A Survey of Trigram Indexing"
        assert entry.author == "Smith, John"
        assert entry.year == "2020"

    def test_unknown_citekey_returns_none(self, store: Store) -> None:
        assert store.get_bib_entry("nonexistent") is None


class TestListBibEntries:
    def test_orders_by_citekey(self, store: Store) -> None:
        store.upsert_bib_entries([_tanaka(), _smith()])
        entries = store.list_bib_entries()
        assert [e.citekey for e in entries] == ["smith2020", "tanaka2019"]

    def test_limit(self, store: Store) -> None:
        store.upsert_bib_entries([_smith(), _tanaka()])
        entries = store.list_bib_entries(limit=1)
        assert len(entries) == 1

    def test_empty_store(self, store: Store) -> None:
        assert store.list_bib_entries() == []
