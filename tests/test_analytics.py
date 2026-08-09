"""Tests for hashline.analytics.

pandas costs about 1.2s to import versus about 40ms for the rest of the app,
so the guard tests below -- which must run before anything else in this file
touches hashline.analytics -- are the whole reason this module is designed
the way it is. See tests/test_ml_search.py for the same pattern applied to
torch/sentence_transformers.
"""

import sys
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest

from hashline import analytics
from hashline.models import BibEntry
from hashline.store import Store


def test_importing_the_cli_does_not_load_pandas() -> None:
    import hashline.cli  # noqa: F401

    assert "pandas" not in sys.modules


def test_importing_the_web_app_does_not_load_pandas() -> None:
    import hashline.web.app  # noqa: F401

    assert "pandas" not in sys.modules


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


class TestNotesFrame:
    def test_empty_store_gives_an_empty_correctly_typed_frame(
        self, store: Store
    ) -> None:
        df = analytics.notes_frame(store)
        assert list(df.columns) == [
            "id",
            "created_at",
            "body",
            "source",
            "page",
            "citekey",
            "parent_id",
        ]
        assert len(df) == 0
        assert df["id"].dtype == "int64"
        assert str(df["created_at"].dtype) == "datetime64[ns, UTC]"
        assert df["parent_id"].dtype == "Int64"
        # groupby/resample must not raise on the empty frame.
        df.groupby("citekey").size()
        df.set_index("created_at").resample("D").size()

    def test_one_row_per_note_with_the_stored_fields(self, store: Store) -> None:
        store.upsert_bib_entries([_smith_bib()])
        note = store.add_note("hello #rust", page="12", citekey="smith2020")
        df = analytics.notes_frame(store)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["id"] == note.id
        assert row["body"] == "hello #rust"
        assert row["page"] == "12"
        assert row["citekey"] == "smith2020"

    def test_created_at_is_tz_aware_utc(self, store: Store) -> None:
        when = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
        store.add_note("one", created_at=when)
        df = analytics.notes_frame(store)
        assert str(df["created_at"].dtype) == "datetime64[ns, UTC]"
        assert df["created_at"].iloc[0].to_pydatetime() == when

    def test_parent_id_is_null_for_root_notes(self, store: Store) -> None:
        import pandas as pd

        root = store.add_note("root")
        store.add_note("reply", parent_id=root.id)
        df = analytics.notes_frame(store).set_index("id")
        assert pd.isna(df.loc[root.id, "parent_id"])

    def test_reply_records_its_parent_id(self, store: Store) -> None:
        root = store.add_note("root")
        reply = store.add_note("reply", parent_id=root.id)
        df = analytics.notes_frame(store).set_index("id")
        assert int(df.loc[reply.id, "parent_id"]) == root.id


class TestTagsFrame:
    def test_empty_store_gives_an_empty_correctly_typed_frame(
        self, store: Store
    ) -> None:
        df = analytics.tags_frame(store)
        assert list(df.columns) == ["note_id", "tag"]
        assert len(df) == 0
        assert df["note_id"].dtype == "int64"

    def test_long_form_one_row_per_note_tag_pair(self, store: Store) -> None:
        note = store.add_note("hello #rust #async")
        df = analytics.tags_frame(store)
        assert len(df) == 2
        assert set(df["tag"]) == {"rust", "async"}
        assert set(df["note_id"]) == {note.id}

    def test_untagged_note_contributes_no_rows(self, store: Store) -> None:
        store.add_note("no tags here")
        df = analytics.tags_frame(store)
        assert len(df) == 0


class TestOverview:
    def test_empty_database(self, store: Store) -> None:
        result = analytics.overview(store)
        assert result == {
            "note_count": 0,
            "tag_count": 0,
            "work_count": 0,
            "first_note_at": None,
            "last_note_at": None,
        }

    def test_counts_and_span(self, store: Store) -> None:
        store.upsert_bib_entries([_smith_bib()])
        first = datetime(2026, 1, 1, tzinfo=UTC)
        last = datetime(2026, 1, 3, tzinfo=UTC)
        store.add_note("one #rust", created_at=first, citekey="smith2020")
        store.add_note("two #async", created_at=last)
        result = analytics.overview(store)
        assert result["note_count"] == 2
        assert result["tag_count"] == 2
        assert result["work_count"] == 1
        assert result["first_note_at"] == first
        assert result["last_note_at"] == last

    def test_first_and_last_are_tz_aware(self, store: Store) -> None:
        store.add_note("one", created_at=datetime(2026, 1, 1, tzinfo=UTC))
        result = analytics.overview(store)
        assert result["first_note_at"] is not None
        assert result["first_note_at"].tzinfo is not None


class TestActivity:
    def test_rejects_an_unknown_freq(self, store: Store) -> None:
        with pytest.raises(ValueError, match="freq"):
            analytics.activity(store, freq="M")

    def test_empty_store_gives_an_empty_correctly_typed_frame(
        self, store: Store
    ) -> None:
        df = analytics.activity(store)
        assert list(df.columns) == ["count"]
        assert len(df) == 0
        assert df["count"].dtype == "int64"
        assert str(df.index.dtype) == "datetime64[ns, UTC]"

    def test_gaps_are_filled_with_zero_not_omitted(self, store: Store) -> None:
        store.add_note("day one", created_at=datetime(2026, 1, 1, tzinfo=UTC))
        store.add_note("day three", created_at=datetime(2026, 1, 3, tzinfo=UTC))
        df = analytics.activity(store, freq="D")
        assert len(df) == 3
        assert list(df["count"]) == [1, 0, 1]

    def test_two_notes_same_period_are_summed(self, store: Store) -> None:
        store.add_note("a", created_at=datetime(2026, 1, 1, 1, tzinfo=UTC))
        store.add_note("b", created_at=datetime(2026, 1, 1, 22, tzinfo=UTC))
        df = analytics.activity(store, freq="D")
        assert len(df) == 1
        assert df["count"].iloc[0] == 2

    def test_weekly_bucketing(self, store: Store) -> None:
        store.add_note("a", created_at=datetime(2026, 1, 1, tzinfo=UTC))
        store.add_note("b", created_at=datetime(2026, 1, 15, tzinfo=UTC))
        df = analytics.activity(store, freq="W")
        assert df["count"].sum() == 2

    def test_month_end_bucketing(self, store: Store) -> None:
        store.add_note("a", created_at=datetime(2026, 1, 5, tzinfo=UTC))
        store.add_note("b", created_at=datetime(2026, 2, 5, tzinfo=UTC))
        df = analytics.activity(store, freq="ME")
        assert len(df) == 2
        assert list(df["count"]) == [1, 1]


class TestTagTrend:
    def test_rejects_an_unknown_freq(self, store: Store) -> None:
        with pytest.raises(ValueError, match="freq"):
            analytics.tag_trend(store, freq="M")

    def test_empty_store_gives_an_empty_frame(self, store: Store) -> None:
        df = analytics.tag_trend(store)
        assert len(df) == 0
        assert len(df.columns) == 0

    def test_wide_form_periods_as_rows_tags_as_columns(self, store: Store) -> None:
        store.add_note("a #rust", created_at=datetime(2026, 1, 1, tzinfo=UTC))
        store.add_note("b #python", created_at=datetime(2026, 1, 1, tzinfo=UTC))
        store.add_note("c #rust", created_at=datetime(2026, 1, 3, tzinfo=UTC))
        df = analytics.tag_trend(store, freq="D")
        assert set(df.columns) == {"rust", "python"}
        assert len(df) == 3  # Jan 1, 2, 3 -- day 2 is a zero-filled gap
        jan1 = df.iloc[0]
        assert jan1["rust"] == 1
        assert jan1["python"] == 1
        jan2 = df.iloc[1]
        assert jan2["rust"] == 0
        assert jan2["python"] == 0
        jan3 = df.iloc[2]
        assert jan3["rust"] == 1
        assert jan3["python"] == 0

    def test_restricted_to_the_top_n_tags_overall(self, store: Store) -> None:
        when = datetime(2026, 1, 1, tzinfo=UTC)
        store.add_note("a #popular", created_at=when)
        store.add_note("b #popular", created_at=when)
        store.add_note("c #rare", created_at=when)
        df = analytics.tag_trend(store, freq="D", top=1)
        assert list(df.columns) == ["popular"]

    def test_cells_are_int64(self, store: Store) -> None:
        store.add_note("a #rust", created_at=datetime(2026, 1, 1, tzinfo=UTC))
        df = analytics.tag_trend(store, freq="D")
        assert df["rust"].dtype == "int64"


def _knuth_bib() -> BibEntry:
    return BibEntry(
        citekey="knuth1997",
        tag="knuth1997",
        entry_type="book",
        title="The Art of Computer Programming",
        raw="@book{knuth1997}",
    )


class TestReadingSummary:
    def test_empty_database(self, store: Store) -> None:
        df = analytics.reading_summary(store)
        assert list(df.columns) == [
            "title",
            "note_count",
            "first_note_at",
            "last_note_at",
            "pages",
        ]
        assert len(df) == 0
        assert df["note_count"].dtype == "int64"
        assert str(df["first_note_at"].dtype) == "datetime64[ns, UTC]"

    def test_one_row_per_citekey_with_notes(self, store: Store) -> None:
        store.upsert_bib_entries([_smith_bib(), _knuth_bib()])
        store.add_note("a", citekey="smith2020")
        store.add_note("b", citekey="smith2020")
        store.add_note("c", citekey="knuth1997")
        df = analytics.reading_summary(store)
        assert set(df.index) == {"smith2020", "knuth1997"}
        assert df.loc["smith2020", "note_count"] == 2
        assert df.loc["knuth1997", "note_count"] == 1

    def test_citekey_with_no_notes_is_absent(self, store: Store) -> None:
        store.upsert_bib_entries([_smith_bib(), _knuth_bib()])
        store.add_note("a", citekey="smith2020")
        df = analytics.reading_summary(store)
        assert "knuth1997" not in df.index

    def test_title_is_joined_in(self, store: Store) -> None:
        store.upsert_bib_entries([_smith_bib()])
        store.add_note("a", citekey="smith2020")
        df = analytics.reading_summary(store)
        assert df.loc["smith2020", "title"] == "A Survey"

    def test_first_and_last_note_timestamps(self, store: Store) -> None:
        store.upsert_bib_entries([_smith_bib()])
        first = datetime(2026, 1, 1, tzinfo=UTC)
        last = datetime(2026, 1, 5, tzinfo=UTC)
        middle = datetime(2026, 1, 3, tzinfo=UTC)
        store.add_note("a", citekey="smith2020", created_at=first)
        store.add_note("b", citekey="smith2020", created_at=middle)
        store.add_note("c", citekey="smith2020", created_at=last)
        df = analytics.reading_summary(store)
        assert df.loc["smith2020", "first_note_at"].to_pydatetime() == first
        assert df.loc["smith2020", "last_note_at"].to_pydatetime() == last

    def test_pages_are_free_form_ordered_and_deduplicated(self, store: Store) -> None:
        store.upsert_bib_entries([_smith_bib()])
        t0 = datetime(2026, 1, 1, tzinfo=UTC)
        store.add_note("a", citekey="smith2020", page="12-15", created_at=t0)
        store.add_note(
            "b", citekey="smith2020", page="xii", created_at=t0.replace(day=2)
        )
        store.add_note(
            "c", citekey="smith2020", page="12-15", created_at=t0.replace(day=3)
        )
        store.add_note(
            "d", citekey="smith2020", page="第3章", created_at=t0.replace(day=4)
        )
        df = analytics.reading_summary(store)
        assert df.loc["smith2020", "pages"] == ["12-15", "xii", "第3章"]

    def test_notes_without_a_page_contribute_nothing_to_pages(
        self, store: Store
    ) -> None:
        store.upsert_bib_entries([_smith_bib()])
        store.add_note("a", citekey="smith2020")
        df = analytics.reading_summary(store)
        assert df.loc["smith2020", "pages"] == []


class TestThreadSummary:
    def test_empty_database(self, store: Store) -> None:
        df = analytics.thread_summary(store)
        assert list(df.columns) == [
            "reply_count",
            "max_depth",
            "first_note_at",
            "last_note_at",
        ]
        assert len(df) == 0
        assert df["reply_count"].dtype == "int64"
        assert str(df["first_note_at"].dtype) == "datetime64[ns, UTC]"

    def test_note_with_no_replies_is_a_thread_of_one(self, store: Store) -> None:
        note = store.add_note("lone note")
        df = analytics.thread_summary(store)
        assert df.loc[note.id, "reply_count"] == 0
        assert df.loc[note.id, "max_depth"] == 0

    def test_three_level_thread(self, store: Store) -> None:
        t0 = datetime(2026, 1, 1, tzinfo=UTC)
        root = store.add_note("root", created_at=t0)
        child = store.add_note(
            "child", created_at=t0.replace(day=2), parent_id=root.id
        )
        store.add_note(
            "grandchild", created_at=t0.replace(day=3), parent_id=child.id
        )
        df = analytics.thread_summary(store)
        assert df.loc[root.id, "reply_count"] == 2
        assert df.loc[root.id, "max_depth"] == 2
        assert df.loc[root.id, "first_note_at"].to_pydatetime() == t0
        assert df.loc[root.id, "last_note_at"].to_pydatetime() == t0.replace(day=3)

    def test_counts_the_whole_subtree_not_just_direct_children(
        self, store: Store
    ) -> None:
        root = store.add_note("root")
        a = store.add_note("child a", parent_id=root.id)
        store.add_note("child b", parent_id=root.id)
        store.add_note("grandchild via a", parent_id=a.id)
        df = analytics.thread_summary(store)
        assert df.loc[root.id, "reply_count"] == 3

    def test_two_roots_two_rows(self, store: Store) -> None:
        first = store.add_note("first root")
        second = store.add_note("second root")
        df = analytics.thread_summary(store)
        assert set(df.index) == {first.id, second.id}
        assert (df["reply_count"] == 0).all()

    def test_reply_is_not_its_own_root_row(self, store: Store) -> None:
        root = store.add_note("root")
        reply = store.add_note("reply", parent_id=root.id)
        df = analytics.thread_summary(store)
        assert reply.id not in df.index
