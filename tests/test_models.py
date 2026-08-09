from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from hashline.models import Note, NoteDraft, SearchHit, Tag, TagCount


def test_note_defaults_source_to_none() -> None:
    note = Note(id=1, body="hello #world", created_at=datetime(2026, 8, 9, tzinfo=UTC))
    assert note.source is None


def test_note_is_immutable() -> None:
    note = Note(id=1, body="hello", created_at=datetime(2026, 8, 9, tzinfo=UTC))
    with pytest.raises(FrozenInstanceError):
        note.body = "changed"  # type: ignore[misc]


def test_note_equality_is_by_value() -> None:
    created = datetime(2026, 8, 9, tzinfo=UTC)
    assert Note(id=1, body="a", created_at=created) == Note(
        id=1, body="a", created_at=created
    )


def test_note_draft_defaults() -> None:
    draft = NoteDraft(body="hello")
    assert draft.created_at is None
    assert draft.source is None
    assert draft.extra_tags == ()


def test_note_draft_carries_extra_tags() -> None:
    draft = NoteDraft(body="hello", source="notes.md", extra_tags=("imported",))
    assert draft.extra_tags == ("imported",)
    assert draft.source == "notes.md"


def test_tag_and_tag_count() -> None:
    assert Tag(id=3, name="sqlite").name == "sqlite"
    assert TagCount(name="sqlite", count=2).count == 2


def test_search_hit_wraps_a_note() -> None:
    note = Note(id=1, body="bm25", created_at=datetime(2026, 8, 9, tzinfo=UTC))
    hit = SearchHit(note=note, score=1.5)
    assert hit.note is note
    assert hit.score == 1.5
