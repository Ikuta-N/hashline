from collections.abc import Iterator
from datetime import UTC, datetime

import pytest

from hashline.store import Store

MODEL = "test-model"
OTHER = "other-model"


@pytest.fixture
def store() -> Iterator[Store]:
    with Store.open(":memory:") as opened:
        yield opened


class TestUpsertEmbedding:
    def test_stores_a_vector(self, store: Store) -> None:
        note = store.add_note("a note")
        store.upsert_embedding(note.id, model=MODEL, vector=b"\x01\x02", dim=2)
        assert list(store.iter_embeddings(MODEL)) == [(note.id, b"\x01\x02")]

    def test_replaces_the_vector_for_the_same_model(self, store: Store) -> None:
        note = store.add_note("a note")
        store.upsert_embedding(note.id, model=MODEL, vector=b"\x01", dim=1)
        store.upsert_embedding(note.id, model=MODEL, vector=b"\x02", dim=1)
        assert list(store.iter_embeddings(MODEL)) == [(note.id, b"\x02")]

    def test_models_coexist_for_one_note(self, store: Store) -> None:
        note = store.add_note("a note")
        store.upsert_embedding(note.id, model=MODEL, vector=b"\x01", dim=1)
        store.upsert_embedding(note.id, model=OTHER, vector=b"\x02", dim=1)
        assert list(store.iter_embeddings(MODEL)) == [(note.id, b"\x01")]
        assert list(store.iter_embeddings(OTHER)) == [(note.id, b"\x02")]

    def test_records_when_it_was_written(self, store: Store) -> None:
        note = store.add_note("a note")
        when = datetime(2026, 1, 2, 3, tzinfo=UTC)
        store.upsert_embedding(
            note.id, model=MODEL, vector=b"\x01", dim=1, updated_at=when
        )
        (stamp,) = store._conn.execute(
            "SELECT updated_at FROM embeddings WHERE note_id = ?", (note.id,)
        ).fetchone()
        assert datetime.fromisoformat(stamp) == when

    def test_defaults_the_timestamp_to_now(self, store: Store) -> None:
        note = store.add_note("a note")
        before = datetime.now(UTC)
        store.upsert_embedding(note.id, model=MODEL, vector=b"\x01", dim=1)
        (stamp,) = store._conn.execute(
            "SELECT updated_at FROM embeddings WHERE note_id = ?", (note.id,)
        ).fetchone()
        assert datetime.fromisoformat(stamp) >= before

    def test_deleting_the_note_drops_the_embedding(self, store: Store) -> None:
        note = store.add_note("a note")
        store.upsert_embedding(note.id, model=MODEL, vector=b"\x01", dim=1)
        store.delete_note(note.id)
        assert list(store.iter_embeddings(MODEL)) == []


class TestNotesWithoutEmbedding:
    def test_lists_every_note_when_nothing_is_embedded(self, store: Store) -> None:
        store.add_note("one")
        store.add_note("two")
        assert len(store.notes_without_embedding(MODEL)) == 2

    def test_skips_notes_this_model_has_seen(self, store: Store) -> None:
        first = store.add_note("one")
        second = store.add_note("two")
        store.upsert_embedding(first.id, model=MODEL, vector=b"\x01", dim=1)
        assert [note.id for note in store.notes_without_embedding(MODEL)] == [second.id]

    def test_another_model_does_not_count(self, store: Store) -> None:
        note = store.add_note("one")
        store.upsert_embedding(note.id, model=OTHER, vector=b"\x01", dim=1)
        assert [n.id for n in store.notes_without_embedding(MODEL)] == [note.id]

    def test_honours_limit(self, store: Store) -> None:
        for index in range(3):
            store.add_note(f"note {index}")
        assert len(store.notes_without_embedding(MODEL, limit=2)) == 2

    def test_nothing_left_to_embed(self, store: Store) -> None:
        note = store.add_note("one")
        store.upsert_embedding(note.id, model=MODEL, vector=b"\x01", dim=1)
        assert store.notes_without_embedding(MODEL) == []


class TestIterEmbeddings:
    def test_orders_by_note_id(self, store: Store) -> None:
        first = store.add_note("one")
        second = store.add_note("two")
        store.upsert_embedding(second.id, model=MODEL, vector=b"\x02", dim=1)
        store.upsert_embedding(first.id, model=MODEL, vector=b"\x01", dim=1)
        assert [note_id for note_id, _ in store.iter_embeddings(MODEL)] == [
            first.id,
            second.id,
        ]

    def test_unknown_model_yields_nothing(self, store: Store) -> None:
        assert list(store.iter_embeddings("never-used")) == []
