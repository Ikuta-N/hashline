"""Aggregation over a store, expressed as pandas DataFrames.

Storage and retrieval stay exactly as they are in :mod:`hashline.store` --
SQLite with FTS5, one row per note. This module exists only for the numbers
neither the CLI nor the web layer wants to compute twice: totals, activity
over time, tag trends, reading and thread summaries. It sits where
``hashline.ml.hybrid`` sits -- above the core, below the adapters -- and reads
a :class:`~hashline.store.Store` through its public API only.

``pandas`` costs about 1.2 seconds to import, next to about 40 ms for the rest
of the app, so it must never load on a path someone takes to capture a note.
Every function here imports pandas inside its own body, never at module
level -- copy this shape from ``hashline.ml.embed``, which does the same for
``sentence_transformers``. ``tests/test_analytics.py`` asserts that importing
``hashline.cli`` and ``hashline.web.app`` never pulls pandas into
``sys.modules``; that test is the whole point of this file existing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # For annotations only -- never imported at module load time.
    import pandas as pd

    from hashline.store import Store


def notes_frame(store: Store) -> pd.DataFrame:
    """One row per note, in the shape every other function here builds on.

    Columns: ``id`` (int64), ``created_at`` (tz-aware UTC datetime64),
    ``body``, ``source``, ``page``, ``citekey`` (all object/str-or-None), and
    ``parent_id`` (nullable Int64, since a root note's parent is ``None``).

    An empty store gives an empty frame with these exact dtypes -- not a
    frame with no columns -- so a caller can ``groupby``/``resample`` it
    without special-casing an empty database first.
    """
    import pandas as pd

    notes = store.list_notes(limit=-1)
    return pd.DataFrame(
        {
            "id": pd.Series([n.id for n in notes], dtype="int64"),
            "created_at": pd.Series(
                [n.created_at for n in notes], dtype="datetime64[ns, UTC]"
            ),
            "body": pd.Series([n.body for n in notes], dtype="object"),
            "source": pd.Series([n.source for n in notes], dtype="object"),
            "page": pd.Series([n.page for n in notes], dtype="object"),
            "citekey": pd.Series([n.citekey for n in notes], dtype="object"),
            "parent_id": pd.Series([n.parent_id for n in notes], dtype="Int64"),
        }
    )


def tags_frame(store: Store) -> pd.DataFrame:
    """Every (note, tag) link, long form: one row per pair.

    Columns: ``note_id`` (int64), ``tag`` (object). A note with several tags
    appears once per tag; a note with none does not appear at all. An empty
    store gives an empty frame with these dtypes, not a frame with no
    columns.
    """
    import pandas as pd

    pairs = list(store.iter_note_tags())
    return pd.DataFrame(
        {
            "note_id": pd.Series([note_id for note_id, _ in pairs], dtype="int64"),
            "tag": pd.Series([tag for _, tag in pairs], dtype="object"),
        }
    )


def overview(store: Store) -> dict[str, object]:
    """The totals a no-selector view shows: notes, tags, works, and the span.

    Both the CLI and the web render this, so it is computed once, here,
    rather than twice. Keys: ``note_count``, ``tag_count`` (distinct tags in
    use), ``work_count`` (distinct citekeys that have at least one note),
    ``first_note_at`` and ``last_note_at`` (tz-aware UTC ``datetime``, or
    ``None`` on an empty database).
    """
    notes = notes_frame(store)
    tags = tags_frame(store)

    if notes.empty:
        first_at = last_at = None
    else:
        first_at = notes["created_at"].min().to_pydatetime()
        last_at = notes["created_at"].max().to_pydatetime()

    return {
        "note_count": int(len(notes)),
        "tag_count": int(tags["tag"].nunique()),
        "work_count": int(notes["citekey"].dropna().nunique()),
        "first_note_at": first_at,
        "last_note_at": last_at,
    }
