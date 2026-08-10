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


#: The only resample frequencies this module accepts. Pinned deliberately:
#: pandas deprecated "M" in favour of "ME", and letting an arbitrary string
#: through would have pandas raise its own, less clear error three frames
#: down instead of here, at the boundary, before any data is touched.
_ALLOWED_FREQ: frozenset[str] = frozenset({"D", "W", "ME"})


def _check_freq(freq: str) -> None:
    if freq not in _ALLOWED_FREQ:
        raise ValueError(f"freq must be one of {sorted(_ALLOWED_FREQ)}, got {freq!r}")


#: The largest value SQLite will take as an INTEGER. Past it, the driver
#: raises `OverflowError`, which is neither a `ValueError` an adapter catches
#: nor a message anyone can act on.
_MAX_TOP: int = 2**63 - 1


def _check_top(top: int) -> None:
    """Reject a `top` that SQLite cannot use as a LIMIT.

    Checked here, at the boundary, for the same reason `freq` is: both end up
    as a query parameter, and the errors SQLite gives back for them are worse
    than the ones this module can give. Zero and below are not merely useless
    -- `LIMIT -1` means *no limit* in SQLite, so a negative `top` silently
    returns every tag and turns the cap into its opposite.
    """
    if not 1 <= top <= _MAX_TOP:
        raise ValueError(f"top must be between 1 and {_MAX_TOP}, got {top}")


def activity(store: Store, *, freq: str = "D") -> pd.DataFrame:
    """Notes per period, zero-filled where a period has no notes.

    Returns a frame indexed by the period start (tz-aware UTC, named
    ``period``), spanning every ``freq`` bucket from the first note to the
    last, with one ``count`` column (int64). A period with no notes is a row
    of 0, not a missing row -- a gap in a chart built from this frame is a
    real gap, not an artifact of the data being absent. ``freq`` must be one
    of ``"D"``, ``"W"`` or ``"ME"``; anything else raises ``ValueError``.
    """
    _check_freq(freq)
    import pandas as pd

    notes = notes_frame(store)
    if notes.empty:
        empty_index = pd.DatetimeIndex(
            [], dtype="datetime64[ns, UTC]", name="period"
        )
        return pd.DataFrame({"count": pd.Series([], dtype="int64", index=empty_index)})

    counts = (
        notes.set_index("created_at")["id"]
        .sort_index()
        .resample(freq)
        .count()
        .astype("int64")
        .rename("count")
        .rename_axis("period")
    )
    return counts.to_frame()


def tag_trend(store: Store, *, freq: str = "W", top: int = 10) -> pd.DataFrame:
    """Note counts per (period, tag), wide form: periods as rows, tags as columns.

    Rows are exactly the periods :func:`activity` would produce for this
    ``freq`` -- the full range spanned by every note, zero-filled. Columns
    are the ``top`` most-used tags overall (:meth:`Store.list_tags`,
    most-used first), so the table stays readable on a library with hundreds
    of tags. A cell is how many notes in that period carried that tag, 0
    rather than missing where none did. ``freq`` is checked exactly as in
    :func:`activity`, and ``top`` must be at least 1 and small enough for
    SQLite to take as a ``LIMIT``; anything else raises ``ValueError``.
    """
    _check_freq(freq)
    _check_top(top)
    import pandas as pd

    periods = activity(store, freq=freq).index
    top_tags = [tag_count.name for tag_count in store.list_tags(limit=top)]

    if not top_tags:
        return pd.DataFrame(
            index=periods, columns=pd.Index([], name="tag"), dtype="int64"
        )

    notes = notes_frame(store)
    tags = tags_frame(store)
    merged = tags[tags["tag"].isin(top_tags)].merge(
        notes[["id", "created_at"]], left_on="note_id", right_on="id"
    )

    wide = (
        merged.set_index("created_at")
        .groupby("tag")["id"]
        .resample(freq)
        .count()
        .unstack("tag", fill_value=0)
        .reindex(index=periods, columns=top_tags, fill_value=0)
        .astype("int64")
    )
    wide = wide.rename_axis("period")
    wide.columns.name = "tag"
    return wide


def reading_summary(store: Store) -> pd.DataFrame:
    """One row per citekey that has at least one note, indexed by citekey.

    Columns: ``title`` (from :meth:`Store.list_bib_entries`, object, may be
    ``None``), ``note_count`` (int64), ``first_note_at`` and ``last_note_at``
    (tz-aware UTC datetime64), and ``pages`` (object: an ordered,
    de-duplicated ``list[str]`` of every page value recorded against that
    citekey, in the order the notes were written).

    Pages are free-form strings (``"12-15"``, ``"xii"``, ``"第3章"``) -- they
    are never coerced to numbers and no range is computed, only collected.

    A citekey in the bibliography with no notes never appears. An empty
    result (no notes reference a citekey) gives an empty frame with these
    dtypes.
    """
    import pandas as pd

    notes = notes_frame(store)
    titles = {entry.citekey: entry.title for entry in store.list_bib_entries()}

    cited = notes.dropna(subset=["citekey"]).sort_values(["citekey", "created_at"])

    citekeys: list[str] = []
    row_titles: list[str | None] = []
    counts: list[int] = []
    firsts: list[pd.Timestamp] = []
    lasts: list[pd.Timestamp] = []
    pages: list[list[str]] = []

    for citekey, group in cited.groupby("citekey", sort=True):
        seen: list[str] = []
        for page in group["page"]:
            if page is not None and page not in seen:
                seen.append(page)
        citekeys.append(str(citekey))
        row_titles.append(titles.get(str(citekey)))
        counts.append(len(group))
        firsts.append(group["created_at"].min())
        lasts.append(group["created_at"].max())
        pages.append(seen)

    return pd.DataFrame(
        {
            "citekey": pd.Series(citekeys, dtype="object"),
            "title": pd.Series(row_titles, dtype="object"),
            "note_count": pd.Series(counts, dtype="int64"),
            "first_note_at": pd.Series(firsts, dtype="datetime64[ns, UTC]"),
            "last_note_at": pd.Series(lasts, dtype="datetime64[ns, UTC]"),
            "pages": pd.Series(pages, dtype="object"),
        }
    ).set_index("citekey")


def thread_summary(store: Store) -> pd.DataFrame:
    """One row per root note in the whole store, indexed by the root's id.

    Columns: ``reply_count`` (int64, every descendant in the subtree, not
    just direct children), ``max_depth`` (int64, 0 for a root with no
    replies), and ``first_note_at``/``last_note_at`` (tz-aware UTC
    datetime64, spanning the root and every descendant). A note with no
    replies is a thread of one: ``reply_count`` 0, ``max_depth`` 0.

    Built on :func:`hashline.outline.build_tree`, which already forms this
    forest and promotes a note whose parent is missing to a root -- this
    function walks the tree it returns rather than ``notes.parent_id``
    itself.
    """
    from datetime import datetime

    import pandas as pd

    from hashline.outline import OutlineNode, build_tree

    notes = store.list_notes(limit=-1)

    def _subtree(node: OutlineNode) -> list[tuple[datetime, int]]:
        members = [(node.note.created_at, 0)]
        for child in node.children:
            members.extend(
                (created_at, depth + 1) for created_at, depth in _subtree(child)
            )
        return members

    root_ids: list[int] = []
    reply_counts: list[int] = []
    max_depths: list[int] = []
    firsts: list[datetime] = []
    lasts: list[datetime] = []

    for root in build_tree(notes):
        members = _subtree(root)
        timestamps = [created_at for created_at, _ in members]
        root_ids.append(root.note.id)
        reply_counts.append(len(members) - 1)
        max_depths.append(max(depth for _, depth in members))
        firsts.append(min(timestamps))
        lasts.append(max(timestamps))

    return pd.DataFrame(
        {
            "root_id": pd.Series(root_ids, dtype="int64"),
            "reply_count": pd.Series(reply_counts, dtype="int64"),
            "max_depth": pd.Series(max_depths, dtype="int64"),
            "first_note_at": pd.Series(firsts, dtype="datetime64[ns, UTC]"),
            "last_note_at": pd.Series(lasts, dtype="datetime64[ns, UTC]"),
        }
    ).set_index("root_id")
