"""Semantic retrieval against a store, shared by the CLI and the web.

The ranking maths lives in :mod:`hashline.ml.search` and knows nothing but
arrays; this is the layer that reads vectors out of a store, asks the model for
one more, and hands back note ids. It exists so that both adapters call the
same code -- an adapter that grew its own copy of the fusion would be note
logic living in a UI.

``sentence_transformers`` is imported inside functions here too, so importing
this module costs no more than numpy.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from hashline.store import Store

#: How deep each ranker contributes to the fusion, regardless of the limit
#: asked for. Fusing only the top 20 of each would let a note both rankers
#: place 25th lose to one that a single ranker happened to put 20th.
FUSION_DEPTH: Final = 100

#: Notes per encode() call. Big enough that the model is not called per note,
#: small enough that progress appears on a large library.
EMBED_BATCH: Final = 32

_log: Final = logging.getLogger("hashline.ml")


class NotIndexed(RuntimeError):
    """Raised when a search runs before anything has been embedded."""


@dataclass(frozen=True, slots=True)
class HybridResult:
    """Ranked note ids, plus what the caller has to tell the user about."""

    #: ``(note_id, fused score)``, best first.
    hits: list[tuple[int, float]]
    #: Notes this model has not embedded yet. A search that quietly ignores
    #: half the library is worse than one that says so.
    pending: int
    #: The ``embeddings.model`` key the search ran against.
    key: str


def is_available() -> bool:
    """Whether semantic search can run at all in this environment."""
    from hashline.ml import embed

    return embed.is_available()


def embedding_key(model_name: str | None = None) -> str:
    """The ``embeddings.model`` key for ``model_name``, or for the default."""
    from hashline.ml import embed

    return embed.embedding_key(model_name or embed.DEFAULT_MODEL)


def pending_count(store: Store, *, model_name: str | None = None) -> int:
    """How many notes this model has not embedded yet."""
    return len(store.notes_without_embedding(embedding_key(model_name)))


def hybrid_search(
    store: Store,
    query: str,
    *,
    tag: str | None = None,
    limit: int = 20,
    model_name: str | None = None,
) -> HybridResult:
    """Rank by keyword and by meaning, then fuse the two orders.

    BM25 and cosine similarity are on unrelated scales, so the ranks are
    combined rather than the scores: reciprocal rank fusion needs no
    normalization constant to tune, and adding a third ranker later costs
    nothing.

    Raises :class:`hashline.ml.embed.MlExtraNotInstalled` when the extra is
    missing and :class:`NotIndexed` when nothing has been embedded yet -- two
    different problems with two different fixes, so the caller can say which.
    """
    from hashline.ml import embed
    from hashline.ml.search import fuse_rankings, rank_by_similarity

    name = model_name or embed.DEFAULT_MODEL
    key = embed.embedding_key(name)

    if not embed.is_available():
        # Checked before the index is consulted so the first message names the
        # actual cause. Otherwise an unindexed database says "run hashline
        # index", and only that command reveals the extra is missing.
        raise embed.MlExtraNotInstalled(
            "semantic search needs the 'ml' extra: uv sync --extra ml"
        )

    rows = list(store.iter_embeddings(key))
    if tag is not None:
        allowed = {note.id for note in store.list_notes(tag=tag, limit=-1)}
        rows = [row for row in rows if row[0] in allowed]
    if not rows:
        raise NotIndexed(f"no notes are indexed for {key}")

    query_vector = embed.embed_texts([query], model=embed.load_model(name))[0]

    ids = [note_id for note_id, _ in rows]
    ranked = rank_by_similarity(
        query_vector, ids, embed.unpack_matrix([blob for _, blob in rows])
    )
    keyword_ids = [
        hit.note.id for hit in store.search_notes(query, tag=tag, limit=FUSION_DEPTH)
    ]
    fused = fuse_rankings(
        [keyword_ids, [note_id for note_id, _ in ranked[:FUSION_DEPTH]]],
        limit=limit,
    )
    return HybridResult(
        hits=fused, pending=pending_count(store, model_name=name), key=key
    )


def index_pending(
    store: Store,
    *,
    model_name: str | None = None,
    limit: int | None = None,
    rebuild: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Embed the notes this model has not seen, and return how many.

    Vectors are L2-normalized before they are stored, so a search is one matrix
    product and the ranker's own normalization becomes a no-op.
    """
    from hashline.ml import embed
    from hashline.ml.search import normalize_rows

    name = model_name or embed.DEFAULT_MODEL
    key = embed.embedding_key(name)

    if rebuild:
        pending = store.list_notes(limit=limit if limit is not None else -1)
    else:
        pending = store.notes_without_embedding(key, limit=limit)
    if not pending:
        return 0

    model = embed.load_model(name)
    done = 0
    for start in range(0, len(pending), EMBED_BATCH):
        batch = pending[start : start + EMBED_BATCH]
        vectors = normalize_rows(
            embed.embed_texts([note.body for note in batch], model=model)
        )
        for note, vector in zip(batch, vectors, strict=True):
            store.upsert_embedding(
                note.id,
                model=key,
                vector=embed.pack_vector(vector),
                dim=int(vector.shape[0]),
            )
        done += len(batch)
        if on_progress is not None:
            on_progress(done, len(pending))
    return done
