"""Ranking maths for semantic search.

Pure numpy. Arrays and rank lists in, ranked ids out. This module must never
import torch or sentence-transformers, so its tests run in CI without the
optional ``ml`` extra.
"""

from collections.abc import Iterable, Sequence
from typing import Final

import numpy as np
from numpy.typing import NDArray

#: Rank-fusion damping. 60 is the value from the original reciprocal rank
#: fusion paper and is what most implementations use.
DEFAULT_RRF_K: Final = 60

_EPSILON: Final = 1e-12


def normalize_rows(matrix: NDArray[np.floating]) -> NDArray[np.float32]:
    """Scale each row to unit length. Zero rows are left as zeros."""
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got {values.ndim}-D")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.asarray(values / np.maximum(norms, _EPSILON), dtype=np.float32)


def cosine_similarity(
    query: NDArray[np.floating], matrix: NDArray[np.floating]
) -> NDArray[np.float32]:
    """Cosine similarity between one query vector and every row of ``matrix``.

    Returns one score per row, in row order. A zero vector scores 0 against
    everything rather than producing a NaN.
    """
    vector = np.asarray(query, dtype=np.float32).reshape(-1)
    rows = np.asarray(matrix, dtype=np.float32)
    if rows.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got {rows.ndim}-D")
    if rows.shape[0] == 0:
        return np.empty(0, dtype=np.float32)
    if rows.shape[1] != vector.shape[0]:
        raise ValueError(
            f"dimension mismatch: query is {vector.shape[0]}, matrix is {rows.shape[1]}"
        )
    unit_query = vector / max(float(np.linalg.norm(vector)), _EPSILON)
    return np.asarray(normalize_rows(rows) @ unit_query, dtype=np.float32)


def rank_by_similarity(
    query: NDArray[np.floating],
    ids: Sequence[int],
    matrix: NDArray[np.floating],
    *,
    limit: int | None = None,
    min_score: float | None = None,
) -> list[tuple[int, float]]:
    """Rank ``ids`` by cosine similarity to ``query``, best first.

    ``ids[i]`` must describe row ``i`` of ``matrix``. Ties break on the smaller
    id, so the order is deterministic.
    """
    rows = np.asarray(matrix, dtype=np.float32)
    if rows.ndim == 2 and len(ids) != rows.shape[0]:
        raise ValueError(f"got {len(ids)} ids for {rows.shape[0]} rows")
    scores = cosine_similarity(query, rows)
    ranked = sorted(
        zip(ids, (float(score) for score in scores), strict=True),
        key=lambda pair: (-pair[1], pair[0]),
    )
    if min_score is not None:
        ranked = [pair for pair in ranked if pair[1] >= min_score]
    return ranked if limit is None else ranked[:limit]


def fuse_rankings(
    rankings: Iterable[Sequence[int]],
    *,
    k: int = DEFAULT_RRF_K,
    limit: int | None = None,
) -> list[tuple[int, float]]:
    """Combine several ranked id lists with reciprocal rank fusion.

    This is how keyword and semantic results get merged: each list contributes
    ``1 / (k + rank)`` per id, so the two rankings can be blended without their
    scores ever having to be on the same scale. Ties break on the smaller id.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    totals: dict[int, float] = {}
    for ranking in rankings:
        for position, identifier in enumerate(ranking, start=1):
            totals[identifier] = totals.get(identifier, 0.0) + 1.0 / (k + position)
    fused = sorted(totals.items(), key=lambda pair: (-pair[1], pair[0]))
    return fused if limit is None else fused[:limit]
