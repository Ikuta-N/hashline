"""Embedding backend behind the optional ``ml`` extra.

``sentence_transformers`` is imported inside functions, never at module level,
so importing this module costs nothing and the app runs fully without the extra
installed -- only semantic search is unavailable.

The vector codec here is pure numpy and always works, which is what lets the
store round-trip vectors it never has to interpret.
"""

import importlib.util
from collections.abc import Sequence
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

#: Small, multilingual-tolerant and quick to download. Overridable per call.
DEFAULT_MODEL: Final = "sentence-transformers/all-MiniLM-L6-v2"

#: In-memory element type. float32 is what sentence-transformers emits, and
#: 4 bytes per dimension is 1.5 KB for a 384-dim vector. float64 would double
#: that and change no ranking; float16 would halve it and cost three decimal
#: digits that numpy widens back before every dot product anyway.
_DTYPE: Final = np.float32

#: The on-disk element type: the same float32, with its byte order fixed.
#:
#: A SQLite file is portable between machines, so byte order has to be part of
#: the format rather than a property of whoever wrote the row. A natively
#: packed vector written on x86 and read on a big-endian host comes back as
#: plausible garbage -- no exception, just silently wrong rankings. Pinning it
#: costs nothing where the native order already matches.
_STORAGE_DTYPE: Final = np.dtype("<f4")

#: Bytes per dimension. Named because the dim column is checked against it.
_ITEMSIZE: Final = _STORAGE_DTYPE.itemsize


class MlExtraNotInstalled(RuntimeError):
    """Raised when an embedding is requested without the ``ml`` extra."""


def is_available() -> bool:
    """Whether embeddings can be computed in this environment.

    Callers use this to disable semantic search gracefully instead of letting
    an ImportError escape.
    """
    return importlib.util.find_spec("sentence_transformers") is not None


def load_model(name: str = DEFAULT_MODEL) -> Any:
    """Load a sentence-transformers model.

    Downloads it on first use, which is why nothing that calls this may run in
    CI. Raises :class:`MlExtraNotInstalled` when the extra is missing.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise MlExtraNotInstalled(
            "semantic search needs the 'ml' extra: uv sync --extra ml"
        ) from exc
    return SentenceTransformer(name)


def embed_texts(
    texts: Sequence[str],
    *,
    model: Any | None = None,
    model_name: str = DEFAULT_MODEL,
) -> NDArray[np.float32]:
    """Embed ``texts`` into one row each.

    Pass ``model`` to reuse a loaded model across batches. Returns a
    ``(len(texts), dim)`` float32 array; an empty input gives ``(0, 0)``.
    """
    if not texts:
        return np.empty((0, 0), dtype=_DTYPE)
    encoder = model if model is not None else load_model(model_name)
    encoded = encoder.encode(list(texts), convert_to_numpy=True)
    return np.asarray(encoded, dtype=_DTYPE).reshape(len(texts), -1)


def pack_vector(vector: NDArray[np.floating]) -> bytes:
    """Serialize one vector for the ``embeddings.vec`` BLOB column."""
    values = np.asarray(vector, dtype=_STORAGE_DTYPE).reshape(-1)
    return values.tobytes()


def unpack_vector(
    blob: bytes, *, expected_dim: int | None = None
) -> NDArray[np.float32]:
    """Read back a vector written by :func:`pack_vector`.

    Pass ``expected_dim`` -- the row's ``embeddings.dim`` -- to have a
    truncated blob rejected. Without it only the byte count is checked, so a
    blob that lost a few dimensions reads back as a shorter, entirely
    plausible vector and every ranking it takes part in is quietly wrong.
    """
    if len(blob) % _ITEMSIZE:
        raise ValueError(f"blob of {len(blob)} bytes is not a float32 vector")
    if expected_dim is not None and len(blob) != expected_dim * _ITEMSIZE:
        raise ValueError(
            f"blob of {len(blob)} bytes holds {len(blob) // _ITEMSIZE} "
            f"dimensions, but the row records {expected_dim}"
        )
    return np.asarray(np.frombuffer(blob, dtype=_STORAGE_DTYPE), dtype=_DTYPE).copy()


def unpack_matrix(blobs: Sequence[bytes]) -> NDArray[np.float32]:
    """Stack packed vectors into the matrix ``ml.search`` ranks against."""
    if not blobs:
        return np.empty((0, 0), dtype=_DTYPE)
    rows = [unpack_vector(blob) for blob in blobs]
    widths = {row.shape[0] for row in rows}
    if len(widths) > 1:
        raise ValueError(f"vectors have mixed dimensions: {sorted(widths)}")
    return np.vstack(rows)
