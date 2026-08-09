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

from hashline.ml.protocols import Embedder

#: Small, genuinely multilingual, 384 dimensions. Overridable per call.
DEFAULT_MODEL: Final = "intfloat/multilingual-e5-small"

#: e5 models are trained to be given a prefix naming the role of the text.
#:
#: Both sides get "query: " here rather than the query/passage pair, because
#: finding notes that mean the same thing as a phrase is a symmetric task --
#: which the e5 authors give as the case for using one prefix throughout. It
#: also means one function embeds both a note and a search for it.
QUERY_PREFIX: Final = "query: "

def embedding_key(model_name: str = DEFAULT_MODEL) -> str:
    """What goes in the ``embeddings.model`` column for ``model_name``.

    Not just the model name: e5 returns different vectors for the same text
    under a different prefix, so the prefix is part of what produced the
    vector. Change either and this key changes, old rows keep their own, and
    the two never mix. A dimension check could not do this job -- e5-small
    and the MiniLM model it replaced are both 384-wide.
    """
    return f"{model_name}+{QUERY_PREFIX.strip().rstrip(':')}"


#: The key for the default model, for callers that do not choose one.
EMBEDDING_KEY: Final = f"{DEFAULT_MODEL}+query"

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


class _SentenceTransformerEmbedder:
    """Adapts a sentence-transformers model to :class:`Embedder`.

    The backend's keyword arguments stop here rather than leaking into the
    protocol, so callers -- and test fakes -- only ever see ``encode(texts)``.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def encode(self, texts: list[str]) -> NDArray[np.floating]:
        encoded = self._model.encode(texts, convert_to_numpy=True)
        return np.asarray(encoded, dtype=_DTYPE)


def load_model(name: str = DEFAULT_MODEL) -> Embedder:
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
    return _SentenceTransformerEmbedder(SentenceTransformer(name))


def embed_texts(
    texts: Sequence[str],
    *,
    model: Embedder | None = None,
    model_name: str = DEFAULT_MODEL,
) -> NDArray[np.float32]:
    """Embed ``texts`` into one row each.

    Each text is prefixed with :data:`QUERY_PREFIX` before it reaches the
    model, so notes and searches are encoded the same way and can be compared
    directly.

    Pass ``model`` to reuse a loaded model across batches. Returns a
    ``(len(texts), dim)`` float32 array; an empty input gives ``(0, 0)``.
    """
    if not texts:
        return np.empty((0, 0), dtype=_DTYPE)
    encoder = model if model is not None else load_model(model_name)
    encoded = encoder.encode([QUERY_PREFIX + text for text in texts])
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
