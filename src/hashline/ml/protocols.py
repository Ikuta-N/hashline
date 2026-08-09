"""The shape of an embedding backend, without naming one.

Pure typing: this module imports nothing but numpy, so it costs nothing and
can be read by code that must run without the ``ml`` extra.
"""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Embedder(Protocol):
    """Anything that turns texts into one vector per text.

    Deliberately one method taking a plain list. ``SentenceTransformer.encode``
    has a dozen keyword arguments; keeping them out of the protocol means a
    test fake is three lines rather than a mock of that signature, and it is
    :func:`hashline.ml.embed.load_model` that owns the real backend's options.
    """

    def encode(self, texts: list[str]) -> NDArray[np.floating]:
        """Return a ``(len(texts), dim)`` array, one row per input."""
        ...
