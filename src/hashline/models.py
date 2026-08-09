"""Core data types.

Pure dataclasses only: no sqlite3, no web framework, no CLI framework.
They are the vocabulary shared by the store, the importer and the adapters.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Note:
    """A stored note."""

    id: int
    body: str
    created_at: datetime
    source: str | None = None
    page: str | None = None
    citekey: str | None = None


@dataclass(frozen=True, slots=True)
class Tag:
    """A stored tag. ``name`` is always normalized (see ``tags.normalize_tag``)."""

    id: int
    name: str


@dataclass(frozen=True, slots=True)
class TagCount:
    """A tag together with how many notes carry it."""

    name: str
    count: int


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A note matched by full-text search.

    ``score`` is normalized so that **higher means a better match**, which is the
    opposite of what SQLite's ``bm25()`` returns.
    """

    note: Note
    score: float


@dataclass(frozen=True, slots=True)
class NoteDraft:
    """A note that has not been stored yet.

    This is the boundary type between the importer and the store: the importer
    produces drafts without touching a database, and the store consumes them
    without knowing where they came from.
    """

    body: str
    created_at: datetime | None = None
    source: str | None = None
    extra_tags: tuple[str, ...] = ()
    page: str | None = None
    citekey: str | None = None


@dataclass(frozen=True, slots=True)
class BibEntry:
    """A parsed BibTeX entry.

    ``tag`` is the citekey normalized into a form that ``tags.normalize_tag``
    accepts, so it can be used as an inline ``#tag``.
    """

    citekey: str
    tag: str
    entry_type: str
    title: str | None = None
    author: str | None = None
    year: str | None = None
    doi: str | None = None
    raw: str = ""
