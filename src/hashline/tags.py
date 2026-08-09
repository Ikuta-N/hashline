"""Hashtag extraction and normalization.

Pure functions over strings. No I/O, no database.
"""

import re
from typing import Final

_TAG_BODY: Final = r"\w[\w-]*"

#: A ``#tag`` in note text.
#:
#: The ``#`` must not follow a word character, ``/`` or another ``#``, which is
#: what keeps URL fragments (``https://example.com/page#frag``) out. Anything
#: else may precede it -- notably Japanese punctuation, so ``朝のメモ。#日記``
#: tags the note.
#:
#: The character right after ``#`` must be a word character, which keeps Markdown
#: ATX headings (``# Heading``, ``## Heading``) out. ``\w`` is Unicode-aware, so
#: ``#日本語`` is a tag.
TAG_RE: Final = re.compile(rf"(?<![\w/#])#({_TAG_BODY})")

_VALID_TAG_RE: Final = re.compile(rf"\A{_TAG_BODY}\Z")


def normalize_tag(name: str) -> str:
    """Return the canonical storage form of a tag name.

    Accepts both ``"#Rust"`` and ``"Rust"``. Raises ``ValueError`` if what is
    left cannot be a tag, so that a bad ``--tag`` on the command line fails
    loudly instead of creating a tag no note text could ever match.
    """
    candidate = name.strip().lstrip("#").strip("-").casefold()
    if not _VALID_TAG_RE.match(candidate):
        raise ValueError(f"invalid tag name: {name!r}")
    return candidate


def extract_tags(body: str) -> list[str]:
    """Return the normalized tags in ``body``, deduplicated, in order of appearance."""
    found: dict[str, None] = {}
    for match in TAG_RE.finditer(body):
        found.setdefault(normalize_tag(match.group(1)), None)
    return list(found)
