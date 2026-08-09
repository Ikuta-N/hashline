"""Turn already-read documents into note drafts.

Pure functions only. Nothing here opens a file or walks a directory -- the CLI
does the reading and hands over :class:`Document` values.
"""

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from hashline.models import NoteDraft
from hashline.tags import normalize_tag

#: How a document is cut into notes.
SplitMode = Literal["line", "heading", "outline"]

#: The one interface a split strategy has to satisfy: document text in, one
#: ``(body, depth)`` per note out. ``depth`` is the note's level in the tree the
#: document describes, and is 0 throughout for a mode that has no hierarchy.
#: A new mode is a function plus an entry in ``SPLITTERS``.
Splitter = Callable[[str], list[tuple[str, int]]]

_FENCE_RE: Final = re.compile(r" {0,3}(`{3,}|~{3,})")
_HEADING_RE: Final = re.compile(r" {0,3}#{1,6}(?:\s|$)")


@dataclass(frozen=True, slots=True)
class Document:
    """One document that has already been read into memory."""

    source: str
    text: str


def iter_fenced_lines(text: str) -> Iterable[tuple[str, bool]]:
    """Yield each line and whether it is inside a fenced code block."""
    fence: str | None = None
    for line in text.splitlines():
        marker = _FENCE_RE.match(line)
        if fence is None and marker is not None:
            fence = marker.group(1)[0]
            yield line, True
        elif fence is not None and marker is not None and marker.group(1)[0] == fence:
            fence = None
            yield line, True
        else:
            yield line, fence is not None


def split_lines(text: str) -> list[str]:
    """One note per non-blank line."""
    return [stripped for line in text.splitlines() if (stripped := line.strip())]


def split_heading_sections(text: str) -> list[tuple[str, int]]:
    """One note per Markdown section, with its depth in the heading tree.

    ``## 1.1`` under ``# 1`` is a child of it, and a later ``## 1.2`` is a
    sibling of ``## 1.1`` rather than a child -- so a chapter/section/subsection
    document comes in as the tree it already was on the page.

    Levels are ranked, not counted: a document that jumps from ``#`` to ``###``
    describes two levels, not three. Text before the first heading is a root of
    its own, so nothing is dropped and nothing is adopted by a heading that
    happens to follow it.
    """
    sections: list[list[str]] = [[]]
    #: Heading levels of the current ancestor chain. The preamble holds no
    #: level, which is why it comes out at depth 0 alongside the first heading.
    ancestors: list[int] = []
    depths: list[int] = [0]

    for line, is_fenced in iter_fenced_lines(text):
        heading = None if is_fenced else _HEADING_RE.match(line)
        if heading is not None:
            level = heading.group(0).count("#")
            while ancestors and ancestors[-1] >= level:
                ancestors.pop()
            ancestors.append(level)
            sections.append([])
            depths.append(len(ancestors) - 1)
        sections[-1].append(line)

    return [
        (body, depth)
        for section, depth in zip(sections, depths, strict=True)
        if (body := "\n".join(section).strip())
    ]


def split_headings(text: str) -> list[str]:
    """One note per Markdown section: a heading plus the lines under it.

    Text before the first heading becomes its own note, so nothing is dropped.
    A ``#`` inside a fenced code block is code, not a heading.
    """
    return [body for body, _ in split_heading_sections(text)]


def _line_items(text: str) -> list[tuple[str, int]]:
    """``split_lines`` as a splitter. Line mode describes no hierarchy."""
    return [(body, 0) for body in split_lines(text)]


def _outline_items(text: str) -> list[tuple[str, int]]:
    """``outline.split_outline`` as a splitter.

    Imported here rather than at module scope: ``hashline.outline`` uses
    ``iter_fenced_lines`` from this module, so the two would import each other.
    """
    from hashline.outline import split_outline

    return [(item.body, item.depth) for item in split_outline(text)]


SPLITTERS: Final[Mapping[SplitMode, Splitter]] = {
    "line": _line_items,
    "heading": split_heading_sections,
    "outline": _outline_items,
}


def parse_document(
    doc: Document,
    *,
    mode: SplitMode = "line",
    common_tags: Sequence[str] = (),
) -> list[NoteDraft]:
    """Cut one document into drafts, tagging each with ``common_tags``.

    ``common_tags`` are carried as metadata; the note body is left exactly as it
    was written. Raises ``ValueError`` for an unknown ``mode`` or an unusable tag.
    """
    try:
        splitter = SPLITTERS[mode]
    except KeyError:
        raise ValueError(f"unknown split mode: {mode!r}") from None
    items = splitter(doc.text)

    extra = _normalize_common_tags(common_tags)

    drafts = []
    stack: list[tuple[int, int]] = []  # (depth, index)

    for i, (body, target_depth) in enumerate(items):
        while stack and stack[-1][0] >= target_depth:
            stack.pop()

        parent_depth = stack[-1][0] if stack else -1
        parent_index = stack[-1][1] if stack else None

        actual_depth = min(target_depth, parent_depth + 1)

        draft = NoteDraft(
            body=body,
            source=doc.source,
            extra_tags=extra,
            parent_index=parent_index,
        )
        drafts.append(draft)

        stack.append((actual_depth, i))

    return drafts


def parse_documents(
    docs: Iterable[Document],
    *,
    mode: SplitMode = "line",
    common_tags: Sequence[str] = (),
) -> list[NoteDraft]:
    """Cut many documents into drafts, keeping the order they were given in."""
    from dataclasses import replace

    drafts: list[NoteDraft] = []
    for doc in docs:
        offset = len(drafts)
        doc_drafts = parse_document(doc, mode=mode, common_tags=common_tags)
        if offset > 0:
            doc_drafts = [
                replace(draft, parent_index=draft.parent_index + offset)
                if draft.parent_index is not None
                else draft
                for draft in doc_drafts
            ]
        drafts.extend(doc_drafts)
    return drafts


def _normalize_common_tags(common_tags: Sequence[str]) -> tuple[str, ...]:
    normalized: dict[str, None] = {}
    for raw in common_tags:
        normalized.setdefault(normalize_tag(raw), None)
    return tuple(normalized)
