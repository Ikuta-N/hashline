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

#: The one interface a split strategy has to satisfy: document text in, note
#: bodies out. A new mode is a function plus an entry in ``SPLITTERS``.
Splitter = Callable[[str], list[str]]

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


def split_headings(text: str) -> list[str]:
    """One note per Markdown section: a heading plus the lines under it.

    Text before the first heading becomes its own note, so nothing is dropped.
    A ``#`` inside a fenced code block is code, not a heading.
    """
    sections: list[list[str]] = [[]]
    for line, is_fenced in iter_fenced_lines(text):
        if not is_fenced and _HEADING_RE.match(line):
            sections.append([])
        sections[-1].append(line)
    return [body for section in sections if (body := "\n".join(section).strip())]


SPLITTERS: Final[Mapping[SplitMode, Splitter]] = {
    "line": split_lines,
    "heading": split_headings,
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
    from hashline.outline import OutlineItem, split_outline

    if mode == "outline":
        items = split_outline(doc.text)
    else:
        try:
            splitter = SPLITTERS[mode]
        except KeyError:
            raise ValueError(f"unknown split mode: {mode!r}") from None
        items = [OutlineItem(body=body, depth=0) for body in splitter(doc.text)]

    extra = _normalize_common_tags(common_tags)

    drafts = []
    stack: list[tuple[int, int]] = []  # (depth, index)

    for i, item in enumerate(items):
        target_depth = item.depth

        while stack and stack[-1][0] >= target_depth:
            stack.pop()

        parent_depth = stack[-1][0] if stack else -1
        parent_index = stack[-1][1] if stack else None

        actual_depth = min(target_depth, parent_depth + 1)

        draft = NoteDraft(
            body=item.body,
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
    drafts: list[NoteDraft] = []
    for doc in docs:
        drafts.extend(parse_document(doc, mode=mode, common_tags=common_tags))
    return drafts


def _normalize_common_tags(common_tags: Sequence[str]) -> tuple[str, ...]:
    normalized: dict[str, None] = {}
    for raw in common_tags:
        normalized.setdefault(normalize_tag(raw), None)
    return tuple(normalized)
