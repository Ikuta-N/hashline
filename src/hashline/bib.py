"""BibTeX parser for Zotero library exports.

Pure functions over strings. No I/O, no external dependencies. LaTeX escapes
are **not** expanded: ``{\\\"o}`` stays as written. Zotero exports are regular
enough that a simple brace-counting parser handles them.
"""

import re
from typing import Final

from hashline.models import BibEntry
from hashline.tags import normalize_tag

#: Characters that ``normalize_tag`` would reject. Replaced with ``-`` in
#: ``citekey_tag`` so that ``Smith:2020a`` becomes ``smith-2020a``.
_BAD_TAG_CHARS: Final = re.compile(r"[^\w-]+")

#: Matches the opening of a BibTeX entry: @type{citekey,
_ENTRY_START: Final = re.compile(r"@(\w+)\s*\{([^,\s]+)\s*,", re.ASCII)

#: Entry types that carry no bibliographic data.
_SKIP_TYPES: Final = frozenset({"comment", "preamble", "string"})


def citekey_tag(citekey: str) -> str | None:
    """Normalize a citekey into a usable ``#tag`` name, or ``None``.

    Lowercases and replaces runs of characters that ``normalize_tag`` would
    reject with ``-``.  Underscores are kept because they are word characters.

    >>> citekey_tag("Smith:2020a")
    'smith-2020a'
    >>> citekey_tag("smith_title_2020")
    'smith_title_2020'
    """
    candidate = _BAD_TAG_CHARS.sub("-", citekey.lower()).strip("-")
    if not candidate:
        return None
    try:
        return normalize_tag(candidate)
    except ValueError:
        return None


def clean_value(value: str) -> str:
    """Strip the outer braces or quotes and collapse internal whitespace.

    >>> clean_value("{Nested {Title}}")
    'Nested {Title}'
    >>> clean_value('"A quoted value"')
    'A quoted value'
    """
    text = value.strip()
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1]
    elif text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    return " ".join(text.split())


def parse_bibtex(text: str) -> tuple[list[BibEntry], list[str]]:
    """Parse a BibTeX string into entries, tolerating broken ones.

    Returns ``(entries, problems)`` where *problems* are human-readable strings
    describing entries that could not be read.  A malformed entry never aborts
    the whole parse.
    """
    entries: list[BibEntry] = []
    problems: list[str] = []

    for raw_entry, entry_type, citekey, complete in _iter_raw_entries(text):
        if entry_type in _SKIP_TYPES:
            continue
        if not complete:
            problems.append(f"@{entry_type}{{{citekey}}}: unclosed braces")
            continue
        try:
            entry = _parse_one(raw_entry, entry_type, citekey)
        except _ParseError as exc:
            problems.append(str(exc))
            continue
        entries.append(entry)

    return entries, problems


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _ParseError(Exception):
    """A single entry could not be parsed."""


def _iter_raw_entries(
    text: str,
) -> list[tuple[str, str, str, bool]]:
    """Find every ``@type{citekey, ...}`` block in *text*.

    Returns ``(raw_text, entry_type_lower, citekey, complete)`` tuples.
    *complete* is ``False`` when the closing brace was never found, which
    means the entry is structurally broken.  Text between entries is silently
    ignored.
    """
    results: list[tuple[str, str, str, bool]] = []
    for match in _ENTRY_START.finditer(text):
        entry_type = match.group(1).lower()
        citekey = match.group(2)
        start = match.start()
        # Find the matching closing brace by counting.  The opening brace
        # is within the matched text but not at match.end() - 1 (that is the
        # trailing comma).  Search for it between the start and the citekey.
        depth = 0
        brace_pos = text.index("{", start, match.end())
        body_start = brace_pos
        end = len(text)
        complete = False
        for i in range(body_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    complete = True
                    break
        results.append((text[start:end], entry_type, citekey, complete))
    return results


def _parse_fields(body: str) -> dict[str, str]:
    """Extract ``field = value`` pairs from the body text after the citekey comma.

    Values may be ``{...}`` (nested braces counted), ``"..."`` or bare tokens.
    """
    fields: dict[str, str] = {}
    i = 0
    n = len(body)

    while i < n:
        # Skip whitespace and commas
        while i < n and body[i] in " \t\n\r,":
            i += 1
        if i >= n:
            break

        # Read the field name
        name_start = i
        while i < n and body[i] not in " \t\n\r=":
            i += 1
        name = body[name_start:i].strip().lower()
        if not name:
            break

        # Skip to '='
        while i < n and body[i] in " \t\n\r":
            i += 1
        if i >= n or body[i] != "=":
            # Not a field assignment; skip this token
            break
        i += 1  # past '='

        # Skip whitespace
        while i < n and body[i] in " \t\n\r":
            i += 1
        if i >= n:
            break

        # Read the value
        if body[i] == "{":
            # Brace-delimited: count nesting
            depth = 0
            val_start = i
            while i < n:
                if body[i] == "{":
                    depth += 1
                elif body[i] == "}":
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                i += 1
            fields[name] = body[val_start:i]
        elif body[i] == '"':
            # Quote-delimited
            val_start = i
            i += 1
            while i < n and body[i] != '"':
                i += 1
            if i < n:
                i += 1  # past closing quote
            fields[name] = body[val_start:i]
        else:
            # Bare token (number, month abbreviation, etc.)
            val_start = i
            while i < n and body[i] not in ",} \t\n\r":
                i += 1
            fields[name] = body[val_start:i]

    return fields


def _parse_one(raw: str, entry_type: str, citekey: str) -> BibEntry:
    """Parse one entry from its raw text, raising ``_ParseError`` on failure."""
    tag = citekey_tag(citekey)
    if tag is None:
        raise _ParseError(f"citekey {citekey!r} cannot be converted to a usable tag")

    # Find the body: everything after the first comma inside the outer braces.
    brace_pos = raw.index("{")
    comma_pos = raw.index(",", brace_pos)
    # The body ends at the last closing brace.
    body = raw[comma_pos + 1 : raw.rindex("}")]

    try:
        fields = _parse_fields(body)
    except Exception as exc:
        raise _ParseError(
            f"failed to parse fields for @{entry_type}{{{citekey}}}: {exc}"
        ) from exc

    return BibEntry(
        citekey=citekey,
        tag=tag,
        entry_type=entry_type,
        title=clean_value(fields["title"]) if "title" in fields else None,
        author=clean_value(fields["author"]) if "author" in fields else None,
        year=clean_value(fields["year"]) if "year" in fields else None,
        doi=clean_value(fields["doi"]) if "doi" in fields else None,
        raw=raw,
    )
