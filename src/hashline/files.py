"""Filesystem I/O shared by the adapters.

This module owns file reading for the CLI and the web layer. The core
(models, tags, store, importer, bib, outline) must never import it,
and it must never import an adapter.
"""

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Final

from hashline.importer import Document

#: Suffixes picked up when a directory is imported. An explicitly named file is
#: read whatever it is called.
TEXT_SUFFIXES: Final = frozenset({".md", ".markdown", ".txt"})


def read_documents(paths: Iterable[Path]) -> tuple[list[Document], list[str]]:
    """Read every importable file under ``paths``.

    Returns the documents plus a list of human-readable problems for files that
    could not be read.
    """
    documents: list[Document] = []
    skipped: list[str] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"no such file or directory: {path}")
        for file in _iter_files(path):
            try:
                text = file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                skipped.append(f"{file}: {exc}")
                continue
            documents.append(Document(source=str(file), text=text))
    return documents, skipped


def _iter_files(path: Path) -> Iterator[Path]:
    if path.is_file():
        yield path
        return
    yield from sorted(
        child
        for child in path.rglob("*")
        if child.is_file() and child.suffix.lower() in TEXT_SUFFIXES
    )


def decode_uploads(
    items: Iterable[tuple[str, bytes]],
) -> tuple[list[Document], list[str]]:
    """Decode uploaded files, filtering by suffix.
    
    Returns the decoded documents plus a list of problems for files that
    had the wrong suffix or could not be decoded as UTF-8.
    """
    documents: list[Document] = []
    skipped: list[str] = []
    
    for filename, content in items:
        path = Path(filename)
        if path.suffix.lower() not in TEXT_SUFFIXES:
            skipped.append(f"{filename}: not a text or markdown file")
            continue
            
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            skipped.append(f"{filename}: {exc}")
            continue
            
        documents.append(Document(source=filename, text=text))
        
    return documents, skipped
