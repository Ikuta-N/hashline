"""Command line adapter.

A thin shell over the core: this module owns all filesystem I/O and all
formatting, and holds no note logic of its own.
"""

import os
import re
from collections.abc import Iterable, Iterator, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Final, cast

import typer

from hashline.importer import Document, parse_documents
from hashline.models import Note
from hashline.store import Store

#: Suffixes picked up when a directory is imported. An explicitly named file is
#: read whatever it is called.
TEXT_SUFFIXES: Final = frozenset({".md", ".markdown", ".txt"})

_DB_ENV_VAR: Final = "HASHLINE_DB"
_BODY_WIDTH: Final = 90
_WHITESPACE_RE: Final = re.compile(r"\s+")


class Mode(StrEnum):
    """How an imported document is cut into notes."""

    line = "line"
    heading = "heading"


app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Local-first micro-notes: one line, inline #hashtags, fast retrieval.",
)


def default_db_path() -> Path:
    """Where notes live unless ``--db`` or ``$HASHLINE_DB`` says otherwise."""
    override = os.environ.get(_DB_ENV_VAR)
    if override:
        return Path(override)
    data_home = os.environ.get("XDG_DATA_HOME")
    root = Path(data_home) if data_home else Path.home() / ".local" / "share"
    return root / "hashline" / "hashline.db"


@app.callback()
def main(
    ctx: typer.Context,
    db: Annotated[
        Path | None,
        typer.Option("--db", help="Database file. Defaults to $HASHLINE_DB."),
    ] = None,
) -> None:
    ctx.obj = db if db is not None else default_db_path()


def _open(ctx: typer.Context) -> Store:
    return Store.open(cast(Path, ctx.obj))


@app.command()
def add(
    ctx: typer.Context,
    text: Annotated[str, typer.Argument(help="The note body. Inline #tags count.")],
    tag: Annotated[
        list[str] | None,
        typer.Option("--tag", "-t", help="Extra tag; repeatable."),
    ] = None,
) -> None:
    """Store one note."""
    with _open(ctx) as store:
        try:
            note = store.add_note(text, extra_tags=tag or ())
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        typer.echo(_format_note(note, store.tags_for_note(note.id)))


@app.command("list")
def list_(
    ctx: typer.Context,
    tag: Annotated[
        str | None, typer.Option("--tag", "-t", help="Only this tag.")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
) -> None:
    """Show the timeline, newest first."""
    with _open(ctx) as store:
        notes = store.list_notes(tag=tag, limit=limit)
        if not notes:
            typer.echo("no notes yet")
            return
        for note in notes:
            typer.echo(_format_note(note, store.tags_for_note(note.id)))


@app.command()
def search(
    ctx: typer.Context,
    query: Annotated[str, typer.Argument(help="Text to look for.")],
    tag: Annotated[
        str | None, typer.Option("--tag", "-t", help="Only this tag.")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
) -> None:
    """Full-text search, best match first."""
    with _open(ctx) as store:
        hits = store.search_notes(query, tag=tag, limit=limit)
        if not hits:
            typer.echo("no matches")
            return
        for hit in hits:
            note = hit.note
            body = _format_note(note, store.tags_for_note(note.id))
            typer.echo(f"{hit.score:6.2f}  {body}")


@app.command()
def tags(
    ctx: typer.Context,
    limit: Annotated[int | None, typer.Option("--limit", "-n")] = None,
) -> None:
    """List tags in use, most used first."""
    with _open(ctx) as store:
        counts = store.list_tags(limit=limit)
        if not counts:
            typer.echo("no tags yet")
            return
        for entry in counts:
            typer.echo(f"{entry.count:>5}  {entry.name}")


@app.command("import")
def import_(
    ctx: typer.Context,
    paths: Annotated[list[Path], typer.Argument(help="Files or directories.")],
    mode: Annotated[
        Mode, typer.Option("--mode", "-m", help="How to cut each document.")
    ] = Mode.line,
    tag: Annotated[
        list[str] | None,
        typer.Option("--tag", "-t", help="Tag applied to every imported note."),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Report what would be imported.")
    ] = False,
) -> None:
    """Import text and Markdown files as notes."""
    documents, skipped = collect_documents(paths)
    for problem in skipped:
        typer.echo(f"skipped {problem}", err=True)
    try:
        drafts = parse_documents(documents, mode=mode.value, common_tags=tag or ())
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if dry_run:
        typer.echo(f"would import {len(drafts)} notes from {len(documents)} files")
        return
    with _open(ctx) as store:
        stored = store.add_notes(drafts)
    typer.echo(f"imported {len(stored)} notes from {len(documents)} files")


def collect_documents(
    paths: Iterable[Path],
) -> tuple[list[Document], list[str]]:
    """Read every importable file under ``paths``.

    Returns the documents plus a list of human-readable problems for files that
    could not be read. This is the only place the importer path touches disk.
    """
    documents: list[Document] = []
    skipped: list[str] = []
    for path in paths:
        if not path.exists():
            raise typer.BadParameter(f"no such file or directory: {path}")
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


def _format_note(note: Note, tag_names: Sequence[str]) -> str:
    stamp = note.created_at.astimezone().strftime("%Y-%m-%d %H:%M")
    suffix = f"  [{', '.join(tag_names)}]" if tag_names else ""
    return f"{note.id:>5}  {stamp}  {_one_line(note.body)}{suffix}"


def _one_line(body: str) -> str:
    collapsed = _WHITESPACE_RE.sub(" ", body).strip()
    if len(collapsed) <= _BODY_WIDTH:
        return collapsed
    return collapsed[: _BODY_WIDTH - 1] + "…"
