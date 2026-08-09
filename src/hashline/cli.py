"""Command line adapter.

A thin shell over the core: this module owns all filesystem I/O and all
formatting, and holds no note logic of its own.
"""

import re
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Final, cast

import typer

from hashline.bib import parse_bibtex
from hashline.files import read_documents
from hashline.importer import parse_documents
from hashline.models import DEFAULT_READING_TAG, BibEntry, Context, Note
from hashline.outline import build_tree, render_markdown
from hashline.store import NoteHasReplies, Store, default_db_path
from hashline.tags import normalize_tag

_BODY_WIDTH: Final = 90
_WHITESPACE_RE: Final = re.compile(r"\s+")


class Mode(StrEnum):
    """How an imported document is cut into notes."""

    line = "line"
    heading = "heading"
    outline = "outline"


app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Local-first micro-notes: one line, inline #hashtags, fast retrieval.",
)


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
    page: Annotated[
        str | None,
        typer.Option("--page", help="Page reference; requires a pinned citekey."),
    ] = None,
    no_context: Annotated[
        bool,
        typer.Option("--no-context", help="Ignore the pinned context for this note."),
    ] = False,
) -> None:
    """Store one note.

    Routes through ``add_note_with_context`` so a pinned reading context
    (see ``hashline read start``) applies automatically, unless
    ``--no-context`` asks for a plain note instead.
    """
    with _open(ctx) as store:
        try:
            if no_context:
                if page is not None:
                    raise typer.BadParameter(
                        "--page requires a pinned citekey; --no-context has none"
                    )
                note = store.add_note(text, extra_tags=tag or ())
            else:
                if page is not None and store.get_context().citekey is None:
                    raise typer.BadParameter(
                        "--page requires a pinned citekey; see `hashline read start`"
                    )
                note = store.add_note_with_context(
                    text, page=page, extra_tags=tag or ()
                )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        typer.echo(_format_note(note, store.tags_for_note(note.id)))


@app.command("list")
def list_(
    ctx: typer.Context,
    tag: Annotated[
        str | None, typer.Option("--tag", "-t", help="Only this tag.")
    ] = None,
    citekey: Annotated[
        str | None, typer.Option("--citekey", "-c", help="Only this citekey.")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
    roots_only: Annotated[
        bool, typer.Option("--roots-only", help="Hide replies.")
    ] = False,
) -> None:
    """Show the timeline, newest first."""
    with _open(ctx) as store:
        notes = store.list_notes(
            tag=tag, citekey=citekey, limit=limit, roots_only=roots_only
        )
        if not notes:
            typer.echo("no notes yet")
            return
        for note in notes:
            typer.echo(_format_note(note, store.tags_for_note(note.id)))


@app.command()
def reply(
    ctx: typer.Context,
    parent_id: Annotated[int, typer.Argument(help="ID of the note to reply to.")],
    text: Annotated[str, typer.Argument(help="The reply body.")],
    page: Annotated[
        str | None,
        typer.Option("--page", help="Page reference; requires a pinned citekey."),
    ] = None,
) -> None:
    """Reply to an existing note."""
    with _open(ctx) as store:
        try:
            if page is not None and store.get_context().citekey is None:
                raise typer.BadParameter(
                    "--page requires a pinned citekey; see `hashline read start`"
                )
            note = store.add_note_with_context(text, page=page, parent_id=parent_id)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        typer.echo(_format_note(note, store.tags_for_note(note.id)))


@app.command()
def thread(
    ctx: typer.Context,
    note_id: Annotated[int, typer.Argument(help="ID of the thread root.")],
) -> None:
    """Show a thread of notes."""
    with _open(ctx) as store:
        try:
            notes = store.thread(note_id)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

        depths: dict[int, int] = {}
        for note in notes:
            if note.parent_id is None or note.parent_id not in depths:
                depths[note.id] = 0
            else:
                depths[note.id] = depths[note.parent_id] + 1

            indent = "  " * depths[note.id]
            body = _format_note(note, store.tags_for_note(note.id))
            typer.echo(f"{indent}{body}")


@app.command()
def rm(
    ctx: typer.Context,
    note_id: Annotated[int, typer.Argument(help="ID of the note to delete.")],
    recursive: Annotated[
        bool, typer.Option("--recursive", help="Delete this note and all its replies.")
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip confirmation.")
    ] = False,
) -> None:
    """Delete a note."""
    if not yes:
        typer.confirm(f"Are you sure you want to delete note {note_id}?", abort=True)
    with _open(ctx) as store:
        try:
            count = store.delete_note(note_id, recursive=recursive)
        except NoteHasReplies as exc:
            typer.echo(
                f"note {exc.note_id} has {exc.reply_count} replies; "
                "use --recursive to delete the whole thread",
                err=True,
            )
            raise typer.Exit(1) from exc
        if count == 0:
            typer.echo(f"note {note_id} not found", err=True)
            raise typer.Exit(1)

        suffix = "s" if count != 1 else ""
        typer.echo(f"deleted {count} note{suffix}")


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
def export(
    ctx: typer.Context,
    tag: Annotated[
        str | None, typer.Option("--tag", "-t", help="Only this tag.")
    ] = None,
    citekey: Annotated[
        str | None, typer.Option("--citekey", "-c", help="Only this citekey.")
    ] = None,
    root: Annotated[
        int | None, typer.Option("--root", help="Only this thread root.")
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write to this file instead of stdout."),
    ] = None,
) -> None:
    """Export notes as a Markdown outline."""
    if root is not None and (tag is not None or citekey is not None):
        raise typer.BadParameter("--root cannot be combined with --tag or --citekey")

    with _open(ctx) as store:
        if root is not None:
            try:
                notes = store.thread(root)
            except ValueError as exc:
                raise typer.BadParameter(str(exc)) from exc
        else:
            notes = store.list_notes(tag=tag, citekey=citekey, limit=-1)

    roots = build_tree(notes)
    markdown = render_markdown(roots)

    if out is not None:
        out.write_text(markdown, encoding="utf-8")
    else:
        typer.echo(markdown, nl=False)


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
    try:
        documents, skipped = read_documents(paths)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
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


bib_app = typer.Typer(
    no_args_is_help=True, help="Manage a BibTeX library used for citations."
)
app.add_typer(bib_app, name="bib")


@bib_app.command("import")
def bib_import(
    ctx: typer.Context,
    path: Annotated[Path, typer.Argument(help="A .bib file.")],
    replace: Annotated[
        bool,
        typer.Option("--replace", help="Clear the library before importing."),
    ] = False,
) -> None:
    """Import bibliography entries from a BibTeX file."""
    if not path.exists():
        raise typer.BadParameter(f"no such file: {path}")
    # cli.py owns all filesystem I/O; bib.py never opens a file itself.
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise typer.BadParameter(f"could not read {path}: {exc}") from exc
    entries, problems = parse_bibtex(text)
    for problem in problems:
        typer.echo(f"skipped {problem}", err=True)
    with _open(ctx) as store:
        written, kept = store.upsert_bib_entries(entries, replace=replace)
    if kept > 0:
        typer.echo(f"kept {kept} entries still cited by notes", err=True)
    typer.echo(f"imported {written} entries from {path}")
    if problems:
        typer.echo(f"skipped {len(problems)} entries")


@bib_app.command("list")
def bib_list(
    ctx: typer.Context,
    limit: Annotated[int | None, typer.Option("--limit", "-n")] = None,
) -> None:
    """List bibliography entries, ordered by citekey."""
    with _open(ctx) as store:
        entries = store.list_bib_entries(limit=limit)
        if not entries:
            typer.echo("no bibliography entries yet")
            return
        for entry in entries:
            typer.echo(_format_bib_entry(entry))


@bib_app.command("show")
def bib_show(
    ctx: typer.Context,
    citekey: Annotated[str, typer.Argument(help="The entry's citekey.")],
) -> None:
    """Show one bibliography entry in full."""
    with _open(ctx) as store:
        entry = store.get_bib_entry(citekey)
    if entry is None:
        raise typer.BadParameter(f"no bibliography entry for citekey {citekey!r}")
    typer.echo(_format_bib_detail(entry))


@app.command()
def pin(
    ctx: typer.Context,
    tag: Annotated[
        list[str] | None,
        typer.Argument(help="Tags to pin; replaces any previously pinned tags."),
    ] = None,
    show: Annotated[
        bool, typer.Option("--show", help="Show the pinned context.")
    ] = False,
    clear: Annotated[bool, typer.Option("--clear", help="Unpin the context.")] = False,
) -> None:
    """Pin a tag context that add_note_with_context applies to new notes.

    With no arguments and no flags this shows the current context, so the
    bare command is never a silent no-op. --show does exactly the same
    thing; it exists so that intent reads clearly in scripts and history.
    """
    if show:
        if tag or clear:
            raise typer.BadParameter("cannot combine --show with tags or --clear")
        with _open(ctx) as store:
            typer.echo(_format_context(store.get_context()))
        return

    if clear:
        if tag:
            raise typer.BadParameter("cannot combine --clear with tags")
        with _open(ctx) as store:
            store.clear_context()
        typer.echo("context cleared")
        return

    with _open(ctx) as store:
        if tag:
            try:
                normalized = [normalize_tag(name) for name in tag]
            except ValueError as exc:
                raise typer.BadParameter(str(exc)) from exc
            current = store.get_context()
            store.set_context(Context(tags=tuple(normalized), citekey=current.citekey))
        typer.echo(_format_context(store.get_context()))


read_app = typer.Typer(no_args_is_help=True, help="Pin the work you are reading.")
app.add_typer(read_app, name="read")


@read_app.command("start")
def read_start(
    ctx: typer.Context,
    citekey: Annotated[str, typer.Argument(help="A citekey from `hashline bib list`.")],
    tag: Annotated[
        str,
        typer.Option("--tag", "-t", help="Extra tag pinned alongside the citekey."),
    ] = DEFAULT_READING_TAG,
) -> None:
    """Pin CITEKEY as the work being read.

    Notes added afterward through ``add`` (without --no-context) carry both
    the citekey's tag and this one automatically.
    """
    with _open(ctx) as store:
        if store.get_bib_entry(citekey) is None:
            raise typer.BadParameter(
                f"no bibliography entry for citekey {citekey!r}; "
                "import it first with `hashline bib import`"
            )
        try:
            normalized_tag = normalize_tag(tag)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        store.set_context(Context(tags=(normalized_tag,), citekey=citekey))
        typer.echo(_format_read_status(store))


@read_app.command("status")
def read_status(ctx: typer.Context) -> None:
    """Show the pinned work, if any."""
    with _open(ctx) as store:
        typer.echo(_format_read_status(store))


@read_app.command("stop")
def read_stop(ctx: typer.Context) -> None:
    """Unpin the current context."""
    with _open(ctx) as store:
        store.clear_context()
    typer.echo("context cleared")


def _format_read_status(store: Store) -> str:
    context = store.get_context()
    if context.citekey is None:
        return "nothing pinned"
    entry = store.get_bib_entry(context.citekey)
    title = entry.title if entry is not None and entry.title else "(no title)"
    tags = ", ".join(context.tags) if context.tags else "(none)"
    return f"{context.citekey}  {title}\ntags: {tags}"


def _format_bib_entry(entry: BibEntry) -> str:
    title = entry.title or "(no title)"
    year = entry.year or "----"
    return f"{entry.citekey:<24} {year}  {title}"


def _format_bib_detail(entry: BibEntry) -> str:
    lines = [
        f"citekey  {entry.citekey}",
        f"tag      #{entry.tag}",
        f"type     {entry.entry_type}",
        f"title    {entry.title or '-'}",
        f"author   {entry.author or '-'}",
        f"year     {entry.year or '-'}",
        f"doi      {entry.doi or '-'}",
    ]
    return "\n".join(lines)


def _format_context(context: Context) -> str:
    if context.is_empty:
        return "no context pinned"
    tags = ", ".join(context.tags) if context.tags else "(none)"
    citekey = context.citekey or "(none)"
    return f"tags: {tags}  citekey: {citekey}"


def _format_note(note: Note, tag_names: Sequence[str]) -> str:
    stamp = note.created_at.astimezone().strftime("%Y-%m-%d %H:%M")
    reference = _format_reference(note)
    suffix = f"  [{', '.join(tag_names)}]" if tag_names else ""
    return f"{note.id:>5}  {stamp}  {_one_line(note.body)}{reference}{suffix}"


def _format_reference(note: Note) -> str:
    """The citekey and page a note is attached to, or "" when there is none."""
    if note.citekey is None:
        return ""
    page = f" p.{note.page}" if note.page else ""
    return f"  ({note.citekey}{page})"


def _one_line(body: str) -> str:
    collapsed = _WHITESPACE_RE.sub(" ", body).strip()
    if len(collapsed) <= _BODY_WIDTH:
        return collapsed
    return collapsed[: _BODY_WIDTH - 1] + "…"
