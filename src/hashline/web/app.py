
"""FastAPI + HTMX adapter.

Holds no note logic: it resolves a request into a store call and renders the
result. A fresh connection is opened per request, because a sqlite3 connection
must not be shared across the threads FastAPI runs sync handlers on.
"""

import re
from collections.abc import Awaitable, Callable, Iterator, Sequence
from pathlib import Path
from typing import Annotated, Any, Final
from urllib.parse import urlsplit

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from hashline.bib import parse_bibtex
from hashline.files import decode_uploads, read_documents
from hashline.importer import parse_documents
from hashline.models import DEFAULT_READING_TAG, Context, Note
from hashline.outline import OutlineNode, build_tree, render_markdown
from hashline.store import NoteHasReplies, Store, default_db_path
from hashline.tags import normalize_tag

_HERE: Final = Path(__file__).parent

templates = Jinja2Templates(directory=str(_HERE / "templates"))

app = FastAPI(title="hashline")
app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")

_SAFE_METHODS: Final = {"GET", "HEAD", "OPTIONS"}


@app.middleware("http")
async def reject_cross_origin_writes(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Reject a state-changing request whose Origin is not this host.

    A cross-origin form POST needs no preflight, so any page left open in a
    browser could otherwise delete notes or replace the bibliography here.
    Browsers attach Origin to every non-GET form post and fetch, same-origin
    ones included, so it is the comparison below -- not the presence of the
    header -- that lets this app's own UI through. A request carrying no
    Origin at all is not from a browser (curl, httpx, the test client) and
    is let through; checking every state-changing route here, in one place,
    beats a check copied into each handler.

    Only the host is compared. Behind anything that terminates TLS -- a
    reverse proxy, a tunnel -- the browser sends ``https`` while this app
    still sees ``http`` on the ASGI scope, and comparing the scheme would
    403 every write and take the whole UI down. The host is the part that
    matters anyway: the browser, not the attacker's page, decides which
    host it connects to.
    """
    origin = request.headers.get("origin")
    if request.method not in _SAFE_METHODS and origin is not None:
        if urlsplit(origin).netloc != request.url.netloc:
            return Response(
                status_code=403, content="cross-origin request rejected"
            )
    return await call_next(request)


def get_store() -> Iterator[Store]:
    """Open a store for the lifetime of one request."""
    with Store.open(default_db_path()) as store:
        yield store


StoreDep = Annotated[Store, Depends(get_store)]


@app.get("/", response_class=HTMLResponse)
def index(
    request: Request,
    store: StoreDep,
    q: str = "",
    tag: str = "",
    citekey: str = "",
    roots_only: bool = False,
    limit: int = 50,
) -> HTMLResponse:
    # tags_for_note needs the db, do it before rendering
    notes = _timeline(
        store, q=q, tag=tag, citekey=citekey, roots_only=roots_only, limit=limit
    )

    context_data = {
        "current_page": "notes",
        "notes": notes,
        "q": q,
        "tag": tag,
        "citekey": citekey,
        "roots_only": roots_only,
        "limit": limit,
        "total": store.count_notes(),
    }

    context_data.update(_context_data(store))

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=context_data,
    )


@app.get("/notes", response_class=HTMLResponse)
def notes_fragment(
    request: Request,
    store: StoreDep,
    q: str = "",
    tag: str = "",
    citekey: str = "",
    roots_only: bool = False,
    limit: int = 50,
) -> HTMLResponse:
    notes = _timeline(
        store, q=q, tag=tag, citekey=citekey, roots_only=roots_only, limit=limit
    )

    return templates.TemplateResponse(
        request=request,
        name="_timeline.html",
        context={
            "notes": notes,
            "q": q,
            "tag": tag,
            "citekey": citekey,
            "roots_only": roots_only,
            "limit": limit,
        },
    )


@app.post("/notes", response_class=HTMLResponse)
def create_note(
    request: Request,
    store: StoreDep,
    body: Annotated[str, Form()],
    q: Annotated[str, Form()] = "",
    tag: Annotated[str, Form()] = "",
    tags: Annotated[str, Form()] = "",
    page: Annotated[str, Form()] = "",
    citekey: Annotated[str, Form()] = "",
    roots_only: Annotated[bool, Form()] = False,
    limit: Annotated[int, Form()] = 50,
    no_context: Annotated[bool, Form()] = False,
    parent_id: Annotated[int | None, Form()] = None,
) -> HTMLResponse:
    """Capture a note, then hand back the refreshed timeline.

    Always answers 200 with the timeline. A 4xx would leave HTMX with nothing to
    swap, so the user would click add and see the page sit there; a rejected
    note has to come back as something they can read.
    """
    error: str | None = None
    if body.strip():
        # A blank submission is a no-op, not a mistake worth reporting. Anything
        # else the store refuses -- a pinned work that has left the library, say
        # -- is the user's to see.
        try:
            extra_tags = tuple(tags.split()) if tags.strip() else ()
            # These messages are read in a browser, so they name what is on the
            # screen -- a page field, a context strip -- and not CLI flags.
            if no_context:
                if page:
                    raise ValueError(
                        "a page needs a pinned work, and this note is being "
                        "captured without the context"
                    )
                store.add_note(body, extra_tags=extra_tags, parent_id=parent_id)
            else:
                if page and store.get_context().citekey is None:
                    raise ValueError(
                        "a page needs a pinned work; start reading one from the "
                        "context strip above"
                    )
                store.add_note_with_context(
                    body, page=page or None, extra_tags=extra_tags, parent_id=parent_id
                )
        except ValueError as exc:
            error = str(exc)
    return templates.TemplateResponse(
        request=request,
        name="_timeline.html",
        context={
            "notes": _timeline(
                store, q=q, tag=tag, citekey=citekey, roots_only=roots_only, limit=limit
            ),
            "q": q,
            "tag": tag,
            # _timeline.html reads these too. Only the delete route sets
            # delete_retry_id today, so leaving them out here renders the
            # same as passing them -- until the next thing added to that
            # error block silently comes back with blank filters.
            "citekey": citekey,
            "roots_only": roots_only,
            "limit": limit,
            "error": error,
        },
    )


@app.get("/notes/{note_id}/reply", response_class=HTMLResponse)
def reply_fragment(
    request: Request,
    store: StoreDep,
    note_id: int,
    tag: str = "",
    q: str = "",
    citekey: str = "",
    roots_only: bool = False,
    limit: int = 50,
) -> HTMLResponse:
    """The reply form fragment."""
    return templates.TemplateResponse(
        request=request,
        name="_reply.html",
        context={
            "note_id": note_id,
            "pinned_citekey": store.get_context().citekey,
            "tag": tag,
            "q": q,
            "citekey": citekey,
            "roots_only": roots_only,
            "limit": limit,
        },
    )


@app.get("/notes/{note_id}/thread", response_class=HTMLResponse)
def thread(
    request: Request,
    store: StoreDep,
    note_id: int,
) -> HTMLResponse:
    try:
        found = store.thread(note_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Note not found") from exc

    def flatten(
        roots: Sequence[OutlineNode], depth: int = 0
    ) -> Iterator[tuple[Note, list[str], int]]:
        for root in roots:
            yield root.note, store.tags_for_note(root.note.id), depth
            yield from flatten(root.children, depth + 1)

    items = list(flatten(build_tree(found)))

    return templates.TemplateResponse(
        request=request,
        name="_timeline.html",
        context={
            "notes": items,
            "q": "",
            "tag": "",
        },
    )


@app.post("/notes/{note_id}/delete", response_class=HTMLResponse)
def delete_note(
    request: Request,
    store: StoreDep,
    note_id: int,
    recursive: Annotated[bool, Form()] = False,
    tag: Annotated[str, Form()] = "",
    q: Annotated[str, Form()] = "",
    citekey: Annotated[str, Form()] = "",
    roots_only: Annotated[bool, Form()] = False,
    limit: Annotated[int, Form()] = 50,
) -> HTMLResponse:
    error: str | None = None
    notice: str | None = None
    delete_retry_id: int | None = None
    try:
        count = store.delete_note(note_id, recursive=recursive)
        if count == 0:
            error = f"note {note_id} not found"
        else:
            suffix = "s" if count != 1 else ""
            notice = f"deleted {count} note{suffix}"
    except NoteHasReplies as exc:
        # No CLI flag names here: the reader of this message has a button, not
        # a --recursive switch.
        error = (
            f"note {exc.note_id} has {exc.reply_count} "
            f"{'reply' if exc.reply_count == 1 else 'replies'}"
        )
        delete_retry_id = note_id

    return templates.TemplateResponse(
        request=request,
        name="_timeline.html",
        context={
            "notes": _timeline(
                store, q=q, tag=tag, citekey=citekey, roots_only=roots_only, limit=limit
            ),
            "q": q,
            "tag": tag,
            # The retry form re-submits these, so they have to survive the
            # round trip -- otherwise "delete the whole thread" answers with
            # an unfiltered timeline.
            "citekey": citekey,
            "roots_only": roots_only,
            "limit": limit,
            "error": error,
            "notice": notice,
            "delete_retry_id": delete_retry_id,
        },
    )


def _timeline(
    store: Store,
    *,
    q: str,
    tag: str,
    citekey: str = "",
    roots_only: bool = False,
    limit: int = 50,
) -> list[tuple[Note, list[str], int]]:
    """Notes plus their tags, either searched (flat) or listed (nested)."""
    filter_tag = tag or None
    filter_citekey = citekey or None
    if q.strip():
        # Ranked list and tree are different things -- render search results FLAT.
        found = [hit.note for hit in store.search_notes(q, tag=filter_tag, limit=limit)]
        return [(note, store.tags_for_note(note.id), 0) for note in found]

    found = store.list_notes(
        tag=filter_tag, citekey=filter_citekey, roots_only=roots_only, limit=limit
    )

    def flatten(
        roots: Sequence[OutlineNode], depth: int = 0
    ) -> Iterator[tuple[Note, list[str], int]]:
        for root in roots:
            yield root.note, store.tags_for_note(root.note.id), depth
            yield from flatten(root.children, depth + 1)

    return list(flatten(list(reversed(build_tree(found)))))


def _context_data(store: Store, error: str | None = None) -> dict[str, Any]:
    context = store.get_context()
    data = {
        "pinned_citekey": context.citekey,
        "pinned_tags": context.tags,
        "pinned_title": None,
        "error": error,
    }
    if context.citekey:
        entry = store.get_bib_entry(context.citekey)
        data["pinned_title"] = entry.title if entry and entry.title else "(no title)"
    return data


@app.get("/context", response_class=HTMLResponse)
def get_context(request: Request, store: StoreDep) -> HTMLResponse:
    """The context strip fragment."""
    return templates.TemplateResponse(
        request=request, name="_context.html", context=_context_data(store)
    )


@app.post("/context/pin", response_class=HTMLResponse)
def pin_context(
    request: Request,
    store: StoreDep,
    context_tag: Annotated[str, Form()] = "",
) -> HTMLResponse:
    """Set pinned tags and preserve the citekey."""
    error = None
    try:
        current = store.get_context()
        store.set_context(
            Context(tags=tuple(context_tag.split()), citekey=current.citekey)
        )
    except ValueError as exc:
        error = str(exc)
    return templates.TemplateResponse(
        request=request, name="_context.html", context=_context_data(store, error=error)
    )


@app.post("/context/read", response_class=HTMLResponse)
def read_context(
    request: Request,
    store: StoreDep,
    context_citekey: Annotated[str, Form()],
    context_tag: Annotated[str, Form()] = "",
) -> HTMLResponse:
    """Pin a citekey, optionally with a tag."""
    citekey, tag = context_citekey, context_tag
    error = None
    if store.get_bib_entry(citekey) is None:
        error = (
            f"no bibliography entry for citekey {citekey!r}; "
            "import one on the Import page first"
        )
    else:
        try:
            # Normalize before comparing: the store keeps tags in canonical
            # form, so a raw "#reading" typed into a strip that is all about
            # #tags would miss the check below and be pinned a second time.
            tag_name = normalize_tag(tag) if tag.strip() else DEFAULT_READING_TAG
            # The context strip shows pinned tags and the pinned work as two
            # independent things with separate clear buttons, so starting a
            # read must add the reading tag, not replace what was pinned.
            pinned_tags = store.get_context().tags
            merged_tags = (
                pinned_tags
                if tag_name in pinned_tags
                else (*pinned_tags, tag_name)
            )
            store.set_context(Context(tags=merged_tags, citekey=citekey))
        except ValueError as exc:
            error = str(exc)
    return templates.TemplateResponse(
        request=request, name="_context.html", context=_context_data(store, error=error)
    )


@app.post("/context/clear", response_class=HTMLResponse)
def clear_context(request: Request, store: StoreDep) -> HTMLResponse:
    """Unpin everything."""
    store.clear_context()
    return templates.TemplateResponse(
        request=request, name="_context.html", context=_context_data(store)
    )


@app.post("/context/clear_tags", response_class=HTMLResponse)
def clear_tags(request: Request, store: StoreDep) -> HTMLResponse:
    """Clear pinned tags and preserve the citekey."""
    current = store.get_context()
    store.set_context(Context(tags=(), citekey=current.citekey))
    return templates.TemplateResponse(
        request=request, name="_context.html", context=_context_data(store)
    )


@app.post("/context/clear_read", response_class=HTMLResponse)
def clear_read(request: Request, store: StoreDep) -> HTMLResponse:
    """Clear pinned citekey and preserve the tags."""
    current = store.get_context()
    store.set_context(Context(tags=current.tags, citekey=None))
    return templates.TemplateResponse(
        request=request, name="_context.html", context=_context_data(store)
    )


@app.get("/bib", response_class=HTMLResponse)
def bib(request: Request, store: StoreDep) -> HTMLResponse:
    """Bibliography management."""
    entries = store.list_bib_entries()
    context_data = {
        "current_page": "bib",
        "total": store.count_notes(),
        "entries": entries,
    }
    context_data.update(_context_data(store))
    return templates.TemplateResponse(
        request=request,
        name="bib.html",
        context=context_data,
    )


@app.get("/bib/{citekey}", response_class=HTMLResponse)
def bib_detail(request: Request, citekey: str, store: StoreDep) -> HTMLResponse:
    """Show one bibliography entry in full."""
    entry = store.get_bib_entry(citekey)
    if entry is None:
        raise HTTPException(status_code=404, detail="Citekey not found")

    context_data = {
        "current_page": "bib",
        "total": store.count_notes(),
        "entry": entry,
    }
    context_data.update(_context_data(store))
    return templates.TemplateResponse(
        request=request,
        name="bib_detail.html",
        context=context_data,
    )


@app.get("/import", response_class=HTMLResponse)
def import_(request: Request, store: StoreDep) -> HTMLResponse:
    """Import notes from files."""
    return templates.TemplateResponse(
        request=request,
        name="import.html",
        context={"current_page": "import", "total": store.count_notes()},
    )


@app.post("/import", response_class=HTMLResponse)
def import_notes(
    request: Request,
    store: StoreDep,
    path: str = Form(""),
    files: list[UploadFile] = File([]),  # noqa: B008
    mode: str = Form("line"),
    tags: str = Form(""),
    dry_run: bool = Form(False),
) -> HTMLResponse:
    documents = []
    skipped: list[str] = []

    def import_page(**extra: Any) -> HTMLResponse:
        """The import page, always carrying what has been skipped so far.

        Refusing the submission is no reason to swallow the per-file
        report: a user told only "unknown import mode" would never learn
        that one of the files they attached could not be read either.
        """
        return templates.TemplateResponse(
            request=request,
            name="import.html",
            context={
                "current_page": "import",
                "total": store.count_notes(),
                "skipped": skipped,
                **extra,
            },
        )

    if path:
        p = Path(path)
        if not p.exists():
            return import_page(error=f"no such file or directory: {path}")
        try:
            docs, skps = read_documents([p])
            documents.extend(docs)
            skipped.extend(skps)
        except FileNotFoundError as exc:
            return import_page(error=str(exc))

    upload_items = [(f.filename, f.file.read()) for f in files if f.filename]
    if upload_items:
        docs, skps = decode_uploads(upload_items)
        documents.extend(docs)
        skipped.extend(skps)

    if not documents and not path and not upload_items:
        return import_page(error="Please provide a path or upload files.")

    tag_list = tags.split() if tags else []
    try:
        if mode not in {"line", "heading", "outline"}:
            raise ValueError(f"unknown import mode: {mode}")
        drafts = parse_documents(
            documents,
            mode=mode,  # type: ignore[arg-type]
            common_tags=tag_list,
        )
    except ValueError as exc:
        return import_page(error=str(exc))

    if dry_run:
        notice = f"would import {len(drafts)} notes from {len(documents)} files"
    else:
        stored = store.add_notes(drafts)
        notice = f"imported {len(stored)} notes from {len(documents)} files"

    return import_page(notice=notice)


@app.post("/bib/import", response_class=HTMLResponse)
def bib_import(
    request: Request,
    store: StoreDep,
    path: str = Form(""),
    file: UploadFile | None = File(None),  # noqa: B008
    replace: bool = Form(False),
) -> HTMLResponse:
    text = ""
    source_names = []

    if path:
        p = Path(path)
        if not p.exists():
            return templates.TemplateResponse(
                request=request,
                name="import.html",
                context={
                    "current_page": "import",
                    "total": store.count_notes(),
                    "error": f"no such file: {path}",
                },
            )
        try:
            text += p.read_text(encoding="utf-8") + "\n"
            source_names.append(str(p))
        except (OSError, UnicodeDecodeError) as exc:
            # UnicodeDecodeError subclasses ValueError, not OSError, so a
            # latin-1 .bib file -- ordinary for BibTeX -- needs its own
            # branch of this except clause to avoid a 500.
            return templates.TemplateResponse(
                request=request,
                name="import.html",
                context={
                    "current_page": "import",
                    "total": store.count_notes(),
                    "error": f"could not read {path}: {exc}",
                },
            )
    if file and file.filename:
        try:
            content = file.file.read()
            text += content.decode("utf-8") + "\n"
            source_names.append(file.filename)
        except UnicodeDecodeError as exc:
            return templates.TemplateResponse(
                request=request,
                name="import.html",
                context={
                    "current_page": "import",
                    "total": store.count_notes(),
                    "error": f"could not decode {file.filename}: {exc}",
                },
            )
    
    if not source_names:
        return templates.TemplateResponse(
            request=request,
            name="import.html",
            context={
                "current_page": "import",
                "total": store.count_notes(),
                "error": "Please provide a path or upload a .bib file.",
            },
        )

    source_name = " and ".join(source_names)

    entries, problems = parse_bibtex(text)

    if not entries:
        return templates.TemplateResponse(
            request=request,
            name="import.html",
            context={
                "current_page": "import",
                "total": store.count_notes(),
                "error": "Parsed to nothing",
                "skipped": problems,
            },
        )

    written, kept = store.upsert_bib_entries(entries, replace=replace)

    notice = f"imported {written} entries from {source_name}"
    if kept > 0:
        notice += f" (kept {kept} entries still cited by notes)"

    return templates.TemplateResponse(
        request=request,
        name="import.html",
        context={
            "current_page": "import",
            "total": store.count_notes(),
            "notice": notice,
            "skipped": problems,
        },
    )


@app.get("/export", response_class=HTMLResponse)
def export(
    request: Request,
    store: StoreDep,
    tag: str = "",
    citekey: str = "",
    root: str = "",
) -> HTMLResponse:
    """Preview exports."""
    error = None
    markdown = ""
    root_id = None

    if root.strip():
        try:
            root_id = int(root.strip())
        except ValueError:
            error = "Root note ID must be a number"

    if not error and root_id is not None and (tag or citekey):
        error = "Root note ID cannot be combined with Tag or Citekey"
    elif not error:
        try:
            if root_id is not None:
                notes = store.thread(root_id)
            else:
                notes = store.list_notes(
                    tag=tag if tag else None,
                    citekey=citekey if citekey else None,
                    limit=-1,
                )

            roots = build_tree(notes)
            markdown = render_markdown(roots)
        except ValueError as exc:
            error = str(exc)

    return templates.TemplateResponse(
        request=request,
        name="export.html",
        context={
            "current_page": "export",
            "total": store.count_notes(),
            "tag": tag,
            "citekey": citekey,
            "root": root,
            "markdown": markdown,
            "error": error,
        },
    )




@app.get("/export/download")
def export_download(
    request: Request,
    store: StoreDep,
    tag: str = "",
    citekey: str = "",
    root: str = "",
) -> Response:
    """Download exported notes as Markdown."""
    error = None
    root_id = None

    if root.strip():
        try:
            root_id = int(root.strip())
        except ValueError:
            error = "Root note ID must be a number"

    if not error and root_id is not None and (tag or citekey):
        error = "Root note ID cannot be combined with Tag or Citekey"

    if error:
        return templates.TemplateResponse(
            request=request,
            name="export.html",
            context={
                "current_page": "export",
                "total": store.count_notes(),
                "tag": tag,
                "citekey": citekey,
                "root": root,
                "error": error,
            },
        )

    try:

        def sanitize(s: str) -> str:
            # \w is Unicode-aware by default, so non-ASCII survives straight
            # into the header -- and Starlette encodes header values as
            # latin-1, which a bare Japanese or accented character cannot
            # satisfy. re.ASCII restricts \w to plain ASCII word characters.
            return re.sub(r"[^\w\-]", "_", s, flags=re.ASCII)

        if root_id is not None:
            notes = store.thread(root_id)
            filename = f"thread_{root_id}.md"
        else:
            notes = store.list_notes(
                tag=tag if tag else None, citekey=citekey if citekey else None, limit=-1
            )
            if tag and citekey:
                filename = f"export_{sanitize(tag)}_{sanitize(citekey)}.md"
            elif tag:
                filename = f"export_{sanitize(tag)}.md"
            elif citekey:
                filename = f"export_{sanitize(citekey)}.md"
            else:
                filename = "export_all.md"

        roots = build_tree(notes)
        markdown = render_markdown(roots)
    except ValueError as exc:
        return templates.TemplateResponse(
            request=request,
            name="export.html",
            context={
                "current_page": "export",
                "total": store.count_notes(),
                "tag": tag,
                "citekey": citekey,
                "root": root,
                "error": str(exc),
            },
        )

    return Response(
        content=markdown,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
