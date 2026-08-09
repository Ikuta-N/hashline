"""FastAPI + HTMX adapter.

Holds no note logic: it resolves a request into a store call and renders the
result. A fresh connection is opened per request, because a sqlite3 connection
must not be shared across the threads FastAPI runs sync handlers on.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any, Final

from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from hashline.models import DEFAULT_READING_TAG, Context, Note
from hashline.store import Store, default_db_path

_HERE: Final = Path(__file__).parent

templates = Jinja2Templates(directory=str(_HERE / "templates"))

app = FastAPI(title="hashline")
app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")


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
) -> HTMLResponse:
    """The whole page: composer, tag sidebar and timeline."""
    context_data = {
        "current_page": "notes",
        "q": q,
        "tag": tag,
        "tags": store.list_tags(limit=30),
        "notes": _timeline(store, q=q, tag=tag),
        "total": store.count_notes(),
    }
    context_data.update(_context_data(store))
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=context_data,
    )


@app.get("/notes", response_class=HTMLResponse)
def notes(
    request: Request,
    store: StoreDep,
    q: str = "",
    tag: str = "",
) -> HTMLResponse:
    """Just the timeline, for HTMX to swap in."""
    return templates.TemplateResponse(
        request=request,
        name="_timeline.html",
        context={"notes": _timeline(store, q=q, tag=tag), "q": q, "tag": tag},
    )


@app.post("/notes", response_class=HTMLResponse)
def create_note(
    request: Request,
    store: StoreDep,
    body: Annotated[str, Form()],
    tag: Annotated[str, Form()] = "",
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
            store.add_note_with_context(body)
        except ValueError as exc:
            error = str(exc)
    return templates.TemplateResponse(
        request=request,
        name="_timeline.html",
        context={
            "notes": _timeline(store, q="", tag=tag),
            "q": "",
            "tag": tag,
            "error": error,
        },
    )


def _timeline(
    store: Store, *, q: str, tag: str, limit: int = 50
) -> list[tuple[Note, list[str]]]:
    """Notes plus their tags, either searched or listed."""
    filter_tag = tag or None
    if q.strip():
        found = [hit.note for hit in store.search_notes(q, tag=filter_tag, limit=limit)]
    else:
        found = store.list_notes(tag=filter_tag, limit=limit)
    return [(note, store.tags_for_note(note.id)) for note in found]


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
    request: Request, store: StoreDep, tag: Annotated[str, Form()] = ""
) -> HTMLResponse:
    """Set pinned tags and preserve the citekey."""
    error = None
    if tag.strip():
        try:
            current = store.get_context()
            store.set_context(Context(tags=tuple(tag.split()), citekey=current.citekey))
        except ValueError as exc:
            error = str(exc)
    return templates.TemplateResponse(
        request=request, name="_context.html", context=_context_data(store, error=error)
    )


@app.post("/context/read", response_class=HTMLResponse)
def read_context(
    request: Request,
    store: StoreDep,
    citekey: Annotated[str, Form()],
    tag: Annotated[str, Form()] = "",
) -> HTMLResponse:
    """Pin a citekey, optionally with a tag."""
    error = None
    if store.get_bib_entry(citekey) is None:
        error = (
            f"no bibliography entry for citekey {citekey!r}; "
            "import it first with `hashline bib import`"
        )
    else:
        try:
            tag_name = tag.strip() or DEFAULT_READING_TAG
            store.set_context(Context(tags=(tag_name,), citekey=citekey))
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


@app.get("/bib", response_class=HTMLResponse)
def bib(request: Request, store: StoreDep) -> HTMLResponse:
    """Bibliography management (stub)."""
    return templates.TemplateResponse(
        request=request,
        name="bib.html",
        context={"current_page": "bib", "total": store.count_notes()},
    )


@app.get("/import", response_class=HTMLResponse)
def import_(request: Request, store: StoreDep) -> HTMLResponse:
    """Import notes from files (stub)."""
    return templates.TemplateResponse(
        request=request,
        name="import.html",
        context={"current_page": "import", "total": store.count_notes()},
    )


@app.get("/export", response_class=HTMLResponse)
def export(request: Request, store: StoreDep) -> HTMLResponse:
    """Export notes (stub)."""
    return templates.TemplateResponse(
        request=request,
        name="export.html",
        context={"current_page": "export", "total": store.count_notes()},
    )
