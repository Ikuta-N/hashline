"""FastAPI + HTMX adapter.

Holds no note logic: it resolves a request into a store call and renders the
result. A fresh connection is opened per request, because a sqlite3 connection
must not be shared across the threads FastAPI runs sync handlers on.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Final

from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from hashline.models import Note
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
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "q": q,
            "tag": tag,
            "tags": store.list_tags(limit=30),
            "notes": _timeline(store, q=q, tag=tag),
            "total": store.count_notes(),
        },
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
    """Capture a note, then hand back the refreshed timeline."""
    try:
        store.add_note(body)
    except ValueError:
        pass  # an empty submission just re-renders the timeline
    return templates.TemplateResponse(
        request=request,
        name="_timeline.html",
        context={"notes": _timeline(store, q="", tag=tag), "q": "", "tag": tag},
    )


def _timeline(
    store: Store, *, q: str, tag: str, limit: int = 50
) -> list[tuple[Note, list[str]]]:
    """Notes plus their tags, either searched or listed."""
    filter_tag = tag or None
    if q.strip():
        found = [
            hit.note for hit in store.search_notes(q, tag=filter_tag, limit=limit)
        ]
    else:
        found = store.list_notes(tag=filter_tag, limit=limit)
    return [(note, store.tags_for_note(note.id)) for note in found]
