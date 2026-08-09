"""Smoke tests for the web adapter: routes and wiring, not note logic."""

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import numpy as np
import pytest
from fastapi.testclient import TestClient

from hashline.models import BibEntry, Context
from hashline.store import Store
from hashline.web.app import app


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("HASHLINE_DB", str(tmp_path / "hashline.db"))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def seeded(client: TestClient, tmp_path: Path) -> TestClient:
    with Store.open(tmp_path / "hashline.db") as store:
        store.add_note("bm25 を調べた #sqlite")
        store.add_note("無関係なメモ #other")
    return client


def _hidden_field_values(html: str) -> dict[str, str]:
    """The name/value pairs of every rendered ``value="..."`` input.

    Used to simulate submitting exactly the fields a fragment actually
    renders, rather than fields we assume it should carry.
    """
    return dict(re.findall(r'name="(\w+)" value="([^"]*)"', html))


def _fail(reason: str) -> object:
    raise AssertionError(reason)


class _FakeEmbedder:
    """Returns one fixed vector, so no model is ever downloaded."""

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    def encode(self, texts: list[str]) -> "np.ndarray":
        return np.array([self._vector for _ in texts], dtype=np.float32)


class TestHtmxTargets:
    """Every ``hx-target="#id"`` on a full page must have a matching element.

    htmx silently drops the request (never sends it) when the target
    selector matches nothing on the page, so a button whose hx-target has
    no matching id is dead. ``/bib`` and ``/bib/{citekey}`` carry "Read"
    buttons targeting ``#context-strip``, an element only ``_context.html``
    renders -- and only ``index.html`` includes it.
    """

    @staticmethod
    def _seed(tmp_path: Path) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.upsert_bib_entries(
                [
                    BibEntry(
                        citekey="smith2020",
                        entry_type="article",
                        title="Test",
                        tag="smith2020",
                    )
                ]
            )
            store.add_note("a note #tag")

    @pytest.mark.parametrize(
        "route",
        ["/", "/bib", "/bib/smith2020", "/import", "/export", "/stats"],
    )
    def test_every_hx_target_has_a_matching_id_on_the_page(
        self, client: TestClient, tmp_path: Path, route: str
    ) -> None:
        self._seed(tmp_path)
        response = client.get(route)
        assert response.status_code == 200
        targets = set(re.findall(r'hx-target="#([\w-]+)"', response.text))
        ids = set(re.findall(r'id="([\w-]+)"', response.text))
        missing = sorted(targets - ids)
        assert not missing, (
            f"page {route!r} has hx-target(s) {missing} with no matching "
            f"id=... element rendered on that same page; htmx will refuse "
            f"to send those requests"
        )


class TestIndex:
    def test_renders_the_page(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "hashline" in response.text

    def test_shows_notes_and_their_tags(self, seeded: TestClient) -> None:
        body = seeded.get("/").text
        assert "bm25 を調べた #sqlite" in body
        assert "sqlite" in body

    def test_reports_an_empty_timeline(self, client: TestClient) -> None:
        assert "no notes yet" in client.get("/").text

    def test_filters_by_tag(self, seeded: TestClient) -> None:
        body = seeded.get("/", params={"tag": "sqlite"}).text
        assert "bm25 を調べた" in body
        assert "無関係なメモ" not in body

    def test_searches(self, seeded: TestClient) -> None:
        body = seeded.get("/", params={"q": "bm25"}).text
        assert "bm25 を調べた" in body
        assert "無関係なメモ" not in body

    def test_escapes_note_bodies(self, client: TestClient, tmp_path: Path) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.add_note("<script>alert(1)</script>")
        assert "<script>alert(1)</script>" not in client.get("/").text


class TestTimelineFragment:
    def test_roots_are_newest_first_in_the_fragment(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            ids = [store.add_note(f"note{i}").id for i in range(4)]
        body = client.get("/notes").text
        positions = [body.index(f'id="note-{note_id}"') for note_id in ids]
        # note3 (last created) must read first: newest-first, like the store.
        assert positions == sorted(positions, reverse=True)

    def test_roots_are_newest_first_on_the_index_page(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            ids = [store.add_note(f"note{i}").id for i in range(4)]
        body = client.get("/").text
        positions = [body.index(f'id="note-{note_id}"') for note_id in ids]
        assert positions == sorted(positions, reverse=True)

    def test_root_order_matches_hashline_list(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            for i in range(4):
                store.add_note(f"note{i}")
            expected_ids = [n.id for n in store.list_notes()]
        body = client.get("/notes").text
        seen_order = sorted(expected_ids, key=lambda i: body.index(f'id="note-{i}"'))
        assert seen_order == expected_ids

    def test_replies_under_one_parent_read_oldest_first(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            root = store.add_note("root")
            first = store.add_note("reply-a", parent_id=root.id)
            second = store.add_note("reply-b", parent_id=root.id)
        body = client.get("/notes").text
        # A conversation reads forward, oldest reply first.
        assert body.index(f'id="note-{first.id}"') < body.index(
            f'id="note-{second.id}"'
        )

    def test_returns_only_the_timeline(self, seeded: TestClient) -> None:
        body = seeded.get("/notes").text
        assert 'id="timeline"' in body
        assert "<html" not in body

    def test_search_narrows_the_fragment(self, seeded: TestClient) -> None:
        body = seeded.get("/notes", params={"q": "bm25"}).text
        assert "bm25 を調べた" in body
        assert "無関係なメモ" not in body

    def test_reports_no_matches(self, seeded: TestClient) -> None:
        assert "no matches" in seeded.get("/notes", params={"q": "zzzzz"}).text

    def test_replies_render_nested_under_their_parent(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.add_note("parent")
            store.add_note("reply", parent_id=1)
        body = client.get("/notes").text
        # The reply should be indented
        assert 'style="margin-left: 20px"' in body
        assert "reply" in body

    def test_search_results_are_flat(self, client: TestClient, tmp_path: Path) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.add_note("parent")
            store.add_note("reply to parent", parent_id=1)
        body = client.get("/notes", params={"q": "parent"}).text
        assert 'style="margin-left: 0px"' in body
        assert 'style="margin-left: 20px"' not in body

    def test_orphaned_reply_appears_as_root_when_filtering(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.add_note("parent")
            store.add_note("reply #tagged", parent_id=1)
        body = client.get("/notes", params={"tag": "tagged"}).text
        assert 'style="margin-left: 0px"' in body
        assert "reply" in body


class TestFilters:
    def test_the_tag_chips_nav_is_gone(self, seeded: TestClient) -> None:
        """The chip nav grew without bound and pushed the timeline off
        screen, so it was removed. ``/?tag=NAME`` must keep working
        though -- the hidden tag input and every htmx control's
        hx-include still carry it -- so that is what replaces the old
        "clicking a tag chip keeps working" coverage below.
        """
        page = seeded.get("/")
        assert page.status_code == 200
        assert 'class="tags"' not in page.text
        assert 'href="/?tag=' not in page.text, (
            "a tag-chip link is still being rendered somewhere on the page"
        )

    def test_tag_query_param_still_filters_the_timeline(
        self, seeded: TestClient
    ) -> None:
        page = seeded.get("/", params={"tag": "sqlite"})
        assert page.status_code == 200
        assert "bm25" in page.text
        assert "無関係なメモ" not in page.text
        # /?tag=NAME must keep working from outside the app (a bookmark, a
        # link elsewhere), and every htmx control still has to pick the
        # value up through hx-include, so the hidden field has to survive.
        assert 'name="tag" value="sqlite"' in page.text

    def test_filter_by_citekey(self, client: TestClient, tmp_path: Path) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.upsert_bib_entries(
                [
                    BibEntry(
                        citekey="smith2020",
                        title="Title",
                        author="",
                        year="",
                        tag="smith2020",
                        entry_type="",
                    )
                ]
            )
            store.set_context(Context(citekey="smith2020", tags=()))
            store.add_note_with_context("about smith")
            store.add_note("unrelated")
        body = client.get("/notes", params={"citekey": "smith2020"}).text
        assert "about smith" in body
        assert "unrelated" not in body

    def test_filter_by_tag_and_citekey(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.upsert_bib_entries(
                [
                    BibEntry(
                        citekey="smith2020",
                        title="Title",
                        author="",
                        year="",
                        tag="smith2020",
                        entry_type="",
                    )
                ]
            )
            store.set_context(Context(citekey="smith2020", tags=()))
            store.add_note_with_context("about smith #tag")
            store.add_note_with_context("about smith")
        body = client.get("/notes", params={"citekey": "smith2020", "tag": "tag"}).text
        assert "about smith #tag" in body
        assert "about smith</p>" not in body

    def test_roots_only_hides_replies(self, client: TestClient, tmp_path: Path) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.add_note("parent")
            store.add_note("reply note", parent_id=1)
        body = client.get("/notes", params={"roots_only": "true"}).text
        assert "parent" in body
        assert "reply note" not in body

    def test_limit_is_honoured(self, client: TestClient, tmp_path: Path) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            for i in range(5):
                store.add_note(f"note {i}")
        body = client.get("/notes", params={"limit": 2}).text
        assert "note 4" in body
        assert "note 3" in body
        assert "note 2" not in body

    def test_filters_survive_search(self, client: TestClient, tmp_path: Path) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.add_note("test")
        body = client.get("/", params={"citekey": "testkey", "roots_only": "true"}).text
        assert 'name="citekey" value="testkey"' in body
        assert 'name="roots_only" value="true" checked' in body

    def test_capture_keeps_the_citekey_filter(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.upsert_bib_entries(
                [
                    BibEntry(
                        citekey="smith2020",
                        title="Title",
                        author="",
                        year="",
                        tag="smith2020",
                        entry_type="",
                    )
                ]
            )
            store.add_note("about smith", citekey="smith2020")
            store.add_note("unrelated")
        response = client.post(
            "/notes", data={"body": "new note #other", "citekey": "smith2020"}
        )
        assert response.status_code == 200
        assert "about smith" in response.text
        assert "unrelated" not in response.text
        assert "new note" not in response.text

    def test_capture_keeps_the_roots_only_filter(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.add_note("parent")
            store.add_note("reply note", parent_id=1)
        response = client.post(
            "/notes", data={"body": "another note", "roots_only": "true"}
        )
        assert response.status_code == 200
        assert "reply note" not in response.text

    def test_delete_keeps_the_roots_only_filter(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.add_note("parent")
            store.add_note("reply note", parent_id=1)
            store.add_note("other root")
        response = client.post("/notes/3/delete", data={"roots_only": "true"})
        assert response.status_code == 200
        assert "reply note" not in response.text

    def test_capture_keeps_the_search_query(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.add_note("bm25 ranking notes")
            store.add_note("something else")
        response = client.post(
            "/notes", data={"body": "unrelated capture", "q": "bm25"}
        )
        assert response.status_code == 200
        assert "bm25 ranking notes" in response.text
        assert "something else" not in response.text
        assert "unrelated capture" not in response.text

    def test_reply_form_carries_the_active_tag_and_query(
        self, seeded: TestClient
    ) -> None:
        response = seeded.get("/notes/1/reply", params={"tag": "sqlite", "q": "bm25"})
        assert response.status_code == 200
        assert 'name="tag" value="sqlite"' in response.text
        assert 'name="q" value="bm25"' in response.text

    def test_index_renders_a_hidden_limit_field(self, client: TestClient) -> None:
        body = client.get("/", params={"limit": 3}).text
        assert 'name="limit" value="3"' in body, (
            "no hidden limit field was rendered, so a search, reply or "
            "delete triggered from this page will snap the limit back to "
            "the default of 50"
        )

    def test_search_reply_and_delete_carry_limit_in_their_hx_include(
        self, seeded: TestClient
    ) -> None:
        body = seeded.get("/", params={"limit": 3}).text

        search_input = re.search(
            r'<input[^>]*name="q"[^>]*hx-get="/notes"[^>]*>', body
        )
        assert search_input is not None, "search input not found"
        assert "[name='limit']" in search_input.group(0), (
            "the search input's hx-include does not mention limit, so "
            "typing a query snaps the timeline back to the default limit"
        )

        reply_button = re.search(r'<button hx-get="/notes/1/reply"[^>]*>', body)
        assert reply_button is not None, "reply button not found"
        assert "[name='limit']" in reply_button.group(0), (
            "the reply button's hx-include does not mention limit"
        )

        delete_button = re.search(r'<button hx-post="/notes/1/delete"[^>]*>', body)
        assert delete_button is not None, "delete button not found"
        assert "[name='limit']" in delete_button.group(0), (
            "the delete button's hx-include does not mention limit, so "
            "deleting a note snaps the timeline back to the default limit"
        )

    def test_notes_fragment_returns_exactly_limit_notes_when_more_exist(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        # This already passes today: the /notes route reads its own limit
        # query param directly. It is written here as a regression guard
        # alongside the hidden-field and hx-include fixes above, which are
        # what actually get that limit onto the request in the first place.
        with Store.open(tmp_path / "hashline.db") as store:
            for i in range(5):
                store.add_note(f"note {i}")
        body = client.get("/notes", params={"limit": 3}).text
        assert len(re.findall(r'class="note"', body)) == 3

    def test_no_filter_name_is_duplicated_as_a_form_field_on_the_index_page(
        self, client: TestClient
    ) -> None:
        # Same invariant as TestHtmxTargets: htmx's hx-include="[name='x']"
        # is a document-wide CSS selector, not scoped to one <form>. Every
        # element sharing a filter's name is a candidate value, and form
        # encoding keeps only the last one -- so two elements named
        # "roots_only" mean an unchecked checkbox can never win.
        body = client.get("/").text
        names = re.findall(r'<(?:input|select|textarea)\b[^>]*\bname="(\w+)"', body)
        counts: dict[str, int] = {}
        for name in names:
            if name in {"q", "tag", "citekey", "roots_only", "limit"}:
                counts[name] = counts.get(name, 0) + 1
        duplicates = {name: n for name, n in counts.items() if n > 1}
        assert not duplicates, (
            f"filter field name(s) {sorted(duplicates)} are rendered as a "
            f"form field more than once on '/'; hx-include=\"[name='...']\" "
            f"matches every one of them and form encoding keeps only the "
            f"last, so the wrong element's value wins"
        )


class TestCreateNote:
    def test_stores_the_note_and_returns_the_timeline(self, client: TestClient) -> None:
        response = client.post("/notes", data={"body": "新しいメモ #web"})
        assert response.status_code == 200
        assert "新しいメモ #web" in response.text
        assert "新しいメモ #web" in client.get("/").text

    def test_tags_come_from_the_body(self, client: TestClient) -> None:
        client.post("/notes", data={"body": "tagged #web"})
        assert "tagged #web" in client.get("/", params={"tag": "web"}).text

    def test_a_blank_submission_stores_nothing(self, client: TestClient) -> None:
        response = client.post("/notes", data={"body": "   "})
        assert response.status_code == 200
        assert "no notes yet" in client.get("/").text

    def test_applies_pinned_context(self, client: TestClient, tmp_path: Path) -> None:
        from hashline.models import Context

        with Store.open(tmp_path / "hashline.db") as store:
            store.set_context(Context(tags=("reading",)))

        response = client.post("/notes", data={"body": "a note"})
        assert response.status_code == 200
        # The returned HTML should show the 'reading' tag
        assert "reading" in response.text

    def test_shows_a_message_when_the_pinned_work_is_missing(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        from hashline.models import Context

        with Store.open(tmp_path / "hashline.db") as store:
            store.set_context(Context(citekey="smith2020"))
            # We explicitly do NOT insert smith2020 into bib_entries

        response = client.post("/notes", data={"body": "a note"})
        # 200 with the timeline, not a 4xx: HTMX swaps the response in, so an
        # error status would leave the user staring at an unchanged page.
        assert response.status_code == 200
        assert 'id="timeline"' in response.text
        assert "is no longer in the bibliography" in response.text

    def test_a_blank_submission_shows_no_error(self, client: TestClient) -> None:
        response = client.post("/notes", data={"body": "   "})
        assert response.status_code == 200
        assert 'class="error"' not in response.text

    def test_page_with_no_citekey_is_error(self, client: TestClient) -> None:
        response = client.post("/notes", data={"body": "a note", "page": "10"})
        assert response.status_code == 200
        assert "a page needs a pinned work" in response.text
        # The reader is looking at a form, not a terminal.
        assert "--page" not in response.text

    def test_no_context_ignores_pinned_context(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        from hashline.models import Context

        with Store.open(tmp_path / "hashline.db") as store:
            store.set_context(Context(tags=("reading",)))

        response = client.post("/notes", data={"body": "a note", "no_context": "on"})
        assert response.status_code == 200
        # The note should not have the reading tag
        assert "reading" not in response.text

    def test_no_context_with_page_is_error(self, client: TestClient) -> None:
        response = client.post(
            "/notes", data={"body": "a note", "page": "10", "no_context": "on"}
        )
        assert response.status_code == 200
        assert "a page needs a pinned work" in response.text
        assert "--no-context" not in response.text

    def test_extra_tags(self, client: TestClient) -> None:
        response = client.post("/notes", data={"body": "a note", "tags": "web ui"})
        assert response.status_code == 200
        assert "ui, web" in response.text

    def test_composer_does_not_duplicate_the_search_qs_hidden_field(
        self, client: TestClient
    ) -> None:
        # The composer used to mirror q as its own hidden field, frozen at
        # page-load time. Typing into the search box after that never
        # touched the composer's copy, so capturing a note swapped in the
        # unfiltered timeline. The fix is for the composer to hx-include
        # the live search input instead of carrying its own q.
        body = client.get("/").text
        composer = re.search(
            r'<form class="composer"\s+hx-post="/notes".*?</form>', body, re.S
        )
        assert composer is not None, "composer form not found"
        markup = composer.group(0)
        assert 'name="q"' not in markup, (
            "the composer still renders its own hidden q field frozen at "
            "page-load time instead of reading the live search box"
        )
        assert "hx-include" in markup, (
            "the composer form has no hx-include, so it cannot pick up "
            "the live search input's value at submit time"
        )
        assert "[name='q']" in markup, (
            "the composer's hx-include does not reference the search "
            "input's name"
        )

    def test_composer_body_is_a_textarea_with_a_ctrl_enter_submit_handler(
        self, client: TestClient
    ) -> None:
        # A plain <input type=text name=body> can never hold a line break,
        # so Enter has to become a newline and Ctrl/Cmd+Enter has to submit
        # instead. requestSubmit() is required here, not form.submit(),
        # because only requestSubmit() fires the cancelable submit event
        # htmx listens for.
        body = client.get("/").text
        textarea = re.search(r"<textarea\b[^>]*name=\"body\"[^>]*>", body)
        assert textarea is not None, "composer body field is not a textarea"
        markup = textarea.group(0)
        assert "onkeydown" in markup, "composer textarea has no keydown handler"
        assert "ctrlKey" in markup and "metaKey" in markup, (
            "composer textarea's keydown handler does not check for "
            "Ctrl/Cmd+Enter"
        )
        assert "requestSubmit()" in markup, (
            "composer textarea must call requestSubmit(), not submit(), so "
            "htmx's submit listener fires"
        )
        assert 'autofocus' in markup
        assert "Ctrl+Enter" in markup or "Ctrl" in markup


class TestReply:
    def test_renders_reply_fragment(self, seeded: TestClient) -> None:
        response = seeded.get("/notes/1/reply")
        assert response.status_code == 200
        assert 'name="parent_id" value="1"' in response.text

    def test_reply_body_is_a_textarea_with_a_ctrl_enter_submit_handler(
        self, seeded: TestClient
    ) -> None:
        body = seeded.get("/notes/1/reply").text
        textarea = re.search(r"<textarea\b[^>]*name=\"body\"[^>]*>", body)
        assert textarea is not None, "reply body field is not a textarea"
        markup = textarea.group(0)
        assert "onkeydown" in markup, "reply textarea has no keydown handler"
        assert "ctrlKey" in markup and "metaKey" in markup, (
            "reply textarea's keydown handler does not check for "
            "Ctrl/Cmd+Enter"
        )
        assert "requestSubmit()" in markup, (
            "reply textarea must call requestSubmit(), not submit(), so "
            "htmx's submit listener fires"
        )

    def test_reply_form_has_a_cancel_button(self, seeded: TestClient) -> None:
        # hx-swap="afterend" leaves the reply form sitting under the note
        # with no way to get rid of it short of submitting or reloading.
        body = seeded.get("/notes/1/reply").text
        cancel = re.search(r'<button type="button"[^>]*>\s*cancel\s*</button>', body)
        assert cancel is not None, "reply fragment has no cancel button"
        assert "this.closest('form').remove()" in cancel.group(0), (
            "cancel button does not remove its own form"
        )

    def test_reply_textarea_dismisses_the_form_on_escape(
        self, seeded: TestClient
    ) -> None:
        body = seeded.get("/notes/1/reply").text
        textarea = re.search(r"<textarea\b[^>]*name=\"body\"[^>]*>", body)
        assert textarea is not None
        markup = textarea.group(0)
        assert "Escape" in markup, (
            "reply textarea's keydown handler does not dismiss on Escape"
        )
        assert ".remove()" in markup, (
            "reply textarea's Escape handler does not remove the form"
        )

    def test_reply_fragment_carries_citekey_roots_only_and_limit(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.upsert_bib_entries(
                [
                    BibEntry(
                        citekey="smith2020",
                        entry_type="article",
                        title="Test",
                        tag="smith2020",
                    )
                ]
            )
            note = store.add_note("about smith", citekey="smith2020")
        response = client.get(
            f"/notes/{note.id}/reply",
            params={"citekey": "smith2020", "roots_only": "true", "limit": 5},
        )
        assert response.status_code == 200
        assert 'name="citekey" value="smith2020"' in response.text
        assert 'name="roots_only" value="true"' in response.text
        assert 'name="limit" value="5"' in response.text

    def test_reply_button_carries_hx_include_for_the_active_filters(
        self, seeded: TestClient
    ) -> None:
        body = seeded.get("/notes").text
        match = re.search(
            r'<button hx-get="/notes/1/reply"[^>]*>', body
        )
        assert match is not None, "reply button not found in the timeline"
        button = match.group(0)
        assert 'hx-include="' in button, (
            "reply button has no hx-include, so its POST will not carry "
            "the active q/tag/citekey/roots_only filters"
        )
        for name in ("q", "tag", "citekey", "roots_only"):
            assert f"[name='{name}']" in button, (
                f"reply button's hx-include is missing [name='{name}']"
            )

    def test_posting_a_reply_from_the_rendered_fragment_keeps_the_citekey_filter(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.upsert_bib_entries(
                [
                    BibEntry(
                        citekey="smith2020",
                        entry_type="article",
                        title="Test",
                        tag="smith2020",
                    )
                ]
            )
            about = store.add_note("about smith", citekey="smith2020")
            store.add_note("unrelated")
        fragment = client.get(
            f"/notes/{about.id}/reply", params={"citekey": "smith2020"}
        ).text
        fields = _hidden_field_values(fragment)
        fields["body"] = "a reply"
        response = client.post("/notes", data=fields)
        assert response.status_code == 200
        assert "about smith" in response.text
        assert "unrelated" not in response.text

    def test_posting_a_reply_from_the_rendered_fragment_keeps_roots_only(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            root = store.add_note("root note")
            store.add_note("existing reply", parent_id=root.id)
        fragment = client.get(
            f"/notes/{root.id}/reply", params={"roots_only": "true"}
        ).text
        fields = _hidden_field_values(fragment)
        fields["body"] = "new reply"
        response = client.post("/notes", data=fields)
        assert response.status_code == 200
        assert "existing reply" not in response.text
        assert 'style="margin-left: 20px"' not in response.text

    def test_creates_a_reply(self, seeded: TestClient) -> None:
        response = seeded.post("/notes", data={"body": "a reply", "parent_id": "1"})
        assert response.status_code == 200
        assert "a reply" in response.text

    def test_unknown_parent_is_error(self, seeded: TestClient) -> None:
        response = seeded.post("/notes", data={"body": "a reply", "parent_id": "999"})
        assert response.status_code == 200
        assert "does not exist" in response.text


class TestThreadView:
    def test_renders_the_subtree(self, client: TestClient, tmp_path: Path) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.add_note("root")
            store.add_note("child 1", parent_id=1)
            store.add_note("grandchild", parent_id=2)
            store.add_note("child 2", parent_id=1)
            store.add_note("unrelated")

        response = client.get("/notes/1/thread")
        assert response.status_code == 200
        body = response.text
        assert "root" in body
        assert "grandchild" in body
        assert "unrelated" not in body

    def test_unknown_id_answers_200_with_an_error_not_a_404(
        self, client: TestClient
    ) -> None:
        # htmx does not swap a non-2xx response, so a stale thread button
        # (the note behind it deleted from another tab, say) would leave the
        # page sitting there doing nothing. The existing error slot in
        # _timeline.html renders instead.
        response = client.get("/notes/999/thread")
        assert response.status_code == 200
        assert 'class="error"' in response.text

    def test_thread_button_carries_hx_include_for_the_active_filters(
        self, seeded: TestClient
    ) -> None:
        body = seeded.get("/notes").text
        match = re.search(r'<button hx-get="/notes/1/thread"[^>]*>', body)
        assert match is not None, "thread button not found in the timeline"
        button = match.group(0)
        assert 'hx-include="' in button, (
            "thread button has no hx-include, so following it drops the "
            "active q/tag/citekey/roots_only/limit filters"
        )
        for name in ("q", "tag", "citekey", "roots_only", "limit"):
            assert f"[name='{name}']" in button, (
                f"thread button's hx-include is missing [name='{name}']"
            )

    def test_thread_view_accepts_the_filter_query_params(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        # _timeline.html itself renders no hidden q/tag/citekey/roots_only/
        # limit fields -- those live in index.html, outside the swapped
        # #timeline -- so what this proves is that thread() takes the same
        # filter params as every other timeline-returning route and passes
        # them into the back control (checked separately below) instead of
        # rejecting them or dropping them silently.
        with Store.open(tmp_path / "hashline.db") as store:
            store.add_note("root #keep")
        response = client.get(
            "/notes/1/thread",
            params={
                "q": "root",
                "tag": "keep",
                "citekey": "",
                "roots_only": "true",
                "limit": 5,
            },
        )
        assert response.status_code == 200
        assert "root" in response.text

    def test_thread_view_has_a_back_control_with_the_active_filters(
        self, seeded: TestClient
    ) -> None:
        response = seeded.get("/notes/1/thread", params={"tag": "sqlite"})
        assert response.status_code == 200
        back = re.search(r'<button hx-get="/notes"[^>]*>', response.text)
        assert back is not None, "thread view has no back-to-notes control"
        button = back.group(0)
        assert 'hx-target="#timeline"' in button
        assert 'hx-swap="outerHTML"' in button
        for name in ("q", "tag", "citekey", "roots_only", "limit"):
            assert f"[name='{name}']" in button, (
                f"back control's hx-include is missing [name='{name}']"
            )
        assert "all notes" in response.text

    def test_pressing_back_restores_exactly_the_filtered_timeline(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.add_note("keep this #keep")
            store.add_note("drop this #other")
        # What the filtered index page looked like before the user opened
        # a thread from inside it.
        before = client.get("/", params={"tag": "keep"}).text
        assert "keep this" in before
        assert "drop this" not in before

        client.get("/notes/1/thread", params={"tag": "keep"})

        # The back control's hx-include names q/tag/citekey/roots_only/limit,
        # which htmx reads from the fields still on the page (outside the
        # swapped #timeline) -- so pressing it fires exactly this request.
        after = client.get("/notes", params={"tag": "keep"}).text
        assert "keep this" in after
        assert "drop this" not in after

    def test_ordinary_timeline_fragments_render_no_back_control(
        self, seeded: TestClient
    ) -> None:
        # _timeline.html is also rendered by routes that pass no thread_root;
        # Jinja renders that as falsey, so nothing should change for them.
        body = seeded.get("/notes").text
        assert "all notes" not in body
        assert 'hx-get="/notes"' not in body

    def test_thread_view_has_an_export_this_thread_link(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.add_note("root")
        response = client.get("/notes/1/thread")
        assert response.status_code == 200
        assert 'href="/export?root=1"' in response.text
        assert "export this thread" in response.text


class TestDeleteNote:
    def test_deleting_leaf_works_and_reports_count(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.add_note("to delete")
        response = client.post("/notes/1/delete")
        assert response.status_code == 200
        assert "deleted 1 note" in response.text
        assert "to delete" not in response.text
        # A completed deletion is not a failure, so it must not land in the
        # error box.
        assert 'class="notice"' in response.text
        assert 'class="error"' not in response.text

    def test_deleting_parent_answers_200_and_offers_recursive(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.add_note("parent")
            store.add_note("child", parent_id=1)
        response = client.post("/notes/1/delete")
        assert response.status_code == 200
        assert "note 1 has 1 reply" in response.text
        assert "parent" in response.text  # keeps the note
        assert 'name="recursive" value="true"' in response.text  # offers recursive
        assert 'class="error"' in response.text
        # The reader of this message has a button, not a command-line flag.
        assert "--recursive" not in response.text

    def test_recursive_delete_removes_whole_thread(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.add_note("parent")
            store.add_note("child", parent_id=1)
        response = client.post("/notes/1/delete", data={"recursive": "true"})
        assert response.status_code == 200
        assert "deleted 2 notes" in response.text
        assert "parent" not in response.text

    def test_replies_guard_retry_form_carries_the_active_filters(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.upsert_bib_entries(
                [
                    BibEntry(
                        citekey="smith2020",
                        entry_type="article",
                        title="Test",
                        tag="smith2020",
                    )
                ]
            )
            store.add_note("parent", citekey="smith2020")
            store.add_note("child", parent_id=1, citekey="smith2020")
        response = client.post(
            "/notes/1/delete",
            data={"citekey": "smith2020", "roots_only": "true"},
        )
        assert response.status_code == 200
        assert 'name="citekey" value="smith2020"' in response.text, (
            "the retry form's hidden citekey field is empty; the active "
            "citekey filter was dropped from the context"
        )
        assert 'name="roots_only" value="true"' in response.text, (
            "the retry form's hidden roots_only field is empty; the active "
            "roots_only filter was dropped from the context"
        )

    def test_submitting_the_retry_form_deletes_the_thread_and_keeps_the_citekey_filter(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.upsert_bib_entries(
                [
                    BibEntry(
                        citekey="smith2020",
                        entry_type="article",
                        title="Test",
                        tag="smith2020",
                    )
                ]
            )
            store.add_note("parent", citekey="smith2020")
            store.add_note("child", parent_id=1, citekey="smith2020")
            store.add_note("unrelated")
        first = client.post("/notes/1/delete", data={"citekey": "smith2020"})
        fields = _hidden_field_values(first.text)
        response = client.post("/notes/1/delete", data=fields)
        assert response.status_code == 200
        assert "deleted 2 notes" in response.text
        assert "unrelated" not in response.text, (
            "the retry form dropped the citekey filter, so an unfiltered "
            "timeline (including notes outside the citekey) came back"
        )
        with Store.open(tmp_path / "hashline.db") as store:
            assert store.count_notes() == 1


class TestStatic:
    def test_htmx_is_served_locally(self, client: TestClient) -> None:
        response = client.get("/static/htmx.min.js")
        assert response.status_code == 200
        assert "htmx" in response.text[:200]


class TestNav:
    def test_notes_is_current_on_index(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert 'href="/" class="current"' in response.text


class TestContext:
    def test_get_context_strip_no_context(self, client: TestClient) -> None:
        response = client.get("/context")
        assert response.status_code == 200
        assert "context-strip" in response.text
        assert 'name="context_tag"' in response.text
        assert 'name="context_citekey"' in response.text

    def test_pin_tags(self, client: TestClient, tmp_path: Path) -> None:
        response = client.post("/context/pin", data={"context_tag": "research urgent"})
        assert response.status_code == 200
        assert "research, urgent" in response.text

        # Pinning tags must not pin a work. This used to be checked by asserting
        # the citekey input was absent from the markup, which quietly required
        # the read form to disappear the moment any tag was pinned -- the bug
        # test_context_still_offers_the_read_form_with_only_tags_pinned fixes.
        with Store.open(tmp_path / "hashline.db") as store:
            context = store.get_context()
        assert context.tags == ("research", "urgent")
        assert context.citekey is None

        # Verify it persisted
        assert "research, urgent" in client.get("/context").text

    def test_pin_invalid_tag(self, client: TestClient) -> None:
        response = client.post("/context/pin", data={"context_tag": "invalid@tag"})
        assert response.status_code == 200
        assert 'class="error"' in response.text

    def test_pinning_an_empty_tag_box_clears_the_tags_and_keeps_the_citekey(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.upsert_bib_entries(
                [
                    BibEntry(
                        citekey="smith2020",
                        tag="smith2020",
                        entry_type="article",
                        title="Test",
                    )
                ]
            )
        client.post(
            "/context/read",data={
                    "context_citekey": "smith2020",
                    "context_tag": "reading",
                }
        )
        client.post("/context/pin", data={"context_tag": "extra"})
        # The pin box was pre-filled with "extra"; the user cleared it by
        # hand and pressed pin, expecting the tags to be gone.
        response = client.post("/context/pin", data={"context_tag": ""})
        assert response.status_code == 200
        with Store.open(tmp_path / "hashline.db") as store:
            context = store.get_context()
        assert context.tags == ()
        assert context.citekey == "smith2020"

    def test_read_start(self, client: TestClient, tmp_path: Path) -> None:
        from hashline.models import BibEntry

        with Store.open(tmp_path / "hashline.db") as store:
            store.upsert_bib_entries(
                [
                    BibEntry(
                        citekey="smith2020",
                        tag="smith2020",
                        entry_type="article",
                        title="Test",
                    )
                ]
            )

        response = client.post(
            "/context/read", data={"context_citekey": "smith2020", "context_tag": ""}
        )
        assert response.status_code == 200
        assert "smith2020" in response.text
        assert "Test" in response.text
        assert "reading" in response.text

        # Verify it persisted
        assert "smith2020" in client.get("/context").text

    def test_starting_a_read_keeps_previously_pinned_tags(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        # The tag strip and the pinned work each have their own clear
        # button, presenting them to the reader as independent. Starting a
        # read must not silently wipe out tags pinned before it.
        with Store.open(tmp_path / "hashline.db") as store:
            store.upsert_bib_entries(
                [
                    BibEntry(
                        citekey="smith2020",
                        tag="smith2020",
                        entry_type="article",
                        title="Test",
                    )
                ]
            )
        client.post("/context/pin", data={"context_tag": "research urgent"})
        response = client.post("/context/read", data={"context_citekey": "smith2020"})
        assert response.status_code == 200
        with Store.open(tmp_path / "hashline.db") as store:
            context = store.get_context()
        assert context.citekey == "smith2020"
        assert set(context.tags) == {"research", "urgent", "reading"}, (
            f"expected the reading tag alongside the previously pinned "
            f"tags, got {context.tags!r}"
        )

    def test_read_start_custom_tag(self, client: TestClient, tmp_path: Path) -> None:
        from hashline.models import BibEntry

        with Store.open(tmp_path / "hashline.db") as store:
            store.upsert_bib_entries(
                [
                    BibEntry(
                        citekey="smith2020",
                        tag="smith2020",
                        entry_type="article",
                        title="Test",
                    )
                ]
            )

        response = client.post(
            "/context/read",data={
                    "context_citekey": "smith2020",
                    "context_tag": "annotating",
                }
        )
        assert response.status_code == 200
        assert "annotating" in response.text

    def test_read_unknown_citekey(self, client: TestClient) -> None:
        response = client.post(
            "/context/read", data={"context_citekey": "nope", "context_tag": ""}
        )
        assert response.status_code == 200
        assert "no bibliography entry for citekey" in response.text

    def test_read_invalid_tag(self, client: TestClient, tmp_path: Path) -> None:
        from hashline.models import BibEntry

        with Store.open(tmp_path / "hashline.db") as store:
            store.upsert_bib_entries(
                [
                    BibEntry(
                        citekey="smith2020",
                        tag="smith2020",
                        entry_type="article",
                        title="Test",
                    )
                ]
            )

        response = client.post(
            "/context/read",data={
                    "context_citekey": "smith2020",
                    "context_tag": "invalid@tag",
                }
        )
        assert response.status_code == 200
        assert 'class="error"' in response.text

    def test_clear_context(self, client: TestClient) -> None:
        client.post("/context/pin", data={"context_tag": "research"})
        response = client.post("/context/clear")
        assert response.status_code == 200
        assert "research" not in response.text
        assert 'name="context_citekey"' in response.text

    def test_pin_tags_form_still_offered_once_a_work_is_pinned(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.upsert_bib_entries(
                [
                    BibEntry(
                        citekey="smith2020",
                        tag="smith2020",
                        entry_type="article",
                        title="Test",
                    )
                ]
            )
        client.post(
            "/context/read", data={"context_citekey": "smith2020", "context_tag": ""}
        )
        # Without this form the reader has no way to change tags once reading.
        response = client.get("/context")
        assert response.status_code == 200
        assert 'hx-post="/context/pin"' in response.text

    def test_context_still_offers_the_read_form_with_only_tags_pinned(
        self, client: TestClient
    ) -> None:
        client.post("/context/pin", data={"context_tag": "idea"})
        response = client.get("/context")
        assert response.status_code == 200
        # Pinning a tag must not hide the only way to start reading a work.
        assert 'hx-post="/context/read"' in response.text

    def test_index_still_offers_the_read_form_with_only_tags_pinned(
        self, client: TestClient
    ) -> None:
        client.post("/context/pin", data={"context_tag": "idea"})
        response = client.get("/")
        assert response.status_code == 200
        assert 'hx-post="/context/read"' in response.text

    def test_clear_tags_keeps_the_pinned_citekey(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.upsert_bib_entries(
                [
                    BibEntry(
                        citekey="smith2020",
                        tag="smith2020",
                        entry_type="article",
                        title="Test",
                    )
                ]
            )
        client.post(
            "/context/read",data={
                    "context_citekey": "smith2020",
                    "context_tag": "reading",
                }
        )
        client.post("/context/pin", data={"context_tag": "extra"})
        response = client.post("/context/clear_tags")
        assert response.status_code == 200
        # Assert on the stored context, not on the HTML: the strip's own copy
        # mentions tag names (the pin field's placeholder says "reading"), so a
        # substring check would pass or fail for the wrong reason.
        with Store.open(tmp_path / "hashline.db") as store:
            context = store.get_context()
        assert context.citekey == "smith2020"
        assert context.tags == ()

    def test_clear_read_keeps_the_pinned_tags(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.upsert_bib_entries(
                [
                    BibEntry(
                        citekey="smith2020",
                        tag="smith2020",
                        entry_type="article",
                        title="Test",
                    )
                ]
            )
        client.post(
            "/context/read",data={
                    "context_citekey": "smith2020",
                    "context_tag": "reading",
                }
        )
        response = client.post("/context/clear_read")
        assert response.status_code == 200
        with Store.open(tmp_path / "hashline.db") as store:
            context = store.get_context()
        assert context.citekey is None
        assert context.tags == ("reading",)

    def test_clear_still_drops_both_the_work_and_the_tags(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.upsert_bib_entries(
                [
                    BibEntry(
                        citekey="smith2020",
                        tag="smith2020",
                        entry_type="article",
                        title="Test",
                    )
                ]
            )
        client.post(
            "/context/read",data={
                    "context_citekey": "smith2020",
                    "context_tag": "reading",
                }
        )
        response = client.post("/context/clear")
        assert response.status_code == 200
        with Store.open(tmp_path / "hashline.db") as store:
            context = store.get_context()
        assert context.citekey is None
        assert context.tags == ()

    def test_read_start_with_a_hash_prefixed_tag_does_not_duplicate_it(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """``#reading`` and ``reading`` are the same pinned tag.

        The read form sits in a strip about ``#tags``, so typing the ``#``
        is the natural gesture. The merge that keeps pinned tags when a
        read starts compares the raw input against tags the store has
        already normalized, so the hash-prefixed spelling misses the check
        and is appended a second time. It compounds: the pin box is
        re-rendered from those tags, so re-pinning keeps the duplicate.
        """
        with Store.open(tmp_path / "hashline.db") as store:
            store.upsert_bib_entries(
                [
                    BibEntry(
                        citekey="smith2020",
                        entry_type="article",
                        title="Test",
                        tag="smith2020",
                    )
                ]
            )
        client.post(
            "/context/read",
            data={"context_citekey": "smith2020", "context_tag": "reading"},
        )
        client.post("/context/clear_read")
        client.post(
            "/context/read",
            data={"context_citekey": "smith2020", "context_tag": "#reading"},
        )
        with Store.open(tmp_path / "hashline.db") as store:
            assert store.get_context().tags == ("reading",)


class TestBib:
    def test_bib_list_empty(self, client: TestClient) -> None:
        response = client.get("/bib")
        assert response.status_code == 200
        assert "Library is empty" in response.text
        assert 'href="/import"' in response.text

    def test_bib_list_with_entries(self, client: TestClient, tmp_path: Path) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.upsert_bib_entries(
                [
                    BibEntry(
                        citekey="smith2020",
                        entry_type="article",
                        title="A title",
                        author="Smith",
                        year="2020",
                        raw="...",
                        tag="smith2020",
                    )
                ]
            )
        response = client.get("/bib")
        assert response.status_code == 200
        # The list identifies an entry by what a person recognises it by --
        # title, author, year -- not by its citekey, which is a lookup key.
        # The citekey survives only as the destination of the title link.
        assert "A title" in response.text
        assert "Smith" in response.text
        assert "2020" in response.text
        assert '<th style="padding: 0.5rem;">Citekey</th>' not in response.text
        assert 'href="/bib/smith2020">A title</a>' in response.text

    def test_bib_list_entry_with_no_title_is_still_clickable(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        # The link moved onto the title text. A malformed BibTeX record
        # with no title must fall back to '(no title)' as the link text,
        # not disappear -- otherwise it becomes unreachable from the UI.
        with Store.open(tmp_path / "hashline.db") as store:
            store.upsert_bib_entries(
                [
                    BibEntry(
                        citekey="notitle2020",
                        entry_type="misc",
                        tag="notitle2020",
                    )
                ]
            )
        response = client.get("/bib")
        assert response.status_code == 200
        assert 'href="/bib/notitle2020">(no title)</a>' in response.text

    def test_bib_detail_found(self, client: TestClient, tmp_path: Path) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.upsert_bib_entries(
                [
                    BibEntry(
                        citekey="smith2020",
                        entry_type="article",
                        title="A title",
                        author="Smith",
                        year="2020",
                        raw="RAW_BIBTEX_CONTENT",
                        tag="smith2020",
                    )
                ]
            )
        response = client.get("/bib/smith2020")
        assert response.status_code == 200
        assert "RAW_BIBTEX_CONTENT" in response.text
        assert "smith2020" in response.text

    def test_bib_detail_not_found(self, client: TestClient) -> None:
        response = client.get("/bib/unknown")
        assert response.status_code == 404

    def test_bib_list_read_button_posts_to_the_fragment_route(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.upsert_bib_entries(
                [
                    BibEntry(
                        citekey="smith2020",
                        entry_type="article",
                        title="A title",
                        tag="smith2020",
                    )
                ]
            )
        response = client.get("/bib")
        assert response.status_code == 200
        assert 'hx-post="/context/read"' in response.text
        # A plain form submit navigates away and renders a bare fragment.
        assert 'action="/context/read"' not in response.text

    def test_bib_detail_read_button_posts_to_the_fragment_route(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.upsert_bib_entries(
                [
                    BibEntry(
                        citekey="smith2020",
                        entry_type="article",
                        title="A title",
                        tag="smith2020",
                    )
                ]
            )
        response = client.get("/bib/smith2020")
        assert response.status_code == 200
        assert 'hx-post="/context/read"' in response.text
        assert 'action="/context/read"' not in response.text


class TestImport:
    def test_import_notes_path(self, client: TestClient, tmp_path: Path) -> None:
        notes_file = tmp_path / "notes.txt"
        notes_file.write_text("a note\n")
        response = client.post("/import", data={"path": str(notes_file)})
        assert response.status_code == 200
        assert "imported 1 notes from 1 files" in response.text

    def test_import_notes_missing_path(self, client: TestClient) -> None:
        response = client.post("/import", data={"path": "/does/not/exist"})
        assert response.status_code == 200
        assert "no such file or directory: /does/not/exist" in response.text

    def test_import_bib_path(self, client: TestClient, tmp_path: Path) -> None:
        bib_file = tmp_path / "library.bib"
        bib_file.write_text("@article{smith2020, title={A title}}")
        response = client.post("/bib/import", data={"path": str(bib_file)})
        assert response.status_code == 200
        assert "imported 1 entries" in response.text

    def test_import_bib_missing_path(self, client: TestClient) -> None:
        response = client.post("/bib/import", data={"path": "/does/not/exist"})
        assert response.status_code == 200
        assert "no such file: /does/not/exist" in response.text

    def test_import_notes_upload(self, client: TestClient) -> None:
        files = {"files": ("notes.txt", b"a note\n")}
        response = client.post("/import", files=files)
        assert response.status_code == 200
        assert "imported 1 notes from 1 files" in response.text

    def test_import_notes_dry_run(self, client: TestClient) -> None:
        files = {"files": ("notes.txt", b"a note\n")}
        response = client.post("/import", data={"dry_run": "true"}, files=files)
        assert response.status_code == 200
        assert "would import 1 notes from 1 files" in response.text

    def test_import_notes_empty(self, client: TestClient) -> None:
        response = client.post("/import", data={})
        assert response.status_code == 200
        assert "Please provide a path or upload files" in response.text

    def test_import_notes_invalid_mode(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        files = {"files": ("notes.txt", b"a note\n")}
        response = client.post("/import", data={"mode": "invalid"}, files=files)
        assert response.status_code == 200
        # The CLI rejects an unknown --mode outright; the web form must too,
        # not silently fall back to line mode.
        assert 'class="error"' in response.text
        assert "imported 1 notes" not in response.text
        with Store.open(tmp_path / "hashline.db") as store:
            assert store.count_notes() == 0

    def test_a_rejected_mode_still_reports_the_files_it_skipped(
        self, client: TestClient
    ) -> None:
        """A refusal must not swallow the per-file report already collected.

        The early return for a bad mode carries only ``error``, so the
        "skipped" list gathered from the other files in the same submission
        never reaches the page: the user is told the mode was wrong and
        silently loses the news that one of their files was unreadable.
        """
        files = [
            ("files", ("good.txt", b"a note\n")),
            ("files", ("bad.txt", b"\xff\xfe not utf-8")),
        ]
        response = client.post("/import", data={"mode": "invalid"}, files=files)
        assert response.status_code == 200
        assert 'class="error"' in response.text
        assert "bad.txt" in response.text, (
            "the mode error hid the file that could not be read"
        )

    def test_import_bib_upload(self, client: TestClient) -> None:
        files = {"file": ("library.bib", b"@article{smith2020, title={A title}}")}
        response = client.post("/bib/import", files=files)
        assert response.status_code == 200
        assert "imported 1 entries" in response.text

    def test_import_bib_upload_replace(self, client: TestClient) -> None:
        files = {"file": ("library.bib", b"@article{smith2020, title={A title}}")}
        response = client.post("/bib/import", data={"replace": "true"}, files=files)
        assert response.status_code == 200
        assert "imported 1 entries" in response.text

    def test_import_bib_empty(self, client: TestClient) -> None:
        response = client.post("/bib/import", data={})
        assert response.status_code == 200
        assert "Please provide a path or upload a .bib file" in response.text

    def test_import_bib_parse_error(self, client: TestClient) -> None:
        files = {"file": ("library.bib", b"invalid bibtex")}
        response = client.post("/bib/import", files=files)
        assert response.status_code == 200
        assert "Parsed to nothing" in response.text

    def test_import_bib_decode_error(self, client: TestClient) -> None:
        files = {"file": ("library.bib", b"\xff\xfe")}
        response = client.post("/bib/import", files=files)
        assert response.status_code == 200
        assert "could not decode" in response.text

    def test_import_bib_path_with_non_utf8_bytes_does_not_500(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        # bib_import's path branch only catches OSError, but
        # Path.read_text(encoding="utf-8") raises UnicodeDecodeError, a
        # ValueError subclass, on a non-UTF-8 file -- so it was escaping
        # uncaught. Use raise_server_exceptions=False so a regression shows
        # up as the real 500 a browser would see, not a raised exception.
        bib_file = tmp_path / "library.bib"
        bib_file.write_bytes(
            "@article{muller2020, author={Müller, Hans}, "
            "title={Titel}}".encode("latin-1")
        )
        safe_client = TestClient(app, raise_server_exceptions=False)
        response = safe_client.post("/bib/import", data={"path": str(bib_file)})
        assert response.status_code == 200
        assert 'class="error"' in response.text
        with Store.open(tmp_path / "hashline.db") as store:
            assert store.list_bib_entries() == []

    def test_import_bib_upload_with_non_utf8_bytes_already_answers_200(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        # The upload branch already catches UnicodeDecodeError -- this is a
        # regression guard, not a new bug, and it already passes.
        content = (
            "@article{muller2020, author={Müller, Hans}, "
            "title={Titel}}".encode("latin-1")
        )
        response = client.post(
            "/bib/import", files={"file": ("library.bib", content)}
        )
        assert response.status_code == 200
        assert "could not decode" in response.text
        with Store.open(tmp_path / "hashline.db") as store:
            assert store.list_bib_entries() == []

    def test_bib_import_where_every_entry_fails_to_parse_keeps_the_library(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        library = Path("tests/fixtures/bib/library.bib").read_bytes()
        client.post("/bib/import", files={"file": ("library.bib", library)})
        with Store.open(tmp_path / "hashline.db") as store:
            before = {e.citekey for e in store.list_bib_entries()}
        assert before, "the seed import itself must have worked"

        response = client.post(
            "/bib/import",
            data={"replace": "true"},
            files={
                "file": ("broken.bib", b"@article{broken2021, title={Unclosed")
            },
        )
        assert response.status_code == 200
        # entries == [] but problems != [] must not be treated as "nothing
        # parsed" -- that branch would run upsert_bib_entries([], replace=True)
        # and wipe the library.
        with Store.open(tmp_path / "hashline.db") as store:
            after = {e.citekey for e in store.list_bib_entries()}
        assert after == before
        assert "broken2021" in response.text

    def test_bib_import_of_a_truly_empty_file_keeps_the_library(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        library = Path("tests/fixtures/bib/library.bib").read_bytes()
        client.post("/bib/import", files={"file": ("library.bib", library)})
        with Store.open(tmp_path / "hashline.db") as store:
            before = {e.citekey for e in store.list_bib_entries()}
        assert before

        response = client.post(
            "/bib/import",
            data={"replace": "true"},
            files={"file": ("empty.bib", b"")},
        )
        assert response.status_code == 200
        assert "Parsed to nothing" in response.text
        with Store.open(tmp_path / "hashline.db") as store:
            after = {e.citekey for e in store.list_bib_entries()}
        assert after == before

    def test_bib_import_replace_with_entries_that_parse_still_replaces(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        library = Path("tests/fixtures/bib/library.bib").read_bytes()
        client.post("/bib/import", files={"file": ("library.bib", library)})
        with Store.open(tmp_path / "hashline.db") as store:
            assert store.list_bib_entries()

        response = client.post(
            "/bib/import",
            data={"replace": "true"},
            files={"file": ("only.bib", b"@article{only2023, title={Only}}")},
        )
        assert response.status_code == 200
        with Store.open(tmp_path / "hashline.db") as store:
            keys = {e.citekey for e in store.list_bib_entries()}
        assert keys == {"only2023"}

    def test_bib_import_with_a_path_and_an_upload_uses_both(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        response = client.post(
            "/bib/import",
            data={"path": "tests/fixtures/bib/library.bib"},
            files={
                "file": ("other.bib", b"@article{other2023, title={Other Title}}")
            },
        )
        assert response.status_code == 200
        with Store.open(tmp_path / "hashline.db") as store:
            keys = {e.citekey for e in store.list_bib_entries()}
        assert "smith2020" in keys  # from the path
        assert "other2023" in keys  # from the upload


class TestExport:
    def test_export_preview_empty(self, client: TestClient) -> None:
        response = client.get("/export")
        assert response.status_code == 200
        assert "No notes match these filters" in response.text

    def test_export_preview_with_notes(self, seeded: TestClient) -> None:
        response = seeded.get("/export", params={"tag": "sqlite"})
        assert response.status_code == 200
        assert "bm25" in response.text
        assert "無関係" not in response.text
        assert "Download .md" in response.text

    def test_export_preview_root_conflict(self, client: TestClient) -> None:
        response = client.get("/export", params={"root": 1, "tag": "test"})
        assert response.status_code == 200
        assert 'class="error"' in response.text
        # The reader is looking at a form, not a terminal.
        assert "--root" not in response.text

    def test_export_form_no_longer_offers_a_free_text_root_box(
        self, client: TestClient
    ) -> None:
        # Typing a note id by hand is what this replaces -- the thread view
        # is where a user already knows which thread they want.
        body = client.get("/export").text
        assert '<input type="number" name="root"' not in body
        assert "Thread Root ID" not in body

    def test_export_with_a_root_renders_it_as_a_read_only_chip(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            parent = store.add_note("parent")
            store.add_note("child", parent_id=parent.id)
        response = client.get("/export", params={"root": parent.id})
        assert response.status_code == 200
        assert f"thread #{parent.id}" in response.text
        # Still carried into the download form.
        assert f'name="root" value="{parent.id}"' in response.text

    def test_export_chip_dismiss_link_drops_root(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            parent = store.add_note("parent")
        body = client.get("/export", params={"root": parent.id}).text
        link = re.search(r'<a href="([^"]*)"[^>]*>&#10005;</a>', body)
        assert link is not None, "no dismiss link found next to the chip"
        assert "root=" not in link.group(1), (
            "the chip's dismiss link still carries the root filter"
        )

    def test_export_tag_is_a_select_populated_from_the_store(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        # A free-text box lets a typo silently export nothing; a <select>
        # can only offer tags that actually exist.
        with Store.open(tmp_path / "hashline.db") as store:
            store.add_note("a #sqlite note")
            store.add_note("another #sqlite note")
            store.add_note("a #web note")
        body = client.get("/export").text
        select = re.search(r'<select name="tag"[^>]*>.*?</select>', body, re.S)
        assert select is not None, "tag is not rendered as a <select>"
        markup = select.group(0)
        assert re.search(r'<option value=""[^>]*>', markup), (
            "no empty-value 'all' option found in the tag select"
        )
        assert "sqlite (2)" in markup
        assert "web (1)" in markup

    def test_export_citekey_select_shows_titles(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.upsert_bib_entries(
                [
                    BibEntry(
                        citekey="smith2020",
                        entry_type="article",
                        title="A Title",
                        tag="smith2020",
                    )
                ]
            )
        body = client.get("/export").text
        select = re.search(r'<select name="citekey"[^>]*>.*?</select>', body, re.S)
        assert select is not None, "citekey is not rendered as a <select>"
        markup = select.group(0)
        assert '<option value="smith2020"' in markup
        assert "A Title" in markup

    def test_export_select_marks_the_current_filter_selected(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.upsert_bib_entries(
                [
                    BibEntry(
                        citekey="smith2020",
                        entry_type="article",
                        title="A Title",
                        tag="smith2020",
                    )
                ]
            )
            store.add_note("a #sqlite note")
        body = client.get(
            "/export", params={"tag": "sqlite", "citekey": "smith2020"}
        ).text
        assert re.search(
            r'<option value="sqlite"[^>]*selected[^>]*>', body
        ), "the tag currently filtered on is not marked selected"
        assert re.search(
            r'<option value="smith2020"[^>]*selected[^>]*>', body
        ), "the citekey currently filtered on is not marked selected"

    def test_export_preview_trigger_covers_the_selects(
        self, client: TestClient
    ) -> None:
        # <select> is not an <input>, so "input from:input, change from:input"
        # alone stops matching the moment Tag/Citekey become dropdowns --
        # changing either would silently stop refreshing the preview.
        body = client.get("/export").text
        form = re.search(r'<form hx-get="/export"[^>]*>', body)
        assert form is not None, "export preview form not found"
        trigger = re.search(r'hx-trigger="([^"]*)"', form.group(0))
        assert trigger is not None, "export preview form has no hx-trigger"
        assert "select" in trigger.group(1), (
            f"hx-trigger {trigger.group(1)!r} does not mention select, so "
            f"choosing a tag or citekey from the dropdown will not refresh "
            f"the preview"
        )

    def test_download_form_hidden_fields_carry_the_selected_tag_and_citekey(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.upsert_bib_entries(
                [
                    BibEntry(
                        citekey="smith2020",
                        entry_type="article",
                        title="A Title",
                        tag="smith2020",
                    )
                ]
            )
            store.add_note("a #sqlite note")
        body = client.get(
            "/export", params={"tag": "sqlite", "citekey": "smith2020"}
        ).text
        download_form = re.search(
            r'<form method="get" action="/export/download".*?</form>', body, re.S
        )
        assert download_form is not None, "download form not found"
        markup = download_form.group(0)
        assert 'name="tag" value="sqlite"' in markup
        assert 'name="citekey" value="smith2020"' in markup

    def test_export_download_success(self, seeded: TestClient) -> None:
        response = seeded.get("/export/download", params={"tag": "sqlite"})
        assert response.status_code == 200
        assert response.headers["Content-Disposition"] == (
            'attachment; filename="export_sqlite.md"'
        )
        assert "bm25" in response.text

    def test_export_download_survives_a_non_ascii_tag(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        # sanitize() is re.sub(r"[^\w\-]", "_", s); Python's \w is
        # Unicode-aware, so Japanese passes straight through into
        # Content-Disposition, which Starlette encodes as latin-1.
        # raise_server_exceptions=False surfaces the real 500 a browser
        # would see instead of raising inside the test.
        with Store.open(tmp_path / "hashline.db") as store:
            store.add_note("メモ #日本語")
        safe_client = TestClient(app, raise_server_exceptions=False)
        response = safe_client.get("/export/download", params={"tag": "日本語"})
        assert response.status_code == 200
        disposition = response.headers["Content-Disposition"]
        disposition.encode("latin-1")  # must not raise UnicodeEncodeError

    def test_export_download_survives_an_accented_citekey(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.upsert_bib_entries(
                [
                    BibEntry(
                        citekey="müller2020",
                        entry_type="article",
                        title="Titel",
                        tag="mueller2020",
                    )
                ]
            )
        safe_client = TestClient(app, raise_server_exceptions=False)
        response = safe_client.get(
            "/export/download", params={"citekey": "müller2020"}
        )
        assert response.status_code == 200
        disposition = response.headers["Content-Disposition"]
        disposition.encode("latin-1")  # must not raise UnicodeEncodeError

    def test_export_download_ascii_tag_still_names_the_file(
        self, client: TestClient
    ) -> None:
        # The non-ASCII fix must not collapse every filename down to a bare
        # "export_.md" -- an ASCII tag should still be readable in it.
        response = client.get("/export/download", params={"tag": "sqlite"})
        assert response.status_code == 200
        disposition = response.headers["Content-Disposition"]
        assert "sqlite" in disposition

    def test_export_download_root_conflict(self, client: TestClient) -> None:
        response = client.get("/export/download", params={"root": 1, "tag": "test"})
        # A rejected form never answers 4xx: it comes back as the export page
        # with a readable error, not a file the browser tries to save.
        assert response.status_code == 200
        assert "Content-Disposition" not in response.headers
        assert 'class="error"' in response.text
        assert "--root" not in response.text

    def test_export_download_thread(self, seeded: TestClient, tmp_path: Path) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            parent = store.add_note("parent")
            store.add_note("child", parent_id=parent.id)
        response = seeded.get("/export/download", params={"root": parent.id})
        assert response.status_code == 200
        assert response.headers["Content-Disposition"] == (
            f'attachment; filename="thread_{parent.id}.md"'
        )
        assert "parent" in response.text
        assert "child" in response.text

    def test_export_download_citekey_and_tag(self, client: TestClient) -> None:
        response = client.get("/export/download", params={"tag": "t", "citekey": "c"})
        assert response.headers["Content-Disposition"] == (
            'attachment; filename="export_t_c.md"'
        )

    def test_export_preview_invalid_root(self, client: TestClient) -> None:
        response = client.get("/export", params={"root": 999})
        assert response.status_code == 200

    def test_export_download_of_a_nonexistent_root_answers_200_not_4xx(
        self, client: TestClient
    ) -> None:
        # A rejected form never answers 4xx in this app (see /export and every
        # other handler): a 4xx here renders raw JSON in the browser instead of
        # the readable error /export shows for the same condition. This replaces
        # an older test that asserted the 400.
        response = client.get("/export/download", params={"root": 999})
        assert response.status_code == 200
        assert "Content-Disposition" not in response.headers
        assert 'class="error"' in response.text

    def test_export_preview_accepts_a_browser_blank_root(
        self, client: TestClient
    ) -> None:
        # A browser submits every field, including the ones left empty.
        response = client.get(
            "/export", params={"tag": "", "citekey": "", "root": ""}
        )
        assert response.status_code == 200

    def test_export_preview_accepts_a_blank_root_alongside_a_tag(
        self, seeded: TestClient
    ) -> None:
        response = seeded.get(
            "/export", params={"tag": "sqlite", "citekey": "", "root": ""}
        )
        assert response.status_code == 200
        assert "bm25" in response.text

    def test_export_download_accepts_a_blank_root_alongside_a_tag(
        self, seeded: TestClient
    ) -> None:
        response = seeded.get(
            "/export/download", params={"tag": "sqlite", "citekey": "", "root": ""}
        )
        assert response.status_code == 200
        assert response.headers["Content-Disposition"].startswith("attachment")

    def test_export_download_accepts_blank_tag_and_citekey_alongside_a_root(
        self, seeded: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            parent = store.add_note("parent")
            store.add_note("child", parent_id=parent.id)
        response = seeded.get(
            "/export/download",
            params={"tag": "", "citekey": "", "root": str(parent.id)},
        )
        assert response.status_code == 200

    def test_download_filename_does_not_leak_a_raw_quote(
        self, client: TestClient
    ) -> None:
        response = client.get("/export/download", params={"tag": 'a"b'})
        assert response.status_code == 200
        disposition = response.headers["Content-Disposition"]
        # Exactly the two quotes that delimit the filename -- none from the tag.
        assert disposition.count('"') == 2

    def test_download_filename_strips_crlf(self, client: TestClient) -> None:
        response = client.get("/export/download", params={"tag": "a\r\nb"})
        assert response.status_code == 200
        disposition = response.headers["Content-Disposition"]
        assert "\r" not in disposition
        assert "\n" not in disposition

    def test_delete_note_not_found(self, client: TestClient) -> None:
        response = client.post("/notes/999/delete")
        assert response.status_code == 200
        assert "note 999 not found" in response.text

    def test_get_import(self, client: TestClient) -> None:
        response = client.get("/import")
        assert response.status_code == 200
        assert "Import Notes" in response.text

    def test_import_notes_read_error(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        notes_file = tmp_path / "notes.txt"
        notes_file.write_text("a note\n")
        from hashline.web import app

        def mock_read(*args, **kwargs):
            raise FileNotFoundError("permission denied")

        monkeypatch.setattr(app, "read_documents", mock_read)
        response = client.post("/import", data={"path": str(notes_file)})
        assert response.status_code == 200
        assert "permission denied" in response.text

    def test_import_notes_parse_error(self, client: TestClient) -> None:
        files = {"files": ("notes.txt", b"- not a valid outline\n")}
        response = client.post("/import", data={"mode": "outline"}, files=files)
        assert response.status_code == 200

    def test_import_bib_read_error(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bib_file = tmp_path / "library.bib"
        bib_file.write_text("")

        original_read_text = Path.read_text

        def mock_read_text(self: Path, *args, **kwargs) -> str:
            if self.name == "library.bib":
                raise OSError("I/O error")
            return original_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", mock_read_text)
        response = client.post("/bib/import", data={"path": str(bib_file)})
        assert response.status_code == 200
        assert "could not read" in response.text

    def test_import_bib_kept(self, seeded: TestClient, tmp_path: Path) -> None:
        from hashline.bib import BibEntry

        with Store.open(tmp_path / "hashline.db") as store:
            store.upsert_bib_entries(
                [BibEntry("smith2020", "article", "@article{smith2020}")]
            )
            store.add_note("A note citing [@smith2020]", citekey="smith2020")

        bib_file = tmp_path / "library.bib"
        bib_file.write_text("@article{other, title={New title}}")
        response = seeded.post(
            "/bib/import", data={"path": str(bib_file), "replace": "true"}
        )
        assert response.status_code == 200
        assert "kept 1 entries still cited" in response.text


class TestStats:
    """The web's read-only view onto ``hashline.analytics``."""

    @pytest.mark.parametrize(
        "view", ["overview", "activity", "tags", "reading", "threads"]
    )
    def test_renders_for_every_view(self, seeded: TestClient, view: str) -> None:
        response = seeded.get("/stats", params={"view": view})
        assert response.status_code == 200
        assert "hashline" in response.text

    def test_nav_entry_exists_and_highlights(self, client: TestClient) -> None:
        response = client.get("/stats")
        assert response.status_code == 200
        assert 'href="/stats" class="current"' in response.text
        # And it must be offered from every other page too.
        assert 'href="/stats"' in client.get("/").text

    def test_overview_is_the_default_view_and_shows_the_totals(
        self, seeded: TestClient
    ) -> None:
        body = seeded.get("/stats").text
        assert "note_count" in body
        assert "tag_count" in body
        assert "work_count" in body

    def test_empty_database_renders_headers_and_no_rows(
        self, client: TestClient
    ) -> None:
        response = client.get("/stats", params={"view": "activity"})
        assert response.status_code == 200
        assert "<th" in response.text
        assert "<td" not in response.text

    def test_bad_freq_answers_200_with_a_visible_error(
        self, seeded: TestClient
    ) -> None:
        response = seeded.get("/stats", params={"view": "activity", "freq": "bogus"})
        assert response.status_code == 200
        assert 'class="error"' in response.text

    def test_bad_freq_on_the_tags_view_also_answers_200_with_an_error(
        self, seeded: TestClient
    ) -> None:
        response = seeded.get("/stats", params={"view": "tags", "freq": "bogus"})
        assert response.status_code == 200
        assert 'class="error"' in response.text

    def test_tags_view_lists_the_top_tags_as_columns(self, seeded: TestClient) -> None:
        body = seeded.get("/stats", params={"view": "tags"}).text
        assert "sqlite" in body
        assert "other" in body

    def test_reading_view_shows_a_cited_work(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.upsert_bib_entries(
                [
                    BibEntry(
                        citekey="smith2020",
                        entry_type="article",
                        title="A title",
                        tag="smith2020",
                    )
                ]
            )
            store.add_note("about smith", citekey="smith2020")
        body = client.get("/stats", params={"view": "reading"}).text
        assert "smith2020" in body
        assert "A title" in body

    def test_threads_view_shows_reply_counts(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            root = store.add_note("root")
            store.add_note("reply", parent_id=root.id)
        body = client.get("/stats", params={"view": "threads"}).text
        assert str(root.id) in body

    def test_escapes_note_derived_text(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.upsert_bib_entries(
                [
                    BibEntry(
                        citekey="xss2020",
                        entry_type="article",
                        title="<script>alert(1)</script>",
                        tag="xss2020",
                    )
                ]
            )
            store.add_note("a note", citekey="xss2020")
        body = client.get("/stats", params={"view": "reading"}).text
        assert "<script>alert(1)</script>" not in body


class TestCsrf:
    """State-changing routes must reject a cross-origin form post.

    A plain form POST needs no CORS preflight, so any page open in the
    browser can submit one to ``127.0.0.1:8000`` -- but browsers always
    attach an ``Origin`` header to a cross-origin POST. Use
    ``raise_server_exceptions=False`` so a still-unprotected route shows up
    as whatever status it actually answers, not a raised exception.
    """

    def test_delete_rejects_a_cross_origin_post(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.add_note("to keep")
        safe_client = TestClient(app, raise_server_exceptions=False)
        response = safe_client.post(
            "/notes/1/delete", headers={"Origin": "https://evil.example"}
        )
        assert response.status_code in (400, 403), (
            f"cross-origin delete was not rejected, got {response.status_code}"
        )
        with Store.open(tmp_path / "hashline.db") as store:
            assert store.count_notes() == 1, (
                "the note was deleted by a cross-origin request"
            )

    def test_bib_import_rejects_a_cross_origin_post(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.upsert_bib_entries(
                [
                    BibEntry(
                        citekey="smith2020",
                        entry_type="article",
                        title="Test",
                        tag="smith2020",
                    )
                ]
            )
        safe_client = TestClient(app, raise_server_exceptions=False)
        files = {"file": ("other.bib", b"@article{other2023, title={Other}}")}
        response = safe_client.post(
            "/bib/import",
            data={"replace": "true"},
            files=files,
            headers={"Origin": "https://evil.example"},
        )
        assert response.status_code in (400, 403), (
            f"cross-origin bib import was not rejected, got "
            f"{response.status_code}"
        )
        with Store.open(tmp_path / "hashline.db") as store:
            keys = {e.citekey for e in store.list_bib_entries()}
        assert keys == {"smith2020"}, (
            "the library was replaced by a cross-origin request"
        )

    def test_delete_without_an_origin_header_still_works(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        # Every existing test posts with no Origin header at all; the fix
        # must not break them.
        with Store.open(tmp_path / "hashline.db") as store:
            store.add_note("to delete")
        response = client.post("/notes/1/delete")
        assert response.status_code == 200
        with Store.open(tmp_path / "hashline.db") as store:
            assert store.count_notes() == 0

    def test_delete_with_a_same_origin_header_still_works(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        # TestClient's default base_url is http://testserver.
        with Store.open(tmp_path / "hashline.db") as store:
            store.add_note("to delete")
        response = client.post(
            "/notes/1/delete", headers={"Origin": "http://testserver"}
        )
        assert response.status_code == 200
        with Store.open(tmp_path / "hashline.db") as store:
            assert store.count_notes() == 0

    def test_a_proxied_https_origin_is_accepted(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """Behind a TLS-terminating proxy the scheme cannot be compared.

        ngrok, a Tailscale funnel or any reverse proxy leaves the ASGI
        scope on ``http`` while the browser sends ``Origin: https://...``.
        Comparing the scheme rejects every write and the whole UI stops
        working; comparing the host is the check that actually stops a
        cross-origin post, because the browser -- not the attacker -- picks
        the host it connects to.
        """
        with Store.open(tmp_path / "hashline.db") as store:
            store.add_note("to delete")
        response = client.post(
            "/notes/1/delete", headers={"Origin": "https://testserver"}
        )
        assert response.status_code == 200, (
            "a same-host request through a TLS-terminating proxy was rejected"
        )
        with Store.open(tmp_path / "hashline.db") as store:
            assert store.count_notes() == 0


class TestFormContracts:
    """Every rendered form must post the field names its route requires.

    ``TestHtmxTargets`` proves a control's request leaves the browser; this
    proves the server accepts it. Renaming a Form parameter without
    renaming it in every template that posts to that route answers 422, and
    htmx does not swap a non-2xx response -- so the button is as dead as a
    missing hx-target, with nothing on screen to say so. Asserting the
    markup contains ``hx-post="/context/read"`` cannot catch that; only
    submitting the fields the page actually renders can.
    """

    #: Fields a browser leaves out of a submission, so neither can we: an
    #: unticked checkbox is not sent at all, and an empty file input is not
    #: a string the route could parse.
    _OMITTED_TYPES: Final = {"checkbox", "file"}

    @staticmethod
    def _seed(tmp_path: Path) -> None:
        with Store.open(tmp_path / "hashline.db") as store:
            store.upsert_bib_entries(
                [
                    BibEntry(
                        citekey="smith2020",
                        entry_type="article",
                        title="Test",
                        tag="smith2020",
                    )
                ]
            )
            store.add_note("a note #tag")

    @staticmethod
    def _forms(html: str) -> list[tuple[str, str, dict[str, str]]]:
        """Every form on the page as ``(method, action, fields)``.

        Covers the htmx forms and the two plain ones alike -- ``/import``
        submits multipart the normal way, and ``/export`` previews with
        ``hx-get`` -- because the mismatch this guards against is about
        field names, not about how the request is sent.
        """
        forms = []
        for block in re.findall(r"<form\b.*?</form>", html, re.DOTALL):
            submission = TestFormContracts._submission(block)
            if submission is None:
                continue
            fields: dict[str, str] = {}
            for control in re.findall(r"<(?:input|textarea|select)\b[^>]*>", block):
                name = re.search(r'name="([^"]+)"', control)
                if name is None:
                    continue
                kind = re.search(r'type="([^"]+)"', control)
                omitted = TestFormContracts._OMITTED_TYPES
                if kind is not None and kind.group(1) in omitted:
                    continue
                value = re.search(r'value="([^"]*)"', control)
                fields[name.group(1)] = value.group(1) if value else "x"
            forms.append((*submission, fields))
        return forms

    @staticmethod
    def _submission(block: str) -> tuple[str, str] | None:
        """The ``(method, action)`` a form sends, htmx attributes first."""
        for attribute, method in (("hx-post", "POST"), ("hx-get", "GET")):
            target = re.search(rf'{attribute}="([^"]+)"', block)
            if target is not None:
                return method, target.group(1)
        target = re.search(r'action="([^"]+)"', block)
        if target is None:
            return None
        declared = re.search(r'method="([^"]+)"', block)
        return (declared.group(1).upper() if declared else "GET"), target.group(1)

    @pytest.mark.parametrize(
        "route",
        ["/", "/bib", "/bib/smith2020", "/import", "/export", "/stats"],
    )
    def test_every_form_submits_the_fields_its_route_requires(
        self, client: TestClient, tmp_path: Path, route: str
    ) -> None:
        self._seed(tmp_path)
        page = client.get(route)
        assert page.status_code == 200
        forms = self._forms(page.text)
        assert forms, f"no form found on {route!r}"
        for method, action, fields in forms:
            if method == "GET":
                response = client.get(action, params=fields)
            else:
                response = client.post(action, data=fields)
            assert response.status_code != 422, (
                f"the form on {route!r} submitting to {action!r} sends "
                f"{sorted(fields)}, which that route rejects as invalid: "
                f"{response.text[:200]}"
            )


class TestSemanticSearch:
    """The web's half of semantic search: the toggle and what it says.

    The ranking itself is tested in tests/test_ml_search.py and the CLI's
    wiring in tests/test_cli.py; these are the adapter's own concerns.
    """

    @staticmethod
    def _embed(tmp_path: Path, bodies: dict[int, list[float]]) -> None:
        """Write vectors straight into the store, no model involved."""
        from hashline.ml.embed import embedding_key, pack_vector

        with Store.open(tmp_path / "hashline.db") as store:
            for note_id, vector in bodies.items():
                store.upsert_embedding(
                    note_id,
                    model=embedding_key(),
                    vector=pack_vector(np.array(vector, dtype=np.float32)),
                    dim=len(vector),
                )

    def test_the_search_form_offers_the_toggle(self, client: TestClient) -> None:
        assert 'name="semantic"' in client.get("/").text

    def test_a_missing_extra_says_so_instead_of_no_matches(
        self, seeded: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty result alone would blame the notes for a missing library."""
        monkeypatch.setattr("hashline.ml.embed.is_available", lambda: False)
        body = seeded.get("/notes", params={"q": "bm25", "semantic": "true"}).text
        assert "uv sync --extra ml" in body
        assert 'class="notice"' in body

    def test_an_empty_index_says_indexing_is_running(
        self, seeded: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("hashline.ml.embed.is_available", lambda: True)
        body = seeded.get("/notes", params={"q": "bm25", "semantic": "true"}).text
        assert "indexing" in body.lower()
        assert 'class="notice"' in body

    def test_it_ranks_by_the_stored_vectors(
        self, seeded: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A note the keyword index cannot reach still comes back.

        "無関係なメモ" contains no part of the query, so plain search never
        returns it. Its vector is the query's nearest, so the semantic ranking
        does -- which is the whole reason the toggle exists. The keyword hit
        still leads, because it is the one note both rankers agree on.
        """
        monkeypatch.setattr("hashline.ml.embed.is_available", lambda: True)
        monkeypatch.setattr(
            "hashline.ml.embed.load_model", lambda name=None: _FakeEmbedder([0.0, 1.0])
        )
        self._embed(tmp_path, {1: [1.0, 0.0], 2: [0.0, 1.0]})
        plain = seeded.get("/notes", params={"q": "bm25"}).text
        assert "無関係なメモ" not in plain

        body = seeded.get("/notes", params={"q": "bm25", "semantic": "true"}).text
        assert "無関係なメモ" in body
        assert body.index("bm25 を調べた") < body.index("無関係なメモ")

    def test_a_half_indexed_library_says_how_much_is_left(
        self, seeded: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A short answer with no explanation reads as "there is nothing else"."""
        monkeypatch.setattr("hashline.ml.embed.is_available", lambda: True)
        monkeypatch.setattr(
            "hashline.ml.embed.load_model", lambda name=None: _FakeEmbedder([1.0, 0.0])
        )
        self._embed(tmp_path, {1: [1.0, 0.0]})
        body = seeded.get("/notes", params={"q": "bm25", "semantic": "true"}).text
        assert "1 notes are still being indexed" in body

    def test_the_toggle_survives_capturing_a_note(
        self, seeded: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every control that redraws the timeline has to carry the mode.

        Otherwise adding a note while a semantic search is on silently drops
        back to keyword results, with the checkbox still ticked.
        """
        monkeypatch.setattr("hashline.ml.embed.is_available", lambda: True)
        monkeypatch.setattr(
            "hashline.ml.embed.load_model", lambda name=None: _FakeEmbedder([1.0, 0.0])
        )
        self._embed(tmp_path, {1: [1.0, 0.0], 2: [0.0, 1.0]})
        response = seeded.post(
            "/notes", data={"body": "another", "q": "bm25", "semantic": "true"}
        )
        assert response.status_code == 200
        assert "1 notes are still being indexed" in response.text

    def test_a_plain_search_never_touches_the_backend(
        self, seeded: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(name: str = "") -> object:
            raise AssertionError("the keyword path loaded an embedding model")

        monkeypatch.setattr("hashline.ml.embed.load_model", explode)
        assert seeded.get("/notes", params={"q": "bm25"}).status_code == 200


class TestStartupIndexing:
    """Indexing runs at startup so the web never needs the CLI.

    Three gates stop it from touching a model; each is tested because the
    default test run has the extra installed on the author's machine and not
    in CI, and both have to stay quiet.
    """

    def test_no_index_env_skips_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from hashline.web import app as web_app

        monkeypatch.setenv("HASHLINE_NO_INDEX", "1")
        monkeypatch.setattr(
            "hashline.ml.hybrid.is_available", lambda: _fail("checked availability")
        )
        web_app._index_in_background()  # must return before the gate below

    def test_an_empty_database_needs_no_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from hashline.web import app as web_app

        monkeypatch.delenv("HASHLINE_NO_INDEX", raising=False)
        monkeypatch.setenv("HASHLINE_DB", str(tmp_path / "hashline.db"))
        monkeypatch.setattr("hashline.ml.hybrid.is_available", lambda: True)
        monkeypatch.setattr(
            "hashline.ml.hybrid.index_pending",
            lambda *a, **k: _fail("embedded an empty database"),
        )
        web_app._index_in_background()

    def test_a_failure_does_not_take_the_app_down(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Serving notes must not depend on an optional model working."""
        from hashline.web import app as web_app

        monkeypatch.delenv("HASHLINE_NO_INDEX", raising=False)
        monkeypatch.setenv("HASHLINE_DB", str(tmp_path / "hashline.db"))
        with Store.open(tmp_path / "hashline.db") as store:
            store.add_note("something to embed")
        monkeypatch.setattr("hashline.ml.hybrid.is_available", lambda: True)

        def boom(*args: object, **kwargs: object) -> int:
            raise RuntimeError("model exploded")

        monkeypatch.setattr("hashline.ml.hybrid.index_pending", boom)
        web_app._index_in_background()  # raising here would kill the server

    def test_it_embeds_what_is_pending(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from hashline.ml.embed import embedding_key
        from hashline.web import app as web_app

        monkeypatch.delenv("HASHLINE_NO_INDEX", raising=False)
        monkeypatch.setenv("HASHLINE_DB", str(tmp_path / "hashline.db"))
        with Store.open(tmp_path / "hashline.db") as store:
            store.add_note("something to embed")
        monkeypatch.setattr("hashline.ml.hybrid.is_available", lambda: True)
        monkeypatch.setattr(
            "hashline.ml.embed.load_model", lambda name=None: _FakeEmbedder([1.0, 0.0])
        )
        web_app._index_in_background()
        with Store.open(tmp_path / "hashline.db") as store:
            assert len(list(store.iter_embeddings(embedding_key()))) == 1
