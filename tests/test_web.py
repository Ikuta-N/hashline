"""Smoke tests for the web adapter: routes and wiring, not note logic."""

from collections.abc import Iterator
from pathlib import Path

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


class TestReply:
    def test_renders_reply_fragment(self, seeded: TestClient) -> None:
        response = seeded.get("/notes/1/reply")
        assert response.status_code == 200
        assert 'name="parent_id" value="1"' in response.text

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

    def test_unknown_id_is_404(self, client: TestClient) -> None:
        response = client.get("/notes/999/thread")
        assert response.status_code == 404


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
        assert 'name="tag"' in response.text
        assert 'name="citekey"' in response.text

    def test_pin_tags(self, client: TestClient) -> None:
        response = client.post("/context/pin", data={"tag": "research urgent"})
        assert response.status_code == 200
        assert "research, urgent" in response.text
        assert 'name="citekey"' not in response.text

        # Verify it persisted
        assert "research, urgent" in client.get("/context").text

    def test_pin_invalid_tag(self, client: TestClient) -> None:
        response = client.post("/context/pin", data={"tag": "invalid@tag"})
        assert response.status_code == 200
        assert 'class="error"' in response.text

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
            "/context/read", data={"citekey": "smith2020", "tag": ""}
        )
        assert response.status_code == 200
        assert "smith2020" in response.text
        assert "Test" in response.text
        assert "reading" in response.text

        # Verify it persisted
        assert "smith2020" in client.get("/context").text

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
            "/context/read", data={"citekey": "smith2020", "tag": "annotating"}
        )
        assert response.status_code == 200
        assert "annotating" in response.text

    def test_read_unknown_citekey(self, client: TestClient) -> None:
        response = client.post("/context/read", data={"citekey": "nope", "tag": ""})
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
            "/context/read", data={"citekey": "smith2020", "tag": "invalid@tag"}
        )
        assert response.status_code == 200
        assert 'class="error"' in response.text

    def test_clear_context(self, client: TestClient) -> None:
        client.post("/context/pin", data={"tag": "research"})
        response = client.post("/context/clear")
        assert response.status_code == 200
        assert "research" not in response.text
        assert 'name="citekey"' in response.text

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
        client.post("/context/read", data={"citekey": "smith2020", "tag": ""})
        # Without this form the reader has no way to change tags once reading.
        response = client.get("/context")
        assert response.status_code == 200
        assert 'hx-post="/context/pin"' in response.text

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
        client.post("/context/read", data={"citekey": "smith2020", "tag": "reading"})
        client.post("/context/pin", data={"tag": "extra"})
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
        client.post("/context/read", data={"citekey": "smith2020", "tag": "reading"})
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
        client.post("/context/read", data={"citekey": "smith2020", "tag": "reading"})
        response = client.post("/context/clear")
        assert response.status_code == 200
        with Store.open(tmp_path / "hashline.db") as store:
            context = store.get_context()
        assert context.citekey is None
        assert context.tags == ()


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
        assert "smith2020" in response.text
        assert "A title" in response.text
        assert "Smith" in response.text

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

    def test_export_download_success(self, seeded: TestClient) -> None:
        response = seeded.get("/export/download", params={"tag": "sqlite"})
        assert response.status_code == 200
        assert response.headers["Content-Disposition"] == (
            'attachment; filename="export_sqlite.md"'
        )
        assert "bm25" in response.text

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

    def test_export_download_invalid_root(self, client: TestClient) -> None:
        response = client.get("/export/download", params={"root": 999})
        assert response.status_code == 400

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
