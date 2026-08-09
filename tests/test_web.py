"""Smoke tests for the web adapter: routes and wiring, not note logic."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

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
        assert "requires a pinned citekey" in response.text

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
        assert "--no-context has none" in response.text

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

    @pytest.mark.parametrize(
        "route, expected_href",
        [
            ("/bib", 'href="/bib" class="current"'),
            ("/import", 'href="/import" class="current"'),
            ("/export", 'href="/export" class="current"'),
        ],
    )
    def test_stubs_render_and_mark_current(
        self, client: TestClient, route: str, expected_href: str
    ) -> None:
        response = client.get(route)
        assert response.status_code == 200
        assert expected_href in response.text
        assert "coming" in response.text


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
