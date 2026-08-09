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


class TestStatic:
    def test_htmx_is_served_locally(self, client: TestClient) -> None:
        response = client.get("/static/htmx.min.js")
        assert response.status_code == 200
        assert "htmx" in response.text[:200]
