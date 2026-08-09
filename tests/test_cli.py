"""Smoke tests for the CLI adapter: wiring, not note logic."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from hashline.cli import app, collect_documents

runner = CliRunner()


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "hashline.db"


def run(db: Path, *args: str) -> str:
    result = runner.invoke(app, ["--db", str(db), *args])
    assert result.exit_code == 0, result.output
    return result.output


class TestAdd:
    def test_stores_and_echoes_the_note(self, db: Path) -> None:
        output = run(db, "add", "bm25 を調べた #sqlite")
        assert "bm25 を調べた #sqlite" in output
        assert "[sqlite]" in output

    def test_extra_tags_are_repeatable(self, db: Path) -> None:
        output = run(db, "add", "plain body", "--tag", "one", "-t", "two")
        assert "[one, two]" in output
        assert "plain body" in output

    def test_blank_body_fails(self, db: Path) -> None:
        result = runner.invoke(app, ["--db", str(db), "add", "   "])
        assert result.exit_code != 0

    def test_unusable_tag_fails(self, db: Path) -> None:
        result = runner.invoke(app, ["--db", str(db), "add", "body", "-t", "two words"])
        assert result.exit_code != 0


class TestList:
    def test_newest_first(self, db: Path) -> None:
        run(db, "add", "older note")
        run(db, "add", "newer note")
        lines = run(db, "list").splitlines()
        assert "newer note" in lines[0]
        assert "older note" in lines[1]

    def test_filters_by_tag(self, db: Path) -> None:
        run(db, "add", "tagged #sqlite")
        run(db, "add", "untagged")
        output = run(db, "list", "--tag", "sqlite")
        assert "tagged" in output
        assert "untagged" not in output

    def test_honours_limit(self, db: Path) -> None:
        for index in range(3):
            run(db, "add", f"note {index}")
        assert len(run(db, "list", "-n", "2").splitlines()) == 2

    def test_reports_an_empty_timeline(self, db: Path) -> None:
        assert "no notes yet" in run(db, "list")

    def test_collapses_a_multiline_body(self, db: Path) -> None:
        run(db, "add", "line one\nline two")
        assert "line one line two" in run(db, "list")


class TestSearch:
    def test_prints_a_score_and_the_note(self, db: Path) -> None:
        run(db, "add", "bm25 を調べた #sqlite")
        run(db, "add", "無関係なメモ")
        output = run(db, "search", "bm25")
        assert "bm25 を調べた" in output
        assert "無関係" not in output

    def test_filters_by_tag(self, db: Path) -> None:
        run(db, "add", "bm25 here #sqlite")
        run(db, "add", "bm25 elsewhere #other")
        output = run(db, "search", "bm25", "--tag", "sqlite")
        assert "bm25 here" in output
        assert "bm25 elsewhere" not in output

    def test_reports_no_matches(self, db: Path) -> None:
        run(db, "add", "something")
        assert "no matches" in run(db, "search", "nothing like this")

    def test_fts5_operators_do_not_crash_it(self, db: Path) -> None:
        run(db, "add", 'a "quoted" thing')
        assert "quoted" in run(db, "search", '"quoted"')


class TestTags:
    def test_lists_tags_by_use(self, db: Path) -> None:
        run(db, "add", "one #sqlite")
        run(db, "add", "two #sqlite #fts5")
        lines = run(db, "tags").splitlines()
        assert lines[0].split() == ["2", "sqlite"]
        assert lines[1].split() == ["1", "fts5"]

    def test_reports_no_tags(self, db: Path) -> None:
        assert "no tags yet" in run(db, "tags")


class TestImport:
    def test_line_mode_imports_every_line(self, db: Path, notes_dir: Path) -> None:
        output = run(db, "import", str(notes_dir / "scratch.txt"))
        assert "imported 3 notes from 1 files" in output
        assert "一行メモをためす" in run(db, "list")

    def test_heading_mode_imports_sections(self, db: Path, notes_dir: Path) -> None:
        run(db, "import", str(notes_dir / "daily.md"), "--mode", "heading")
        assert len(run(db, "list").splitlines()) == 3

    def test_walks_a_directory_and_skips_other_suffixes(
        self, db: Path, notes_dir: Path
    ) -> None:
        output = run(db, "import", str(notes_dir))
        assert "from 4 files" in output
        assert "not a text or markdown file" not in run(db, "list", "-n", "50")

    def test_common_tag_lands_on_every_note(self, db: Path, notes_dir: Path) -> None:
        run(db, "import", str(notes_dir / "scratch.txt"), "--tag", "Imported")
        assert len(run(db, "list", "--tag", "imported").splitlines()) == 3

    def test_common_tag_does_not_enter_the_body(
        self, db: Path, notes_dir: Path
    ) -> None:
        run(db, "import", str(notes_dir / "scratch.txt"), "--tag", "imported")
        assert "#imported" not in run(db, "list")

    def test_dry_run_writes_nothing(self, db: Path, notes_dir: Path) -> None:
        output = run(db, "import", str(notes_dir / "scratch.txt"), "--dry-run")
        assert "would import 3 notes" in output
        assert "no notes yet" in run(db, "list")

    def test_missing_path_fails(self, db: Path, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["--db", str(db), "import", str(tmp_path / "nope")]
        )
        assert result.exit_code != 0

    def test_unusable_tag_fails(self, db: Path, notes_dir: Path) -> None:
        result = runner.invoke(
            app,
            ["--db", str(db), "import", str(notes_dir), "--tag", "two words"],
        )
        assert result.exit_code != 0


BIB_FIXTURE = Path(__file__).parent / "fixtures" / "bib" / "library.bib"


class TestBibImport:
    def test_reports_the_count(self, db: Path) -> None:
        output = run(db, "bib", "import", str(BIB_FIXTURE))
        assert "imported 7 entries" in output

    def test_the_malformed_entry_is_reported_and_the_rest_still_land(
        self, db: Path
    ) -> None:
        result = runner.invoke(
            app, ["--db", str(db), "bib", "import", str(BIB_FIXTURE)]
        )
        assert result.exit_code == 0, result.output
        assert "skipped" in result.output
        assert "unclosed braces" in result.output
        assert "imported 7 entries" in result.output

    def test_replace_does_not_accumulate_duplicates(self, db: Path) -> None:
        run(db, "bib", "import", str(BIB_FIXTURE))
        run(db, "bib", "import", str(BIB_FIXTURE), "--replace")
        assert len(run(db, "bib", "list").splitlines()) == 7

    def test_missing_file_fails(self, db: Path, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["--db", str(db), "bib", "import", str(tmp_path / "nope.bib")]
        )
        assert result.exit_code != 0


class TestBibList:
    def test_lists_entries_by_citekey(self, db: Path) -> None:
        run(db, "bib", "import", str(BIB_FIXTURE))
        lines = run(db, "bib", "list").splitlines()
        assert lines == sorted(lines)
        assert "smith2020" in "\n".join(lines)

    def test_reports_an_empty_library(self, db: Path) -> None:
        assert "no bibliography entries yet" in run(db, "bib", "list")


class TestBibShow:
    def test_shows_the_entry(self, db: Path) -> None:
        run(db, "bib", "import", str(BIB_FIXTURE))
        output = run(db, "bib", "show", "smith2020")
        assert "smith2020" in output
        assert "A Survey of Trigram Indexing" in output

    def test_unknown_citekey_fails(self, db: Path) -> None:
        result = runner.invoke(app, ["--db", str(db), "bib", "show", "nope"])
        assert result.exit_code != 0


class TestPin:
    def test_bare_pin_reports_no_context(self, db: Path) -> None:
        assert "no context pinned" in run(db, "pin")

    def test_pin_sets_tags_and_shows_them(self, db: Path) -> None:
        output = run(db, "pin", "research", "urgent")
        assert "research" in output
        assert "urgent" in output

    def test_pin_show_reports_the_pinned_tags(self, db: Path) -> None:
        run(db, "pin", "research")
        assert "research" in run(db, "pin", "--show")

    def test_pin_clear_empties_the_context(self, db: Path) -> None:
        run(db, "pin", "research")
        run(db, "pin", "--clear")
        assert "no context pinned" in run(db, "pin")

    def test_an_unusable_tag_fails(self, db: Path) -> None:
        result = runner.invoke(app, ["--db", str(db), "pin", "two words"])
        assert result.exit_code != 0


class TestCollectDocuments:
    def test_reads_an_explicit_file_whatever_its_suffix(
        self, notes_dir: Path
    ) -> None:
        documents, skipped = collect_documents([notes_dir / "ignored.json"])
        assert len(documents) == 1
        assert skipped == []

    def test_directory_walk_keeps_only_text_suffixes(self, notes_dir: Path) -> None:
        documents, _ = collect_documents([notes_dir])
        assert {Path(doc.source).name for doc in documents} == {
            "daily.md",
            "empty.md",
            "fenced.md",
            "scratch.txt",
        }

    def test_reports_a_file_it_cannot_decode(self, tmp_path: Path) -> None:
        broken = tmp_path / "broken.txt"
        broken.write_bytes(b"\xff\xfe not utf-8")
        documents, skipped = collect_documents([tmp_path])
        assert documents == []
        assert len(skipped) == 1
        assert "broken.txt" in skipped[0]
