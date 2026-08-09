"""Smoke tests for the CLI adapter: wiring, not note logic."""

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from hashline.cli import app
from hashline.ml import embed
from hashline.store import Store

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

    def test_page_with_nothing_pinned_fails(self, db: Path) -> None:
        result = runner.invoke(app, ["--db", str(db), "add", "body", "--page", "5"])
        assert result.exit_code != 0

    def test_no_context_bypasses_a_pinned_context(self, db: Path) -> None:
        run(db, "bib", "import", str(BIB_FIXTURE))
        run(db, "read", "start", "smith2020")
        output = run(db, "add", "bypass #own", "--no-context")
        assert "own" in output
        assert "reading" not in output
        assert "smith2020" not in output

    def test_no_context_with_page_fails(self, db: Path) -> None:
        result = runner.invoke(
            app, ["--db", str(db), "add", "body", "--page", "5", "--no-context"]
        )
        assert result.exit_code != 0

    def test_add_fails_cleanly_when_pinned_work_is_removed(
        self, db: Path, tmp_path: Path
    ) -> None:
        run(db, "bib", "import", str(BIB_FIXTURE))
        run(db, "read", "start", "smith2020")

        # Replace bibliography with an empty one
        empty_bib = tmp_path / "empty.bib"
        empty_bib.write_text("", encoding="utf-8")
        runner.invoke(
            app, ["--db", str(db), "bib", "import", str(empty_bib), "--replace"]
        )

        # Adding a note should now fail with a nice error
        result = runner.invoke(app, ["--db", str(db), "add", "a note"])
        assert result.exit_code != 0
        assert "no longer in the bibliography" in result.output
        assert "Traceback" not in result.output


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

    def test_roots_only_excludes_replies(self, db: Path) -> None:
        output = run(db, "add", "parent note")
        # Extract the note ID from the output string "    1  2026-08-09 ... parent note"
        parent_id = output.split()[0]
        run(db, "reply", parent_id, "child note")

        timeline = run(db, "list", "--roots-only")
        assert "parent note" in timeline
        assert "child note" not in timeline


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
        result = runner.invoke(app, ["--db", str(db), "import", str(tmp_path / "nope")])
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

    def test_non_utf8_file_fails_with_a_readable_message(
        self, db: Path, tmp_path: Path
    ) -> None:
        # Path.read_text(encoding="utf-8") raises UnicodeDecodeError, a
        # ValueError subclass, on a latin-1 .bib file -- ordinary for
        # BibTeX. It must come back as a BadParameter, not a traceback.
        bib_file = tmp_path / "library.bib"
        bib_file.write_bytes(
            "@article{muller2020, author={Müller, Hans}, "
            "title={Titel}}".encode("latin-1")
        )
        result = runner.invoke(
            app, ["--db", str(db), "bib", "import", str(bib_file)]
        )
        assert result.exit_code != 0
        assert "could not read" in result.output

    def test_replace_keeps_cited_entries(self, db: Path, tmp_path: Path) -> None:
        run(db, "bib", "import", str(BIB_FIXTURE))
        run(db, "read", "start", "smith2020")
        run(db, "add", "note that cites the work")

        # Replace with an empty bibliography
        empty_bib = tmp_path / "empty.bib"
        empty_bib.write_text("", encoding="utf-8")

        result = runner.invoke(
            app, ["--db", str(db), "bib", "import", str(empty_bib), "--replace"]
        )
        assert result.exit_code == 0
        assert "kept 1 entries still cited by notes" in result.output


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

    def test_show_conflicts_with_clear(self, db: Path) -> None:
        result = runner.invoke(app, ["--db", str(db), "pin", "--show", "--clear"])
        assert result.exit_code != 0
        assert "cannot combine" in result.output

    def test_show_conflicts_with_tags(self, db: Path) -> None:
        result = runner.invoke(app, ["--db", str(db), "pin", "tag", "--show"])
        assert result.exit_code != 0
        assert "cannot combine" in result.output

    def test_clear_conflicts_with_tags(self, db: Path) -> None:
        result = runner.invoke(app, ["--db", str(db), "pin", "tag", "--clear"])
        assert result.exit_code != 0
        assert "cannot combine" in result.output


class TestRead:
    def test_status_reports_nothing_pinned(self, db: Path) -> None:
        assert "nothing pinned" in run(db, "read", "status")

    def test_start_then_status_reports_the_work(self, db: Path) -> None:
        run(db, "bib", "import", str(BIB_FIXTURE))
        run(db, "read", "start", "smith2020")
        output = run(db, "read", "status")
        assert "smith2020" in output
        assert "A Survey of Trigram Indexing" in output
        assert "reading" in output

    def test_stop_clears_the_context(self, db: Path) -> None:
        run(db, "bib", "import", str(BIB_FIXTURE))
        run(db, "read", "start", "smith2020")
        run(db, "read", "stop")
        assert "nothing pinned" in run(db, "read", "status")

    def test_start_with_an_unknown_citekey_fails(self, db: Path) -> None:
        result = runner.invoke(app, ["--db", str(db), "read", "start", "nope"])
        assert result.exit_code != 0

    def test_start_with_an_unusable_tag_fails(self, db: Path) -> None:
        run(db, "bib", "import", str(BIB_FIXTURE))
        result = runner.invoke(
            app,
            ["--db", str(db), "read", "start", "smith2020", "--tag", "two words"],
        )
        assert result.exit_code != 0

    def test_custom_tag_replaces_the_default(self, db: Path) -> None:
        run(db, "bib", "import", str(BIB_FIXTURE))
        run(db, "read", "start", "smith2020", "--tag", "annotating")
        output = run(db, "read", "status")
        assert "annotating" in output
        assert "reading" not in output

    def test_a_note_added_while_reading_carries_both_tags_and_is_unchanged(
        self, db: Path
    ) -> None:
        run(db, "bib", "import", str(BIB_FIXTURE))
        run(db, "read", "start", "smith2020")
        output = run(db, "add", "序論の主張が弱い")
        assert "序論の主張が弱い" in output
        assert "reading" in output
        assert "smith2020" in output

    @pytest.mark.parametrize("page", ["12-15", "xii", "第3章"])
    def test_page_is_stored_verbatim(self, db: Path, page: str) -> None:
        run(db, "bib", "import", str(BIB_FIXTURE))
        run(db, "read", "start", "smith2020")
        output = run(db, "add", "a note", "--page", page)
        assert page in output


class TestReply:
    def test_reply_attaches_to_parent(self, db: Path) -> None:
        output = run(db, "add", "parent")
        parent_id = output.split()[0]

        reply_out = run(db, "reply", parent_id, "child")
        assert "child" in reply_out

        timeline = run(db, "list")
        assert "child" in timeline
        assert "parent" in timeline

    def test_reply_missing_parent_fails(self, db: Path) -> None:
        result = runner.invoke(app, ["--db", str(db), "reply", "999", "nope"])
        assert result.exit_code != 0
        assert "parent_id 999 does not exist" in result.output


class TestThread:
    def test_prints_thread_with_indentation(self, db: Path) -> None:
        p_out = run(db, "add", "parent")
        parent_id = p_out.split()[0]
        c1_out = run(db, "reply", parent_id, "child 1")
        child1_id = c1_out.split()[0]
        run(db, "reply", child1_id, "grandchild")
        run(db, "reply", parent_id, "child 2")

        thread = run(db, "thread", parent_id).splitlines()
        assert len(thread) == 4
        assert thread[0].startswith(" ")
        assert "parent" in thread[0]
        assert thread[1].startswith("    ")
        assert "child 1" in thread[1]
        assert thread[2].startswith("      ")
        assert "grandchild" in thread[2]
        assert thread[3].startswith("    ")
        assert "child 2" in thread[3]

    def test_missing_thread_root_fails(self, db: Path) -> None:
        result = runner.invoke(app, ["--db", str(db), "thread", "999"])
        assert result.exit_code != 0
        assert "does not exist" in result.output


class TestRm:
    def test_rm_deletes_a_note(self, db: Path) -> None:
        out = run(db, "add", "doomed")
        note_id = out.split()[0]
        run(db, "rm", note_id, "--yes")
        assert "no notes yet" in run(db, "list")

    def test_rm_missing_note_fails(self, db: Path) -> None:
        result = runner.invoke(app, ["--db", str(db), "rm", "999", "--yes"])
        assert result.exit_code != 0
        assert "note 999 not found" in result.output

    def test_rm_refuses_to_delete_parent_without_recursive(self, db: Path) -> None:
        p_out = run(db, "add", "parent")
        parent_id = p_out.split()[0]
        run(db, "reply", parent_id, "child")

        result = runner.invoke(app, ["--db", str(db), "rm", parent_id, "--yes"])
        assert result.exit_code != 0
        assert "has 1 replies" in result.output
        assert "use --recursive" in result.output

    def test_rm_recursive_deletes_thread(self, db: Path) -> None:
        p_out = run(db, "add", "parent")
        parent_id = p_out.split()[0]
        run(db, "reply", parent_id, "child")

        out = run(db, "rm", parent_id, "--yes", "--recursive")
        assert "deleted 2 notes" in out


class TestExport:
    def test_exports_everything(self, db: Path) -> None:
        run(db, "add", "parent")
        run(db, "add", "sibling")
        output = run(db, "export")
        assert "- parent" in output
        assert "- sibling" in output

    def test_orphan_promoted_to_root(self, db: Path) -> None:
        out = run(db, "add", "parent")
        parent_id = out.split()[0]
        run(db, "reply", parent_id, "child #target")

        output = run(db, "export", "--tag", "target")
        assert "- child" in output
        assert "- parent" not in output

    def test_root_filter(self, db: Path) -> None:
        p_out = run(db, "add", "parent")
        parent_id = p_out.split()[0]
        run(db, "reply", parent_id, "child")
        run(db, "add", "other")

        output = run(db, "export", "--root", parent_id)
        assert "- parent" in output
        assert "  - child" in output
        assert "other" not in output

    def test_writes_to_file(self, db: Path, tmp_path: Path) -> None:
        run(db, "add", "content")
        out_file = tmp_path / "out.md"
        run(db, "export", "-o", str(out_file))
        assert "- content" in out_file.read_text(encoding="utf-8")

    def test_conflicting_options_fail(self, db: Path) -> None:
        result = runner.invoke(
            app, ["--db", str(db), "export", "--root", "1", "--tag", "foo"]
        )
        assert result.exit_code != 0
        assert "cannot be combined" in result.output

    def test_import_outline_then_export(self, db: Path, tmp_path: Path) -> None:
        plan = tmp_path / "plan.md"
        plan.write_text("- parent\n  - child\n", encoding="utf-8")

        run(db, "import", str(plan), "--mode", "outline")

        output = run(db, "export")
        assert "- parent\n  - child\n" in output

    def test_multiple_outlines_keep_parents_local(
        self, db: Path, tmp_path: Path
    ) -> None:
        file_a = tmp_path / "a.md"
        file_a.write_text("- root A\n  - child A\n", encoding="utf-8")
        file_b = tmp_path / "b.md"
        file_b.write_text("- root B\n  - child B\n", encoding="utf-8")

        run(db, "import", str(file_a), str(file_b), "--mode", "outline")

        # root A is 1, child A is 2, root B is 3, child B is 4.
        # Check child B's parent is root B (3), not root A (1).
        import sqlite3

        conn = sqlite3.connect(db)
        query = "SELECT id, body, parent_id FROM notes ORDER BY id"
        rows = conn.execute(query).fetchall()
        assert rows[3] == (4, "child B", 3)


class FakeEmbedder:
    """Encodes a text as a bag of characters, deterministically.

    Enough for the wiring: identical texts get identical vectors, and a
    query sharing characters with a note scores above one that does not.
    No model is downloaded, so these tests run in CI.
    """

    dim = 8

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for character in text:
                rows[row][ord(character) % self.dim] += 1.0
        return rows


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> FakeEmbedder:
    embedder = FakeEmbedder()
    monkeypatch.setattr("hashline.ml.embed.load_model", lambda name=None: embedder)
    return embedder


class TestIndex:
    def test_embeds_every_note_and_reports_the_count(
        self, db: Path, fake_model: FakeEmbedder
    ) -> None:
        run(db, "add", "one")
        run(db, "add", "two")
        assert "indexed 2 notes" in run(db, "index")

    def test_a_second_run_has_nothing_to_do(
        self, db: Path, fake_model: FakeEmbedder
    ) -> None:
        run(db, "add", "one")
        run(db, "index")
        assert "nothing to index" in run(db, "index")

    def test_rebuild_re_embeds_what_is_already_there(
        self, db: Path, fake_model: FakeEmbedder
    ) -> None:
        run(db, "add", "one")
        run(db, "index")
        assert "indexed 1 notes" in run(db, "index", "--rebuild")

    def test_honours_limit(self, db: Path, fake_model: FakeEmbedder) -> None:
        for body in ("one", "two", "three"):
            run(db, "add", body)
        assert "indexed 2 notes" in run(db, "index", "--limit", "2")
        assert "indexed 1 notes" in run(db, "index")

    def test_stores_unit_length_vectors_under_the_prefixed_key(
        self, db: Path, fake_model: FakeEmbedder
    ) -> None:
        """Normalized on write, so a search is one matrix product.

        The key carries the prefix convention, not just the model name --
        vectors made under a different convention must never be read as if
        they were these.
        """
        run(db, "add", "one")
        run(db, "index")
        with Store.open(db) as store:
            rows = list(store.iter_embeddings(embed.EMBEDDING_KEY))
        assert len(rows) == 1
        vector = embed.unpack_vector(rows[0][1], expected_dim=FakeEmbedder.dim)
        assert np.isclose(float(np.linalg.norm(vector)), 1.0)

    def test_another_key_does_not_see_these_vectors(
        self, db: Path, fake_model: FakeEmbedder
    ) -> None:
        run(db, "add", "one")
        run(db, "index", "--model", "some/other-model")
        with Store.open(db) as store:
            assert list(store.iter_embeddings(embed.EMBEDDING_KEY)) == []
            assert len(list(store.iter_embeddings("some/other-model+query"))) == 1

    def test_without_the_extra_it_says_how_to_get_it(
        self, db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(name: str = "") -> object:
            raise embed.MlExtraNotInstalled("needs the 'ml' extra: uv sync --extra ml")

        monkeypatch.setattr("hashline.ml.embed.load_model", refuse)
        run(db, "add", "one")
        result = runner.invoke(app, ["--db", str(db), "index"])
        assert result.exit_code == 1
        assert "--extra ml" in result.output

    def test_an_empty_database_needs_no_model(self, db: Path) -> None:
        # No fake_model fixture: nothing should try to load one.
        assert "nothing to index" in run(db, "index")
