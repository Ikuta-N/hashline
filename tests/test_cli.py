"""Smoke tests for the CLI adapter: wiring, not note logic."""

import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from hashline.cli import app
from hashline.ml import embed, hybrid
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


class TestStats:
    def test_overview_on_an_empty_database(self, db: Path) -> None:
        output = run(db, "stats")
        assert "notes: 0" in output
        assert "tags:  0" in output
        assert "works: 0" in output
        assert "no notes yet" in output
        assert "None" not in output

    def test_overview_with_seeded_data(self, db: Path) -> None:
        run(db, "bib", "import", str(BIB_FIXTURE))
        run(db, "read", "start", "smith2020")
        run(db, "add", "one #rust")
        run(db, "add", "two #async", "--no-context")
        output = run(db, "stats")
        assert "notes: 2" in output
        assert "works: 1" in output
        assert "first note:" in output
        assert "last note:" in output

    def test_activity_selector(self, db: Path) -> None:
        run(db, "add", "one")
        run(db, "add", "two")
        output = run(db, "stats", "--activity")
        assert "count" in output
        assert "period" in output

    def test_tags_selector(self, db: Path) -> None:
        run(db, "add", "one #rust")
        run(db, "add", "two #rust")
        run(db, "add", "three #python")
        output = run(db, "stats", "--tags")
        assert "rust" in output
        assert "python" in output

    def test_tags_selector_honours_top(self, db: Path) -> None:
        run(db, "add", "a #popular")
        run(db, "add", "b #popular")
        run(db, "add", "c #rare")
        output = run(db, "stats", "--tags", "--top", "1")
        assert "popular" in output
        assert "rare" not in output

    @pytest.mark.parametrize("top", ["-1", "0", "99999999999999999999"])
    def test_an_unusable_top_is_a_bad_parameter_not_a_traceback(
        self, db: Path, top: str
    ) -> None:
        """`--top -1` used to mean *every tag*: SQLite reads LIMIT -1 as none.

        `"Traceback" not in output` cannot express this: CliRunner captures
        the exception instead of printing it, so an uncaught OverflowError
        leaves the output EMPTY and a non-zero exit -- both of which that
        assertion happily accepts. Verified by stubbing the guard out: the
        test passed with the bug restored. What distinguishes the two is
        whether an exception escaped at all, and whether the message a user
        needs actually reached them.
        """
        run(db, "add", "a #popular")
        result = runner.invoke(app, ["--db", str(db), "stats", "--tags", "--top", top])
        assert result.exit_code != 0
        assert result.exception is None or isinstance(result.exception, SystemExit)
        assert "top must be between" in result.output

    def test_reading_selector(self, db: Path) -> None:
        run(db, "bib", "import", str(BIB_FIXTURE))
        run(db, "read", "start", "smith2020")
        run(db, "add", "a note on it")
        output = run(db, "stats", "--reading")
        assert "smith2020" in output
        assert "note_count" in output

    def test_threads_selector(self, db: Path) -> None:
        out = run(db, "add", "root note")
        root_id = out.split()[0]
        run(db, "reply", root_id, "a reply")
        output = run(db, "stats", "--threads")
        assert "reply_count" in output
        assert "max_depth" in output

    def test_csv_writes_the_selected_frame(self, db: Path, tmp_path: Path) -> None:
        run(db, "add", "one #rust", "-t", "extra")
        run(db, "add", "two #rust")
        csv_path = tmp_path / "tags.csv"
        run(db, "stats", "--tags", "--csv", str(csv_path))
        contents = csv_path.read_text(encoding="utf-8")
        assert "period" in contents
        assert "rust" in contents
        assert "2" in contents

    @pytest.mark.parametrize(
        "args",
        [
            ["--reading", "--freq", "W"],
            ["--threads", "--freq", "W"],
            ["--freq", "W"],
            ["--reading", "--top", "3"],
            ["--activity", "--top", "3"],
            ["--top", "3"],
        ],
    )
    def test_an_option_that_would_do_nothing_is_refused(
        self, db: Path, args: list[str]
    ) -> None:
        """Refusing two selectors but silently discarding --freq was inconsistent."""
        run(db, "add", "one #rust")
        result = runner.invoke(app, ["--db", str(db), "stats", *args])
        assert result.exit_code != 0
        assert "only applies to" in result.output

    @pytest.mark.parametrize(
        "args", [["--activity", "--freq", "W"], ["--tags", "--freq", "W"]]
    )
    def test_freq_is_accepted_where_it_applies(self, db: Path, args: list[str]) -> None:
        run(db, "add", "one #rust")
        assert "period" in run(db, "stats", *args)

    def test_csv_pages_are_joined_not_a_python_repr(
        self, db: Path, tmp_path: Path
    ) -> None:
        """A CSV is read by a program, so `"['12-15', '40']"` served no one."""
        import csv as csv_module

        run(db, "bib", "import", str(BIB_FIXTURE))
        run(db, "read", "start", "smith2020")
        run(db, "add", "first", "--page", "12-15")
        run(db, "add", "second", "--page", "第3章")
        csv_path = tmp_path / "reading.csv"
        run(db, "stats", "--reading", "--csv", str(csv_path))

        with csv_path.open(encoding="utf-8", newline="") as handle:
            row = next(iter(csv_module.DictReader(handle)))
        assert row["pages"] == "12-15;第3章"

    def test_csv_pages_stay_a_list_on_screen(self, db: Path) -> None:
        """Only the copy on its way to disk is flattened."""
        run(db, "bib", "import", str(BIB_FIXTURE))
        run(db, "read", "start", "smith2020")
        run(db, "add", "first", "--page", "12-15")
        assert "[12-15]" in run(db, "stats", "--reading")

    def test_csv_with_no_selector_writes_the_overview(
        self, db: Path, tmp_path: Path
    ) -> None:
        run(db, "add", "one #rust")
        csv_path = tmp_path / "overview.csv"
        run(db, "stats", "--csv", str(csv_path))
        contents = csv_path.read_text(encoding="utf-8")
        assert "note_count" in contents
        assert "tag_count" in contents
        lines = contents.splitlines()
        assert len(lines) == 2
        assert "1" in lines[1]

    def test_two_selectors_at_once_fails(self, db: Path) -> None:
        result = runner.invoke(
            app, ["--db", str(db), "stats", "--activity", "--tags"]
        )
        assert result.exit_code != 0
        assert "only one of" in result.output

    def test_bad_freq_fails(self, db: Path) -> None:
        result = runner.invoke(
            app, ["--db", str(db), "stats", "--activity", "--freq", "M"]
        )
        assert result.exit_code != 0

    def test_bad_freq_with_tags_selector_fails(self, db: Path) -> None:
        result = runner.invoke(
            app, ["--db", str(db), "stats", "--tags", "--freq", "bogus"]
        )
        assert result.exit_code != 0


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
    # The fake stands in for an installed backend, so availability has to
    # agree with it -- the semantic search asks before it reads the index.
    monkeypatch.setattr("hashline.ml.embed.is_available", lambda: True)
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


class TestSemanticSearch:
    def test_finds_a_note_the_keyword_index_cannot_reach(
        self, db: Path, fake_model: FakeEmbedder
    ) -> None:
        """The whole point: a match with no shared substring.

        The fake embedder scores on shared characters, so "xyz" reaches the
        note containing them while the trigram index -- searched for the
        literal phrase -- does not.
        """
        run(db, "add", "aaa xyz aaa")
        run(db, "add", "bbb ccc ddd")
        run(db, "index")
        assert "no matches" in run(db, "search", "xyz zyx")
        assert "aaa xyz aaa" in run(db, "search", "xyz zyx", "--semantic")

    def test_a_keyword_match_still_ranks(
        self, db: Path, fake_model: FakeEmbedder
    ) -> None:
        run(db, "add", "bm25 を調べた")
        run(db, "index")
        assert "bm25 を調べた" in run(db, "search", "bm25", "--semantic")

    def test_scores_are_the_fused_ranks(
        self, db: Path, fake_model: FakeEmbedder
    ) -> None:
        # RRF with k=60: a note first in both lists scores 2/61 = 0.0328.
        run(db, "add", "one note")
        run(db, "index")
        assert "0.0328" in run(db, "search", "one note", "--semantic")

    def test_honours_the_tag_filter(
        self, db: Path, fake_model: FakeEmbedder
    ) -> None:
        run(db, "add", "aaa xyz #keep")
        run(db, "add", "aaa xyz #drop")
        run(db, "index")
        output = run(db, "search", "xyz", "--semantic", "--tag", "keep")
        assert "#keep" in output
        assert "#drop" not in output

    def test_honours_limit(self, db: Path, fake_model: FakeEmbedder) -> None:
        for body in ("aaa one", "aaa two", "aaa three"):
            run(db, "add", body)
        run(db, "index")
        output = run(db, "search", "aaa", "--semantic", "--limit", "2")
        assert len([line for line in output.splitlines() if line.strip()]) == 2

    def test_says_so_when_nothing_is_indexed(
        self, db: Path, fake_model: FakeEmbedder
    ) -> None:
        run(db, "add", "one")
        output = run(db, "search", "one", "--semantic")
        assert "run `hashline index`" in output
        assert "no matches" not in output

    def test_a_missing_extra_is_named_before_the_empty_index(
        self, db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The first message has to be the actual cause.

        Checked the other way round, a database with no vectors answers "run
        hashline index" -- and only that command then reveals the extra was
        never installed.
        """
        monkeypatch.setattr("hashline.ml.embed.is_available", lambda: False)
        run(db, "add", "one")
        result = runner.invoke(app, ["--db", str(db), "search", "one", "--semantic"])
        assert result.exit_code == 1
        assert "--extra ml" in result.output
        assert "hashline index" not in result.output

    def test_warns_about_notes_added_since_the_last_index(
        self, db: Path, fake_model: FakeEmbedder
    ) -> None:
        """A short result list must not be the only sign of a stale index."""
        run(db, "add", "aaa one")
        run(db, "index")
        run(db, "add", "aaa two")
        result = runner.invoke(app, ["--db", str(db), "search", "aaa", "--semantic"])
        assert result.exit_code == 0
        assert "1 notes are not indexed yet" in result.output

    def test_without_the_extra_it_says_how_to_get_it(
        self, db: Path, fake_model: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run(db, "add", "one")
        run(db, "index")

        def refuse(name: str = "") -> object:
            raise embed.MlExtraNotInstalled("needs the 'ml' extra: uv sync --extra ml")

        monkeypatch.setattr("hashline.ml.embed.load_model", refuse)
        # A loaded model is cached for the life of the process, so the index
        # above would otherwise satisfy this search without a second load --
        # which is the point of the cache, and has to be undone to simulate a
        # backend that cannot be loaded at all.
        hybrid.forget_models()
        result = runner.invoke(app, ["--db", str(db), "search", "one", "--semantic"])
        assert result.exit_code == 1
        assert "--extra ml" in result.output

    def test_a_plain_search_needs_no_model(self, db: Path) -> None:
        # No fake_model fixture: the keyword path must not touch the backend.
        run(db, "add", "bm25 を調べた")
        assert "bm25 を調べた" in run(db, "search", "bm25")

    def test_reports_no_matches_like_the_keyword_path(
        self, db: Path, fake_model: FakeEmbedder
    ) -> None:
        """The two paths answer an empty result the same way.

        A ranking over indexed notes always has entries, so a zero limit is
        the only way to empty it -- but the two searches sit behind one
        command and must not disagree about what nothing looks like.
        """
        run(db, "add", "one")
        run(db, "index")
        assert "no matches" in run(db, "search", "one", "--limit", "0")
        assert "no matches" in run(db, "search", "one", "--semantic", "--limit", "0")


class TestStatsTimezone:
    """One command must not report the same note at two different hours."""

    def test_event_timestamps_print_in_local_time(self, db: Path) -> None:
        """The overview and `hashline list` already do; the frames did not."""
        run(db, "add", "a note")
        overview = run(db, "stats")
        threads = run(db, "stats", "--threads")
        hour = [
            line.split("first note: ")[1][:16]
            for line in overview.splitlines()
            if line.startswith("first note:")
        ][0]
        assert hour in threads, (
            f"the overview says {hour!r} but --threads reports\n{threads}"
        )

    @pytest.mark.skipif(
        not hasattr(time, "tzset"), reason="TZ only takes effect on POSIX"
    )
    def test_a_winter_note_is_not_printed_at_a_summer_offset(
        self, db: Path, monkeypatch: pytest.MonkeyPatch, in_zone: None
    ) -> None:
        """The offset belongs to the timestamp, not to the moment of running.

        `test_event_timestamps_print_in_local_time` above cannot catch this on
        its own: it captures "now", so both sides agree whenever the run and
        the note fall in the same half of the year. Seeding a January note and
        reading it from a July-ish zone is what splits a per-instant conversion
        from a snapshot of `datetime.now()`.
        """
        monkeypatch.setenv("TZ", "America/Los_Angeles")
        time.tzset()
        with Store.open(db) as store:
            store.add_note(
                "a winter note", created_at=datetime(2026, 1, 15, 20, tzinfo=UTC)
            )

        # 20:00 UTC in January is 12:00 PST. A fixed PDT offset would say 13:00.
        assert "2026-01-15 12:00:00" in run(db, "stats", "--threads")
        assert "2026-01-15 12:00" in run(db, "stats")

    def test_period_buckets_stay_in_utc(self, db: Path) -> None:
        """A bucket is not an instant.

        Shifting the resample index would label a UTC day "09:00" for a reader
        nine hours ahead, which says something the data does not.
        """
        run(db, "add", "a note")
        assert "+00:00" in run(db, "stats", "--activity")


class TestListTagsLimit:
    """The cap belongs to the LIMIT, so it is checked where the LIMIT is issued.

    `hashline stats --tags --top` was fixed one call away from this, while
    `hashline tags --limit` -- the other caller of `Store.list_tags` -- still
    read `-1` as *every tag* and blew up on a large integer.
    """

    def test_a_negative_limit_is_refused_not_read_as_no_limit(
        self, db: Path
    ) -> None:
        for index in range(3):
            run(db, "add", f"note {index} #tag{index}")
        result = runner.invoke(app, ["--db", str(db), "tags", "--limit", "-1"])
        assert result.exit_code != 0
        assert "limit must be between" in result.output

    def test_a_limit_sqlite_cannot_bind_is_refused(self, db: Path) -> None:
        run(db, "add", "a #tag")
        result = runner.invoke(
            app, ["--db", str(db), "tags", "--limit", "99999999999999999999"]
        )
        assert result.exit_code != 0
        assert result.exception is None or isinstance(result.exception, SystemExit)
        assert "limit must be between" in result.output

    def test_a_usable_limit_still_caps(self, db: Path) -> None:
        for index in range(3):
            run(db, "add", f"note {index} #tag{index}")
        assert len(run(db, "tags", "--limit", "2").splitlines()) == 2


class TestStatsFreqEdges:
    def test_an_empty_freq_is_refused_like_any_other_bad_one(self, db: Path) -> None:
        """`freq or "D"` read "" as "not given" and printed daily buckets.

        The guard above it asks `is not None`, so an empty string had already
        been accepted as "the user passed --freq" -- and then silently became
        the default anyway.
        """
        run(db, "add", "a note")
        result = runner.invoke(
            app, ["--db", str(db), "stats", "--activity", "--freq", ""]
        )
        assert result.exit_code != 0
        assert "freq must be one of" in result.output


class TestImplicitAdd:
    """`hashline TEXT` is `hashline add TEXT`: the note is what gets typed most."""

    def test_japanese_text_becomes_a_note(self, db: Path) -> None:
        output = run(db, "今日は寝不足だった")
        assert "今日は寝不足だった" in output

    def test_text_with_spaces_becomes_a_note(self, db: Path) -> None:
        output = run(db, "bm25 を調べた #sqlite")
        assert "[sqlite]" in output

    def test_text_starting_with_a_hash_becomes_a_note(self, db: Path) -> None:
        output = run(db, "#日記 晴れ")
        assert "[日記]" in output

    def test_the_add_options_still_apply(self, db: Path) -> None:
        output = run(db, "眠い", "--tag", "体調")
        assert "[体調]" in output

    def test_a_command_name_is_still_a_command(self, db: Path) -> None:
        run(db, "add", "a note")
        assert "a note" in run(db, "list")

    def test_a_mistyped_command_is_an_error_not_a_note(self, db: Path) -> None:
        """The reason a lone ASCII word is left alone.

        Sending every unknown word to `add` would turn a typo into a note that
        nobody meant to write, and the mistake would only surface later, in the
        timeline.
        """
        result = runner.invoke(app, ["--db", str(db), "serach"])
        assert result.exit_code != 0
        assert "No such command 'serach'" in result.output
        assert "no notes yet" in run(db, "list")

    def test_an_option_is_left_to_the_group(self, db: Path) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.output


class TestServe:
    """`hashline` with no command opens the web UI."""

    @pytest.fixture
    def calls(self, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
        """Record what `serve` would hand uvicorn, without binding a port."""
        import uvicorn

        recorded: list[dict[str, object]] = []

        def fake_run(target: str, **kwargs: object) -> None:
            recorded.append({"target": target, **kwargs})

        monkeypatch.setattr(uvicorn, "run", fake_run)
        # Registered with monkeypatch so that _run_server's write is undone.
        monkeypatch.setenv("HASHLINE_DB", "")
        return recorded

    def test_no_command_opens_the_web_ui(
        self, db: Path, calls: list[dict[str, object]]
    ) -> None:
        result = runner.invoke(app, ["--db", str(db)])
        assert result.exit_code == 0, result.output
        assert calls == [
            {
                "target": "hashline.web.app:app",
                "host": "127.0.0.1",
                "port": 8000,
                "reload": False,
            }
        ]

    def test_serve_passes_its_options_through(
        self, db: Path, calls: list[dict[str, object]]
    ) -> None:
        result = runner.invoke(
            app, ["--db", str(db), "serve", "--port", "9000", "--reload"]
        )
        assert result.exit_code == 0, result.output
        assert calls[0]["port"] == 9000
        assert calls[0]["reload"] is True

    def test_the_db_option_reaches_the_server(
        self, db: Path, calls: list[dict[str, object]]
    ) -> None:
        """The web adapter resolves the database itself, from the environment."""
        import os

        runner.invoke(app, ["--db", str(db), "serve"])
        assert os.environ["HASHLINE_DB"] == str(db)

    def test_the_default_host_is_this_machine(
        self, db: Path, calls: list[dict[str, object]]
    ) -> None:
        """The import routes read the local filesystem; do not offer them around."""
        runner.invoke(app, ["--db", str(db), "serve"])
        assert calls[0]["host"] == "127.0.0.1"

    def test_a_host_off_this_machine_is_called_out(
        self, db: Path, calls: list[dict[str, object]]
    ) -> None:
        """Nothing authenticates and `/import` reads local files.

        The README says not to expose the app, but the person typing
        `--host 0.0.0.0` is not reading the README at that moment.
        """
        result = runner.invoke(app, ["--db", str(db), "serve", "--host", "0.0.0.0"])
        assert result.exit_code == 0, result.output
        assert "reachable from other machines" in result.output

    def test_a_loopback_host_is_not_called_out(
        self, db: Path, calls: list[dict[str, object]]
    ) -> None:
        for host in ("127.0.0.1", "::1", "localhost"):
            result = runner.invoke(app, ["--db", str(db), "serve", "--host", host])
            assert "reachable from other machines" not in result.output
