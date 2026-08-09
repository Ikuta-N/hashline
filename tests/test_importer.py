from pathlib import Path

import pytest

from hashline.importer import (
    SPLITTERS,
    Document,
    parse_document,
    parse_documents,
    split_headings,
    split_lines,
)
from hashline.outline import build_tree, render_markdown
from hashline.store import Store


class TestSplitLines:
    def test_one_note_per_line(self) -> None:
        assert split_lines("first\nsecond") == ["first", "second"]

    def test_drops_blank_lines(self) -> None:
        assert split_lines("first\n\n   \nsecond\n") == ["first", "second"]

    def test_strips_each_line(self) -> None:
        assert split_lines("  padded  \n\tindented") == ["padded", "indented"]

    def test_empty_text(self) -> None:
        assert split_lines("") == []


class TestSplitHeadings:
    def test_one_note_per_section(self) -> None:
        text = "# One\nbody one\n\n## Two\nbody two"
        assert split_headings(text) == ["# One\nbody one", "## Two\nbody two"]

    def test_keeps_text_before_the_first_heading(self) -> None:
        assert split_headings("preamble\n\n# One\nbody") == ["preamble", "# One\nbody"]

    def test_keeps_a_heading_with_no_body(self) -> None:
        assert split_headings("# One\n\n# Two\nbody") == ["# One", "# Two\nbody"]

    def test_all_six_heading_levels_start_a_section(self) -> None:
        text = "\n".join(f"{'#' * level} H{level}" for level in range(1, 7))
        assert len(split_headings(text)) == 6

    def test_seven_hashes_is_not_a_heading(self) -> None:
        assert split_headings("# One\n####### not a heading") == [
            "# One\n####### not a heading"
        ]

    def test_hash_without_a_space_is_not_a_heading(self) -> None:
        assert split_headings("# One\n#tag stays in the body") == [
            "# One\n#tag stays in the body"
        ]

    def test_ignores_headings_inside_a_backtick_fence(self) -> None:
        text = "# One\n\n```\n# not a heading\n```\n\n## Two"
        assert split_headings(text) == [
            "# One\n\n```\n# not a heading\n```",
            "## Two",
        ]

    def test_ignores_headings_inside_a_tilde_fence(self) -> None:
        text = "# One\n\n~~~\n# not a heading\n~~~\n\n## Two"
        assert len(split_headings(text)) == 2

    def test_a_different_fence_marker_does_not_close_the_fence(self) -> None:
        text = "# One\n\n```\n~~~\n# not a heading\n```\n\n## Two"
        assert len(split_headings(text)) == 2

    def test_no_headings_yields_one_note(self) -> None:
        assert split_headings("just\nsome\ntext") == ["just\nsome\ntext"]

    def test_empty_text(self) -> None:
        assert split_headings("") == []

    def test_whitespace_only_text(self) -> None:
        assert split_headings("\n  \n\t\n") == []


class TestParseDocument:
    def test_defaults_to_line_mode(self) -> None:
        drafts = parse_document(Document(source="a.txt", text="one\ntwo"))
        assert [draft.body for draft in drafts] == ["one", "two"]

    def test_heading_mode(self) -> None:
        doc = Document(source="a.md", text="# One\nbody\n## Two\nbody")
        drafts = parse_document(doc, mode="heading")
        assert [draft.body for draft in drafts] == ["# One\nbody", "## Two\nbody"]

    def test_records_the_source_on_every_draft(self) -> None:
        drafts = parse_document(Document(source="a.txt", text="one\ntwo"))
        assert {draft.source for draft in drafts} == {"a.txt"}

    def test_common_tags_are_normalized_and_deduplicated(self) -> None:
        doc = Document(source="a.txt", text="one")
        drafts = parse_document(doc, common_tags=["#Imported", "imported", "Archive"])
        assert drafts[0].extra_tags == ("imported", "archive")

    def test_common_tags_do_not_change_the_body(self) -> None:
        doc = Document(source="a.txt", text="a plain line")
        assert parse_document(doc, common_tags=["imported"])[0].body == "a plain line"

    def test_leaves_created_at_for_the_store_to_fill_in(self) -> None:
        drafts = parse_document(Document(source="a.txt", text="one"))
        assert drafts[0].created_at is None

    def test_rejects_an_unusable_common_tag(self) -> None:
        with pytest.raises(ValueError):
            parse_document(Document(source="a.txt", text="one"), common_tags=["a b"])

    def test_rejects_an_unknown_mode(self) -> None:
        with pytest.raises(ValueError, match="unknown split mode"):
            parse_document(
                Document(source="a.txt", text="one"),
                mode="paragraph",  # type: ignore[arg-type]
            )

    def test_empty_document_yields_nothing(self) -> None:
        assert parse_document(Document(source="a.txt", text="")) == []

    def test_outline_mode(self) -> None:
        text = "preamble\n- one\n  - two\n- three"
        doc = Document(source="a.md", text=text)
        drafts = parse_document(doc, mode="outline")

        assert len(drafts) == 4
        assert drafts[0].body == "preamble"
        assert drafts[0].parent_index is None
        assert drafts[1].body == "one"
        assert drafts[1].parent_index is None
        assert drafts[2].body == "two"
        assert drafts[2].parent_index == 1
        assert drafts[3].body == "three"
        assert drafts[3].parent_index is None

    def test_outline_clamps_overdeep_jumps(self) -> None:
        text = "- one\n        - jump"  # jumped by 8 spaces, depth 2 but clamped to 1
        doc = Document(source="a.md", text=text)
        drafts = parse_document(doc, mode="outline")

        assert len(drafts) == 2
        assert drafts[0].body == "one"
        assert drafts[0].parent_index is None
        assert drafts[1].body == "jump"
        assert drafts[1].parent_index == 0


class TestParseDocuments:
    def test_keeps_document_order(self) -> None:
        drafts = parse_documents(
            [
                Document(source="a.txt", text="a1\na2"),
                Document(source="b.txt", text="b1"),
            ]
        )
        assert [(d.source, d.body) for d in drafts] == [
            ("a.txt", "a1"),
            ("a.txt", "a2"),
            ("b.txt", "b1"),
        ]

    def test_no_documents(self) -> None:
        assert parse_documents([]) == []

    def test_shifts_parent_index_for_subsequent_documents(self) -> None:
        doc1 = Document(source="a.md", text="- root A\n  - child A")
        doc2 = Document(source="b.md", text="- root B\n  - child B")
        drafts = parse_documents([doc1, doc2], mode="outline")
        
        assert len(drafts) == 4
        assert drafts[0].body == "root A"
        assert drafts[0].parent_index is None
        assert drafts[1].body == "child A"
        assert drafts[1].parent_index == 0
        assert drafts[2].body == "root B"
        assert drafts[2].parent_index is None
        assert drafts[3].body == "child B"
        assert drafts[3].parent_index == 2


class TestSplitters:
    def test_registry_covers_both_modes(self) -> None:
        assert set(SPLITTERS) == {"line", "heading"}


class TestAgainstFixtures:
    """The importer is pure, so the test does the reading."""

    def test_line_mode_on_a_text_file(self, notes_dir: Path) -> None:
        text = (notes_dir / "scratch.txt").read_text(encoding="utf-8")
        drafts = parse_document(Document(source="scratch.txt", text=text))
        assert [draft.body for draft in drafts] == [
            "FTS5 の bm25 を調べた #sqlite",
            "trigram tokenizer は日本語に効く #sqlite #検索",
            "一行メモをためす",
        ]

    def test_heading_mode_on_a_markdown_file(self, notes_dir: Path) -> None:
        text = (notes_dir / "daily.md").read_text(encoding="utf-8")
        drafts = parse_document(Document(source="daily.md", text=text), mode="heading")
        assert len(drafts) == 3
        assert drafts[0].body.startswith("# 2026-08-09")
        assert drafts[2].body == "## やること\n\nREADME を書く #todo"

    def test_fenced_code_does_not_start_a_section(self, notes_dir: Path) -> None:
        text = (notes_dir / "fenced.md").read_text(encoding="utf-8")
        drafts = parse_document(Document(source="fenced.md", text=text), mode="heading")
        assert [draft.body.splitlines()[0] for draft in drafts] == [
            "前書き。ここは見出しの前にある。",
            "# コード例",
            "## 続き",
        ]

    def test_empty_file_yields_nothing(self, notes_dir: Path) -> None:
        text = (notes_dir / "empty.md").read_text(encoding="utf-8")
        assert parse_document(Document(source="empty.md", text=text)) == []

    def test_outline_round_trip(self, tmp_path: Path) -> None:
        plan_path = Path("tests/fixtures/outline/plan.md")
        text = plan_path.read_text(encoding="utf-8")
        doc = Document(source="plan.md", text=text)
        drafts = parse_document(doc, mode="outline")

        db_path = tmp_path / "test.db"
        store = Store.open(db_path)
        store.init_schema()
        store.add_notes(drafts)

        notes = list(store.list_notes())
        roots = build_tree(notes)
        out = render_markdown(roots)

        # Check that the shape matches
        doc_parsed = parse_document(
            Document(source="plan.md", text=out), mode="outline"
        )

        assert len(drafts) == len(doc_parsed)
        for expected, actual in zip(drafts, doc_parsed, strict=True):
            assert expected.parent_index == actual.parent_index
            assert expected.body.strip() == actual.body.strip()
