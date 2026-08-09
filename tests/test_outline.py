from datetime import UTC, datetime

from hashline.models import Note
from hashline.outline import (
    OutlineItem,
    OutlineNode,
    build_tree,
    render_markdown,
    split_outline,
)


class TestBuildTree:
    def test_flat_list(self) -> None:
        n1 = Note(1, "one", datetime(2020, 1, 1, tzinfo=UTC))
        n2 = Note(2, "two", datetime(2020, 1, 2, tzinfo=UTC))

        roots = build_tree([n1, n2])
        assert roots == [
            OutlineNode(n1, ()),
            OutlineNode(n2, ()),
        ]

    def test_two_and_three_levels(self) -> None:
        n1 = Note(1, "root", datetime(2020, 1, 1, tzinfo=UTC))
        n2 = Note(2, "child", datetime(2020, 1, 2, tzinfo=UTC), parent_id=1)
        n3 = Note(3, "grandchild", datetime(2020, 1, 3, tzinfo=UTC), parent_id=2)
        n4 = Note(4, "sibling", datetime(2020, 1, 4, tzinfo=UTC), parent_id=1)

        roots = build_tree([n1, n2, n3, n4])
        assert roots == [
            OutlineNode(
                n1,
                (
                    OutlineNode(n2, (OutlineNode(n3, ()),)),
                    OutlineNode(n4, ()),
                ),
            )
        ]

    def test_sibling_ordering(self) -> None:
        # Ordered by (created_at, id) ascending
        n1 = Note(1, "root", datetime(2020, 1, 1, tzinfo=UTC))
        n2 = Note(2, "child 2", datetime(2020, 1, 3, tzinfo=UTC), parent_id=1)
        n3 = Note(3, "child 1", datetime(2020, 1, 2, tzinfo=UTC), parent_id=1)

        roots = build_tree([n1, n2, n3])
        assert roots[0].children == (
            OutlineNode(n3, ()),
            OutlineNode(n2, ()),
        )

    def test_orphan_promoted_to_root(self) -> None:
        # parent 1 is missing, so 2 should become a root
        n2 = Note(2, "orphan child", datetime(2020, 1, 2, tzinfo=UTC), parent_id=1)

        roots = build_tree([n2])
        assert roots == [
            OutlineNode(n2, ()),
        ]


class TestRenderMarkdown:
    def test_empty_input(self) -> None:
        assert render_markdown([]) == ""

    def test_flat_list(self) -> None:
        n1 = Note(1, "one", datetime(2020, 1, 1, tzinfo=UTC))
        n2 = Note(2, "two", datetime(2020, 1, 2, tzinfo=UTC))

        roots = build_tree([n1, n2])
        assert render_markdown(roots) == "- one\n- two\n"

    def test_two_and_three_levels(self) -> None:
        n1 = Note(1, "root", datetime(2020, 1, 1, tzinfo=UTC))
        n2 = Note(2, "child", datetime(2020, 1, 2, tzinfo=UTC), parent_id=1)
        n3 = Note(3, "grandchild", datetime(2020, 1, 3, tzinfo=UTC), parent_id=2)

        roots = build_tree([n1, n2, n3])
        assert render_markdown(roots) == "- root\n  - child\n    - grandchild\n"

    def test_multi_line_body(self) -> None:
        n1 = Note(1, "line 1\nline 2\nline 3", datetime(2020, 1, 1, tzinfo=UTC))
        n2 = Note(2, "child\nmore text", datetime(2020, 1, 2, tzinfo=UTC), parent_id=1)

        roots = build_tree([n1, n2])
        out = render_markdown(roots)
        assert out == ("- line 1\n  line 2\n  line 3\n  - child\n    more text\n")

    def test_empty_body(self) -> None:
        n1 = Note(1, "", datetime(2020, 1, 1, tzinfo=UTC))
        roots = build_tree([n1])
        assert render_markdown(roots) == "- \n"

    def test_custom_indent(self) -> None:
        n1 = Note(1, "root", datetime(2020, 1, 1, tzinfo=UTC))
        n2 = Note(2, "child", datetime(2020, 1, 2, tzinfo=UTC), parent_id=1)

        roots = build_tree([n1, n2])
        out = render_markdown(roots, indent="    ")
        assert out == "- root\n    - child\n"


class TestSplitOutline:
    def test_preamble_and_bullets(self) -> None:
        text = "preamble\n- one\n  - two\n- three"
        items = split_outline(text)
        assert items == [
            OutlineItem("preamble", 0),
            OutlineItem("one", 0),
            OutlineItem("two", 1),
            OutlineItem("three", 0),
        ]

    def test_code_fence(self) -> None:
        text = "- one\n  ```\n  - not a bullet\n  ```\n- two"
        items = split_outline(text)
        assert items == [
            OutlineItem("one\n```\n- not a bullet\n```", 0),
            OutlineItem("two", 0),
        ]

    def test_mixed_indent(self) -> None:
        text = "- one\n  - two\n    - three\n\t- four\n- five"
        items = split_outline(text)
        assert items == [
            OutlineItem("one", 0),
            OutlineItem("two", 1),
            OutlineItem("three", 2),
            OutlineItem("four", 2),
            OutlineItem("five", 0),
        ]

    def test_multi_line_item(self) -> None:
        text = "- one\n  two\n  three\n- four"
        items = split_outline(text)
        assert items == [
            OutlineItem("one\ntwo\nthree", 0),
            OutlineItem("four", 0),
        ]
