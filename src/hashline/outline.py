"""Markdown outline export and import."""

from collections.abc import Sequence
from dataclasses import dataclass

from hashline.models import Note


@dataclass(frozen=True, slots=True)
class OutlineNode:
    """A note and its replies in a tree."""

    note: Note
    children: tuple["OutlineNode", ...] = ()


def build_tree(notes: Sequence[Note]) -> list[OutlineNode]:
    """Build a forest of OutlineNodes from a flat list of notes.
    
    A note whose parent is not in the list becomes a root.
    Siblings are ordered by (created_at, id) ascending.
    """
    children_by_parent: dict[int | None, list[Note]] = {}
    note_ids = {n.id for n in notes}

    for n in notes:
        # A note whose parent is NOT in the given list becomes a ROOT
        parent = n.parent_id if n.parent_id in note_ids else None
        children_by_parent.setdefault(parent, []).append(n)

    for group in children_by_parent.values():
        group.sort(key=lambda n: (n.created_at, n.id))

    def build_node(note: Note) -> OutlineNode:
        children = children_by_parent.get(note.id, [])
        return OutlineNode(
            note=note,
            children=tuple(build_node(c) for c in children),
        )

    return [build_node(root) for root in children_by_parent.get(None, [])]


def render_markdown(roots: Sequence[OutlineNode], *, indent: str = "  ") -> str:
    """Render a tree of notes as a Markdown indented bullet list."""
    lines = []

    def _render(node: OutlineNode, current_indent: str) -> None:
        body_lines = node.note.body.splitlines()

        if not body_lines:
            lines.append(f"{current_indent}- ")
        else:
            lines.append(f"{current_indent}- {body_lines[0]}")
            for line in body_lines[1:]:
                # continuation lines are indented to align with the text
                lines.append(f"{current_indent}  {line}")

        for child in node.children:
            _render(child, current_indent + indent)

    for root in roots:
        _render(root, "")

    if not lines:
        return ""

    return "\n".join(lines) + "\n"
