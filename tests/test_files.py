"""Tests for the shared filesystem I/O module."""

from pathlib import Path

from hashline.files import decode_uploads, read_documents


class TestReadDocuments:
    def test_reads_an_explicit_file_whatever_its_suffix(self, notes_dir: Path) -> None:
        documents, skipped = read_documents([notes_dir / "ignored.json"])
        assert len(documents) == 1
        assert skipped == []

    def test_directory_walk_keeps_only_text_suffixes(self, notes_dir: Path) -> None:
        documents, _ = read_documents([notes_dir])
        assert {Path(doc.source).name for doc in documents} == {
            "daily.md",
            "empty.md",
            "fenced.md",
            "scratch.txt",
        }

    def test_reports_a_file_it_cannot_decode(self, tmp_path: Path) -> None:
        broken = tmp_path / "broken.txt"
        broken.write_bytes(b"\xff\xfe not utf-8")
        documents, skipped = read_documents([tmp_path])
        assert documents == []
        assert len(skipped) == 1
        assert "broken.txt" in skipped[0]


class TestDecodeUploads:
    def test_decodes_valid_files(self) -> None:
        items = [
            ("notes.md", b"some markdown"),
            ("log.txt", b"some text"),
        ]
        documents, skipped = decode_uploads(items)
        assert len(documents) == 2
        assert documents[0].source == "notes.md"
        assert documents[0].text == "some markdown"
        assert documents[1].source == "log.txt"
        assert documents[1].text == "some text"
        assert skipped == []

    def test_skips_wrong_suffix(self) -> None:
        items = [
            ("notes.md", b"valid"),
            ("image.png", b"fake png bytes"),
        ]
        documents, skipped = decode_uploads(items)
        assert len(documents) == 1
        assert len(skipped) == 1
        assert "image.png" in skipped[0]
        assert "not a text or markdown file" in skipped[0]

    def test_skips_undecodable_bytes(self) -> None:
        items = [
            ("broken.md", b"\xff\xfe not utf-8"),
        ]
        documents, skipped = decode_uploads(items)
        assert len(documents) == 0
        assert len(skipped) == 1
        assert "broken.md" in skipped[0]

    def test_handles_empty_input(self) -> None:
        documents, skipped = decode_uploads([])
        assert documents == []
        assert skipped == []
