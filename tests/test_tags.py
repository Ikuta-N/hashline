import pytest

from hashline.tags import extract_tags, normalize_tag


class TestNormalizeTag:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("rust", "rust"),
            ("#rust", "rust"),
            ("Rust", "rust"),
            ("  #Rust  ", "rust"),
            ("full-text", "full-text"),
            ("#full-text-", "full-text"),
            ("日本語", "日本語"),
            ("snake_case", "snake_case"),
        ],
    )
    def test_normalizes(self, raw: str, expected: str) -> None:
        assert normalize_tag(raw) == expected

    @pytest.mark.parametrize("raw", ["", "#", "---", "two words", "!!", "#!"])
    def test_rejects_names_no_note_body_could_produce(self, raw: str) -> None:
        with pytest.raises(ValueError):
            normalize_tag(raw)


class TestExtractTags:
    def test_extracts_in_order_of_appearance(self) -> None:
        assert extract_tags("looked at #sqlite and #fts5 today") == ["sqlite", "fts5"]

    def test_tag_at_start_of_body(self) -> None:
        assert extract_tags("#sqlite is fine") == ["sqlite"]

    def test_tag_after_newline(self) -> None:
        assert extract_tags("first line\n#sqlite") == ["sqlite"]

    def test_deduplicates_case_insensitively(self) -> None:
        assert extract_tags("#Rust and #rust and #RUST") == ["rust"]

    def test_extracts_japanese_tags(self) -> None:
        assert extract_tags("bm25 を調べた #検索 #日本語") == ["検索", "日本語"]

    def test_stops_at_trailing_punctuation(self) -> None:
        assert extract_tags("done #sqlite, then #fts5.") == ["sqlite", "fts5"]

    def test_keeps_internal_hyphens(self) -> None:
        assert extract_tags("#full-text search") == ["full-text"]

    def test_follows_japanese_punctuation(self) -> None:
        assert extract_tags("朝のメモ。#日記、あとで見る") == ["日記"]

    def test_follows_a_bracket(self) -> None:
        assert extract_tags("あとで(#todo)") == ["todo"]

    @pytest.mark.parametrize(
        "text",
        [
            "see https://example.com/page#section",
            "see https://example.com/#top",
            "issue tracker#42",
        ],
    )
    def test_ignores_a_hash_glued_to_the_previous_word(self, text: str) -> None:
        assert extract_tags(text) == []

    def test_ignores_markdown_headings(self) -> None:
        assert extract_tags("# Heading\n## Sub heading\ntext") == []

    def test_heading_and_tag_can_coexist(self) -> None:
        assert extract_tags("# Heading\nbody with #sqlite") == ["sqlite"]

    def test_bare_hash_is_not_a_tag(self) -> None:
        assert extract_tags("a # b #- c") == []

    def test_no_tags(self) -> None:
        assert extract_tags("just a plain thought") == []

    def test_empty_body(self) -> None:
        assert extract_tags("") == []
