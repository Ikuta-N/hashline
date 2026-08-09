"""Tests for the BibTeX parser."""

from pathlib import Path

import pytest

from hashline.bib import citekey_tag, clean_value, parse_bibtex

FIXTURE = Path(__file__).parent / "fixtures" / "bib" / "library.bib"


@pytest.fixture
def library_text() -> str:
    return FIXTURE.read_text(encoding="utf-8")


class TestParseBibtex:
    def test_parses_an_article(self, library_text: str) -> None:
        entries, _ = parse_bibtex(library_text)
        smith = next(e for e in entries if e.citekey == "smith2020")
        assert smith.entry_type == "article"
        assert smith.title == "A Survey of Trigram Indexing"
        assert smith.author == "Smith, John and Doe, Jane"
        assert smith.year == "2020"
        assert smith.doi == "10.1234/synth.2020.001"
        assert smith.tag == "smith2020"

    def test_parses_a_book(self, library_text: str) -> None:
        entries, _ = parse_bibtex(library_text)
        tanaka = next(e for e in entries if e.citekey == "tanaka2019")
        assert tanaka.entry_type == "book"
        assert tanaka.title == "Introduction to Full-Text Search"
        assert tanaka.author == "Tanaka, Yuki"
        assert tanaka.year == "2019"

    def test_parses_inproceedings(self, library_text: str) -> None:
        entries, _ = parse_bibtex(library_text)
        garcia = next(e for e in entries if e.citekey == "garcia_sqlite_2021")
        assert garcia.entry_type == "inproceedings"
        assert garcia.tag == "garcia_sqlite_2021"
        assert garcia.year == "2021"

    def test_skips_comment_and_string(self, library_text: str) -> None:
        entries, _ = parse_bibtex(library_text)
        types = {e.entry_type for e in entries}
        assert "comment" not in types
        assert "string" not in types

    def test_preserves_nested_braces_in_title(self, library_text: str) -> None:
        entries, _ = parse_bibtex(library_text)
        nested = next(e for e in entries if e.citekey == "nested2023")
        assert "Braces {Within Braces}" in (nested.title or "")

    def test_handles_quoted_values(self, library_text: str) -> None:
        entries, _ = parse_bibtex(library_text)
        quoted = next(e for e in entries if e.citekey == "quoted2024")
        assert quoted.title == "Quoted Values Instead of Braces"
        assert quoted.author == "Lee, Soo-Hyun"

    def test_preserves_latex_accents(self, library_text: str) -> None:
        entries, _ = parse_bibtex(library_text)
        mueller = next(e for e in entries if e.citekey == "mueller2022")
        assert mueller.author is not None
        # LaTeX escapes are NOT expanded: {\"u} stays as written.
        assert "ller" in mueller.author
        assert "\\" in mueller.author

    def test_sanitises_citekey_into_tag(self, library_text: str) -> None:
        entries, _ = parse_bibtex(library_text)
        bad = next(e for e in entries if e.citekey == "Bad:Key!Here")
        assert bad.tag == "bad-key-here"

    def test_malformed_entry_is_reported_and_skipped(self, library_text: str) -> None:
        entries, problems = parse_bibtex(library_text)
        citekeys = {e.citekey for e in entries}
        assert "malformed_entry" not in citekeys
        assert len(problems) >= 1

    def test_neighbours_survive_a_malformed_entry(self, library_text: str) -> None:
        entries, _ = parse_bibtex(library_text)
        # At least 7 well-formed entries should parse
        assert len(entries) >= 7

    def test_stores_raw_entry_text(self, library_text: str) -> None:
        entries, _ = parse_bibtex(library_text)
        smith = next(e for e in entries if e.citekey == "smith2020")
        assert "@article{smith2020," in smith.raw

    def test_empty_file_returns_nothing(self) -> None:
        entries, problems = parse_bibtex("")
        assert entries == []
        assert problems == []

    def test_entry_type_is_lowercased(self) -> None:
        text = "@Article{Test2024, title = {Test}, year = {2024}}"
        entries, _ = parse_bibtex(text)
        assert entries[0].entry_type == "article"

    def test_skips_a_comment_entry_that_has_a_trailing_comma(self) -> None:
        """A @comment or @string block only reaches the type filter if it
        happens to be shaped like ``@type{token,`` -- most real exports (see
        the fixture) are not, and never even match as a raw entry. This
        covers the filter itself for the rare export that is shaped that way.
        """
        entries, problems = parse_bibtex("@comment{c1, ignored text}")
        assert entries == []
        assert problems == []

    def test_an_unconvertible_citekey_is_reported_as_a_problem(self) -> None:
        entries, problems = parse_bibtex("@article{:::, title = {X}}")
        assert entries == []
        assert len(problems) == 1
        assert ":::" in problems[0]

    def test_a_field_with_no_name_stops_that_entrys_field_parsing(self) -> None:
        """A leading ``=`` before any field name reads as an empty name,
        which halts parsing of the remaining fields in that entry."""
        entries, _ = parse_bibtex("@article{key8, = {x}, title = {After Bad Field}}")
        assert entries[0].title is None

    def test_a_bare_token_with_no_assignment_stops_field_parsing(self) -> None:
        entries, _ = parse_bibtex("@article{key6, year}")
        assert entries[0].year is None

    def test_a_field_missing_its_value_stops_field_parsing(self) -> None:
        entries, _ = parse_bibtex("@article{key7, title =}")
        assert entries[0].title is None

    def test_bare_unquoted_values_are_read_as_tokens(self) -> None:
        """Not every export braces or quotes every field; a bare token like
        an unquoted year must still be read as a value."""
        entries, _ = parse_bibtex(
            "@article{key9, title = {Bare Value Test}, year = 2020}"
        )
        assert entries[0].title == "Bare Value Test"
        assert entries[0].year == "2020"


class TestCitekeyTag:
    def test_lowercases_and_replaces_colons(self) -> None:
        assert citekey_tag("Smith:2020a") == "smith-2020a"

    def test_keeps_underscores(self) -> None:
        assert citekey_tag("smith_title_2020") == "smith_title_2020"

    def test_returns_none_for_empty_result(self) -> None:
        assert citekey_tag(":::") is None

    def test_strips_leading_trailing_hyphens(self) -> None:
        result = citekey_tag(":test:")
        assert result is not None
        assert not result.startswith("-")
        assert not result.endswith("-")

    def test_simple_key(self) -> None:
        assert citekey_tag("smith2020") == "smith2020"

    def test_returns_none_when_casefold_produces_a_non_word_character(
        self,
    ) -> None:
        """``\\N{LATIN SMALL LETTER J WITH CARON}`` survives ``.lower()`` as a
        single word character, so ``_BAD_TAG_CHARS`` leaves it alone -- but
        ``normalize_tag``'s ``.casefold()`` decomposes it into ``j`` plus a
        combining caron, which is not a word character. ``citekey_tag`` must
        catch the resulting ``ValueError`` rather than let it escape.
        """
        assert citekey_tag("smithǰ2020") is None


class TestCleanValue:
    def test_strips_outer_braces(self) -> None:
        assert clean_value("{Hello World}") == "Hello World"

    def test_strips_outer_quotes(self) -> None:
        assert clean_value('"Hello World"') == "Hello World"

    def test_collapses_whitespace(self) -> None:
        assert clean_value("{Hello   World\n  Again}") == "Hello World Again"

    def test_bare_value(self) -> None:
        assert clean_value("2024") == "2024"

    def test_preserves_inner_braces(self) -> None:
        assert clean_value("{Nested {Title}}") == "Nested {Title}"
