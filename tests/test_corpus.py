"""Tests for ragvlc.corpus -- the CORPUS.md (CC-BY attribution) builder.

Pure functions, hand-constructed fixtures; no dependency on the real
data/manifest.csv or data/crossref/ so these run the same in CI as locally.
"""

from __future__ import annotations

from pathlib import Path

from ragvlc.corpus import (
    NA,
    Entry,
    build_entry,
    format_author,
    format_authors,
    format_licence,
    format_pages,
    initials,
    load_crossref,
    missing_fields,
    render_corpus_md,
    render_entry,
    sort_key,
    year_from_crossref,
)


def _entry(paper_id: str, sort_key_value: str) -> Entry:
    return Entry(
        paper_id=paper_id, authors="A. Author", title="T", venue="V",
        volume="1", issue="1", pages="p. 1", year="2020", doi="10.1/x",
        licence="CC BY 4.0", sort_key=sort_key_value,
    )


# --------------------------------------------------------------------------- #
# Author formatting
# --------------------------------------------------------------------------- #
def test_initials_handles_multi_word_hyphenated_and_pre_abbreviated_names():
    assert initials("Yeong Min") == "Y. M."
    assert initials("Chih-Lin") == "C.-L."
    assert initials("Md.") == "M."
    assert initials("") == ""


def test_format_author_drops_entries_with_no_family_name():
    assert format_author({"given": "Ann", "family": "Author"}) == "A. Author"
    assert format_author({"given": "Ann", "family": ""}) is None
    assert format_author({"given": "Ann"}) is None


def test_format_authors_joins_with_ieee_style_oxford_comma():
    a = [{"given": "A", "family": "One"}]
    b = a + [{"given": "B", "family": "Two"}]
    c = b + [{"given": "C", "family": "Three"}]
    assert format_authors(a) == "A. One"
    assert format_authors(b) == "A. One and B. Two"
    assert format_authors(c) == "A. One, B. Two, and C. Three"
    assert format_authors([]) == NA


def test_format_authors_keeps_every_author_no_et_al_truncation():
    many = [{"given": str(i), "family": f"Author{i}"} for i in range(9)]
    rendered = format_authors(many)
    assert rendered.count("Author") == 9
    assert "et al" not in rendered.lower()


# --------------------------------------------------------------------------- #
# Other fields
# --------------------------------------------------------------------------- #
def test_format_pages_range_uses_en_dash():
    assert format_pages("1234-1245") == "pp. 1234–1245"


def test_format_pages_single_article_number():
    assert format_pages("2514") == "p. 2514"


def test_format_pages_missing():
    assert format_pages(None) == f"pp. {NA}"
    assert format_pages("") == f"pp. {NA}"


def test_format_licence_normalizes_manifest_style():
    assert format_licence("CC-BY-4.0") == "CC BY 4.0"
    assert format_licence(None) == NA


def test_year_from_crossref_prefers_published_over_created():
    cr = {"created": {"date-parts": [[1999]]}, "published": {"date-parts": [[2021]]}}
    assert year_from_crossref(cr) == "2021"


def test_year_from_crossref_falls_back_through_the_key_order():
    cr = {"issued": {"date-parts": [[2020]]}}
    assert year_from_crossref(cr) == "2020"
    assert year_from_crossref({}) is None


def test_sort_key_is_accent_insensitive():
    assert sort_key("Íñiguez") < sort_key("Zeta")
    assert sort_key("bouclé") == sort_key("BOUCLE")


# --------------------------------------------------------------------------- #
# load_crossref
# --------------------------------------------------------------------------- #
def test_load_crossref_missing_file_returns_empty_dict(tmp_path: Path):
    assert load_crossref(tmp_path / "nope.json") == {}


def test_load_crossref_malformed_json_returns_empty_dict(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_crossref(bad) == {}


def test_load_crossref_unwraps_the_crossref_key(tmp_path: Path):
    good = tmp_path / "p1.json"
    good.write_text('{"doi": "10.1/x", "crossref": {"volume": "3"}}', encoding="utf-8")
    assert load_crossref(good) == {"volume": "3"}


# --------------------------------------------------------------------------- #
# build_entry: manifest fallback + n/a marking
# --------------------------------------------------------------------------- #
def test_build_entry_uses_crossref_when_present(tmp_path: Path):
    crossref_dir = tmp_path
    (crossref_dir / "p1.json").write_text(
        """{"doi": "10.1/x", "crossref": {
            "author": [{"given": "Ann", "family": "Author"}],
            "title": ["Real Title"], "container-title": ["Real Venue"],
            "volume": "5", "issue": "2", "page": "10-20",
            "published": {"date-parts": [[2022]]}, "DOI": "10.1/x"
        }}""",
        encoding="utf-8",
    )
    manifest_row = {
        "paper_id": "p1", "doi": "10.1/x", "title": "Manifest Title",
        "year": "2019", "venue": "Manifest Venue", "licence": "CC-BY-4.0",
    }
    entry = build_entry(manifest_row, crossref_dir)
    assert entry.authors == "A. Author"
    assert entry.title == "Real Title"  # Crossref wins over manifest
    assert entry.venue == "Real Venue"
    assert entry.volume == "5" and entry.issue == "2"
    assert entry.pages == "pp. 10–20"
    assert entry.year == "2022"
    assert missing_fields(entry) == []


def test_build_entry_falls_back_to_manifest_when_crossref_is_missing(tmp_path: Path):
    manifest_row = {
        "paper_id": "p2", "doi": "10.1/y", "title": "Manifest Title",
        "year": "2019", "venue": "Manifest Venue", "licence": "CC-BY-4.0",
    }
    entry = build_entry(manifest_row, tmp_path)  # no p2.json in tmp_path
    assert entry.authors == NA  # manifest has no author list to fall back to
    assert entry.title == "Manifest Title"
    assert entry.venue == "Manifest Venue"
    assert entry.doi == "10.1/y"
    assert entry.year == "2019"
    assert set(missing_fields(entry)) == {"authors", "volume", "issue", "pages"}


def test_build_entry_marks_absent_fields_na_not_missing(tmp_path: Path):
    manifest_row = {"paper_id": "p3", "doi": None, "title": None, "year": None, "venue": None, "licence": None}
    entry = build_entry(manifest_row, tmp_path)
    assert entry.title == NA and entry.venue == NA and entry.doi == NA and entry.licence == NA
    # every field renders, nothing is silently dropped
    rendered = render_entry(1, entry)
    assert NA in rendered
    assert "p3" in rendered


# --------------------------------------------------------------------------- #
# Whole-file rendering
# --------------------------------------------------------------------------- #
def test_render_corpus_md_sorts_by_first_author_surname_and_numbers_sequentially():
    zed = _entry("zed-paper", "zeta")
    alpha = _entry("alpha-paper", "alpha")
    beta = _entry("beta-paper", "beta")
    text = render_corpus_md([zed, alpha, beta])
    assert text.index("`alpha-paper`") < text.index("`beta-paper`") < text.index("`zed-paper`")
    assert "[1]" in text and "[2]" in text and "[3]" in text


def test_render_corpus_md_header_mentions_pdfs_not_redistributed_and_licence():
    text = render_corpus_md([])
    assert "not redistributed" in text
    assert "CC BY 4.0" in text
    assert "0 papers" in text


def test_paper_id_appears_as_a_trailing_code_span():
    entry = build_entry(
        {"paper_id": "my-paper-id", "title": "T", "venue": "V", "year": "2020", "doi": "d", "licence": "CC-BY-4.0"},
        Path("/nonexistent"),
    )
    rendered = render_entry(1, entry)
    assert rendered.rstrip().endswith("`my-paper-id`")
