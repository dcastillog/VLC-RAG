"""Build `CORPUS.md` -- the CC-BY attribution list for the 36-paper corpus.

One IEEE-style numbered entry per paper in `data/manifest.csv`, sorted by
first author surname, built from the cached Crossref metadata in
`data/crossref/{paper_id}.json` (richer than the manifest -- full author
list, container title, volume/issue/pages) with `data/manifest.csv` as the
fallback for anything Crossref lacks. A field present in neither source is
printed as `n/a` rather than dropped, so a thin entry still has the right
shape and is visibly incomplete rather than silently short a field.

`data/crossref/` can hold cache entries for papers outside the 36-paper
corpus (fixtures, retired candidates); entries are only ever looked up by a
`paper_id` already read from the manifest, so an extra file there is simply
never read.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

NA = "n/a"


# --------------------------------------------------------------------------- #
# Small formatting helpers
# --------------------------------------------------------------------------- #
def first_of(values: object) -> str | None:
    """First element of a Crossref list-valued field (title, container-title
    are always arrays, even when there is exactly one value)."""
    if isinstance(values, list) and values:
        return str(values[0])
    return None


def initials(given_name: str) -> str:
    """'Yeong Min' -> 'Y. M.'; 'Chih-Lin' -> 'C.-L.'; 'Md.' -> 'M.'

    IEEE style reduces every given name to initials, so a hyphenated given
    name ('Chih-Lin') keeps its hyphen between initials rather than
    collapsing to the first letter of the whole token; an already-abbreviated
    token ('Md.') just loses its own trailing period and gets one back.
    """
    rendered_tokens = []
    for token in given_name.split():
        sub_initials = [f"{part[0].upper()}." for part in token.split("-") if part]
        if sub_initials:
            rendered_tokens.append("-".join(sub_initials))
    return " ".join(rendered_tokens)


def format_author(author: dict) -> str | None:
    family = (author.get("family") or "").strip()
    if not family:
        return None
    given = (author.get("given") or "").strip()
    return f"{initials(given)} {family}".strip()


def format_authors(authors: list[dict]) -> str:
    """IEEE-style author list. Every author is kept -- this file's entire
    purpose is attribution, so truncating a long author list with "et al."
    would undermine the one thing it exists to do."""
    names = [name for name in (format_author(a) for a in authors) if name]
    if not names:
        return NA
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def format_pages(page: str | None) -> str:
    if not page or not page.strip():
        return f"pp. {NA}"
    page = page.strip()
    if "-" in page:
        start, _, end = page.partition("-")
        start, end = start.strip(), end.strip()
        if start and end:
            return f"pp. {start}–{end}"  # en dash, IEEE style
    # MDPI-style journals give a single article number, not a page range.
    return f"p. {page}"


def format_licence(licence: str | None) -> str:
    """'CC-BY-4.0' -> 'CC BY 4.0'."""
    if not licence:
        return NA
    match = re.match(r"^CC-?BY-?(\d+\.\d+)$", licence.strip(), re.IGNORECASE)
    return f"CC BY {match.group(1)}" if match else licence.strip()


def year_from_crossref(cr: dict) -> str | None:
    """Crossref records a publication date under several possible keys
    depending on how/when the work was deposited; try the ones that mean
    "this is when the article appeared" in order of preference."""
    for key in ("published", "published-print", "published-online", "issued", "created"):
        field = cr.get(key)
        try:
            return str(field["date-parts"][0][0])
        except (KeyError, IndexError, TypeError):
            continue
    return None


def sort_key(surname: str) -> str:
    """Accent-insensitive casefold, so e.g. 'Íñiguez' sorts under I, not after
    every plain-ASCII name (Unicode codepoint order would put it last)."""
    decomposed = unicodedata.normalize("NFKD", surname)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.casefold()


# --------------------------------------------------------------------------- #
@dataclass
class Entry:
    paper_id: str
    authors: str
    title: str
    venue: str
    volume: str
    issue: str
    pages: str
    year: str
    doi: str
    licence: str
    sort_key: str


def load_crossref(crossref_path: Path) -> dict:
    """The cached Crossref `work` object for one paper, or `{}` if there is
    no cache file or it fails to parse -- callers then fall back to the
    manifest for every field, exactly as if Crossref had none of them."""
    if not crossref_path.is_file():
        return {}
    try:
        doc = json.loads(crossref_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return doc.get("crossref") or {}


def build_entry(manifest_row: dict, crossref_dir: Path) -> Entry:
    paper_id = manifest_row["paper_id"]
    cr = load_crossref(crossref_dir / f"{paper_id}.json")

    cr_authors = cr.get("author") or []
    first_surname = (cr_authors[0].get("family") or "").strip() if cr_authors else ""

    return Entry(
        paper_id=paper_id,
        authors=format_authors(cr_authors),
        title=first_of(cr.get("title")) or manifest_row.get("title") or NA,
        venue=first_of(cr.get("container-title")) or manifest_row.get("venue") or NA,
        volume=str(cr.get("volume")) if cr.get("volume") else NA,
        issue=str(cr.get("issue")) if cr.get("issue") else NA,
        pages=format_pages(cr.get("page")),
        year=year_from_crossref(cr) or manifest_row.get("year") or NA,
        doi=cr.get("DOI") or manifest_row.get("doi") or NA,
        licence=format_licence(manifest_row.get("licence")),
        # A surname Crossref never gave us sorts after every real one rather
        # than before (empty string would sort first and look like a name).
        sort_key=sort_key(first_surname) if first_surname else "￿",
    )


def render_entry(index: int, entry: Entry) -> str:
    return (
        f"[{index}] {entry.authors}, \"{entry.title},\" *{entry.venue}*, "
        f"vol. {entry.volume}, no. {entry.issue}, {entry.pages}, {entry.year}, "
        f"doi: {entry.doi}. Licensed under {entry.licence}. `{entry.paper_id}`"
    )


HEADER = """\
# Corpus

All {n} papers in this corpus are open access, published under Creative
Commons licences (CC BY 4.0). This file exists to provide the attribution
CC BY requires: author, title, and publication details for every paper, each
traceable to its parsed text in `data/normalized/` by the trailing `paper_id`.
**The PDFs themselves are not redistributed in this repository** -- fetch
them from the publisher via the DOI in each entry.

Generated by `scripts/build_corpus_md.py` from `data/manifest.csv` and the
cached Crossref metadata in `data/crossref/`. A field neither source has is
printed as `n/a` rather than silently dropped -- do not hand-edit this file;
re-run the script instead.
"""


def render_corpus_md(entries: list[Entry]) -> str:
    ordered = sorted(entries, key=lambda e: e.sort_key)
    lines = [HEADER.format(n=len(entries))]
    lines += [render_entry(i, entry) for i, entry in enumerate(ordered, start=1)]
    return "\n".join(lines) + "\n"


def missing_fields(entry: Entry) -> list[str]:
    """Field names on `entry` that ended up `n/a` (or, for `pages`, ended up
    with an `n/a` article/page number) -- neither Crossref nor the manifest
    had them."""
    plain = {
        "authors": entry.authors, "title": entry.title, "venue": entry.venue,
        "volume": entry.volume, "issue": entry.issue, "year": entry.year, "doi": entry.doi,
    }
    found_missing = [name for name, value in plain.items() if value == NA]
    if entry.pages.endswith(NA):
        found_missing.append("pages")
    return found_missing
