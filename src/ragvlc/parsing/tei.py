"""Parse GROBID TEI XML into structured header metadata and content units.

This module turns one GROBID TEI document into a `ParsedPaper`: unreliable
header metadata (title/authors/date/venue -- everything except the DOI, which
comes from Crossref instead, see `crossref.py`) plus a flat, document-ordered
list of `Unit`s (abstract / section / caption). It does **not** normalize unit
text -- see `normalize.py` for why that has to stay a separate, frozen step --
and it does no file I/O; `pipeline.py` (a later stage) is what reads TEI files
from disk, calls this module, normalizes each unit, and writes the canonical
output.

TEI namespace: http://www.tei-c.org/ns/1.0

Three things below go beyond the base extraction rules, added after inspecting
real GROBID output from this corpus rather than the TEI spec in the abstract:

1. **Section hierarchy reconstruction.** GROBID emits a flat list of `<div>`
   siblings -- it does not nest subsections inside their parent. We recover an
   approximate hierarchy by pattern-matching each heading's numbering prefix
   (see `_classify_prefix`), and carry the most recent level-1 heading forward
   as `parent_heading` on every unit so `section_type` can be derived from it.
   A heading with no numbering prefix at all defaults to level 1 -- but an
   *unnumbered* heading is also often GROBID mis-parsing a fragment of body
   prose as a `<head>` (a caption label, a stray sentence), and letting that
   become the carried-forward context corrupts every subsection after it. So
   an unnumbered heading is additionally screened as "junk" (see
   `_is_junk_heading`): it still gets `section_level=1` for its own unit, but
   a junk heading does not update `parent_heading` going forward.

   Note this level/junk logic classifies a unit's *own* level independently of
   whether it's eligible to become other units' parent -- level 2/3 headings
   never update `parent_heading` regardless, junk or not (only a *level-1,
   non-junk* heading does).

2. **Merged-heading splitting.** GROBID sometimes concatenates a section head
   with the immediately following subsection head into a single `<head>` (see
   `_try_split_merged_heading`). Splitting these is what makes (1) accurate.

   Both a performed split and a junk-heading classification are recorded as
   plain strings on `ParsedPaper.warnings` rather than printed -- this module
   returns data, it doesn't own terminal output. `parse_corpus.py` (a later
   stage) is what will print them, with whatever paper-level framing it wants.

3. **Character-composition stats.** Broken math-font encoding in the source
   PDF shows up in GROBID's output as Private Use Area characters or the
   Unicode replacement character. We count these per unit (and the non-ASCII
   ratio) so `parse_corpus.py`'s quality gates can flag it -- we do not try to
   repair it, and we deliberately also leave alone the *other*, untagged kind
   of broken math (flattened inline formulas like "A p" for "A_p"): it isn't
   marked with any element we could target, it's unrecoverable, and it's
   harmless downstream since gold spans and chunks are computed against the
   same normalized text either way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from lxml import etree

TEI_NS = "http://www.tei-c.org/ns/1.0"


def _q(tag: str) -> str:
    """Clark-notation-qualify a bare tag name with the TEI namespace."""
    return f"{{{TEI_NS}}}{tag}"


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Header:
    """Header metadata. Everything here is unreliable except `doi` -- GROBID's
    header parsing is inconsistent, especially for venue and author names.
    `crossref.py` fetches authoritative title/authors/year/venue from the DOI;
    these fields exist for a paper with no DOI, and as a sanity check.
    """

    title: str | None
    authors: list[str]
    year: int | None
    venue: str | None
    doi: str | None
    abstract: str | None  # raw (not yet normalized) abstract text


@dataclass
class Unit:
    """One piece of extracted text plus everything pipeline.py needs to place
    and describe it. `raw_text` is deliberately *not* normalized here.
    """

    type: str  # "abstract" | "section" | "caption"
    section_heading: str
    section_index: int  # -1 for the abstract; else 0-based, over kept divs only
    section_level: int | None  # None for the abstract; else 1/2/3/... (see amendment 1)
    heading_unnumbered: bool
    heading_junk: bool  # unnumbered heading screened out of parent-heading tracking
    parent_heading: str | None  # most recent level-1 heading, carried forward
    section_type: str  # "abstract" | keyword-derived | "other"
    raw_text: str
    contains_equation: bool
    n_pua_chars: int
    n_replacement_chars: int
    non_ascii_ratio: float


@dataclass
class ParsedPaper:
    header: Header
    units: list[Unit] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)  # merged-heading splits, junk headings, ...
    paper_id: str | None = None


# --------------------------------------------------------------------------- #
# Inline text extraction (shared by headings, abstract, paragraphs, captions)
# --------------------------------------------------------------------------- #
def _extract_inline_text(elem: etree._Element) -> str:
    """Depth-first text extraction that applies PROMPT_1's inline-element rules.

    This is a hand-rolled alternative to `elem.itertext()`: plain `itertext()`
    can't selectively drop elements while keeping the text that follows them,
    which is exactly what the spec requires for citation markers and formulas
    (drop the element's own content, but keep its *tail* -- the prose that
    continues right after it. That tail text is where artifacts like
    "2024 , which" come from once a `<ref type="bibr">[3]</ref>` is removed;
    `normalize()` cleans that up downstream, not here.

    - `<ref type="bibr">`: drop entirely (own text, not tail).
    - `<formula>`: drop entirely (own text, not tail); see `_contains_equation`
      for flagging the containing unit.
    - everything else (including `<ref type="figure/table/...">`): recurse and
      keep both its text and its tail.
    """
    parts: list[str] = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        local = etree.QName(child).localname
        if local == "ref" and child.get("type") == "bibr":
            pass
        elif local == "formula":
            pass
        else:
            parts.append(_extract_inline_text(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def _clean_ws(text: str) -> str:
    """Collapse whitespace runs and strip -- for headings/metadata, not for
    unit body text. Deliberately not `normalize()`: headings aren't part of
    the offset-sensitive canonical text, so they have no reason to be coupled
    to that frozen contract's punctuation rules.
    """
    return " ".join(text.split())


def _contains_equation(elem: etree._Element) -> bool:
    """Whether `elem` has a `<formula>` descendant anywhere within it."""
    return elem.find(f".//{_q('formula')}") is not None


# --------------------------------------------------------------------------- #
# Character-composition stats (amendment 3)
# --------------------------------------------------------------------------- #
_PUA_LOW, _PUA_HIGH = 0xE000, 0xF8FF
_REPLACEMENT_CHAR = "�"


def _char_composition_stats(text: str) -> tuple[int, int, float]:
    """Return (n_pua_chars, n_replacement_chars, non_ascii_ratio) for `text`.

    Computed on the raw (pre-normalize) unit text: none of `normalize()`'s
    steps touch these code points, and keeping this module independent of
    `normalize` keeps "extract" and "normalize" as separate, separately
    testable stages.
    """
    if not text:
        return 0, 0, 0.0
    n_pua = sum(1 for ch in text if _PUA_LOW <= ord(ch) <= _PUA_HIGH)
    n_replacement = text.count(_REPLACEMENT_CHAR)
    n_non_ascii = sum(1 for ch in text if ord(ch) > 127)
    return n_pua, n_replacement, n_non_ascii / len(text)


# --------------------------------------------------------------------------- #
# Heading classification (amendment 1) and merged-heading splitting (amendment 2)
# --------------------------------------------------------------------------- #
# These are deliberately loose pattern matches over the numbering *conventions*
# observed in this corpus's GROBID output, not a validator of "real" roman
# numerals -- e.g. "IIII." would happily be treated as a level-1 roman
# numeral. Rejecting it would buy nothing: a heading whose numbering doesn't
# look like real prose either way needs human review regardless.
_ROMAN_TOKEN_RE = re.compile(r"^([MDCLXVI]+)\.(?=\s|$)")
_LETTER_TOKEN_RE = re.compile(r"^([A-Z])\.(?=\s|$)")
_LOWER_LETTER_TOKEN_RE = re.compile(r"^([a-z])[:)](?=\s|$)")
_PAREN_DIGIT_RE = re.compile(r"^(\d+)\)(?=\s|$)")
_DOTTED_DECIMAL_RE = re.compile(r"^((?:\d+\.)+)(?=\s|$)")

# For merged-heading splitting: a level-2 marker embedded *later* in the
# string, i.e. preceded by whitespace rather than anchored at position 0
# (which is what _LETTER_TOKEN_RE checks for a heading classified on its own).
_EMBEDDED_LEVEL2_MARKER_RE = re.compile(r"\s([A-Z])\.(?=\s|$)")


def _classify_prefix(token: str) -> tuple[int, bool]:
    """Classify a heading's numbering prefix into (section_level, heading_unnumbered).

    `token` is either a GROBID `n` attribute value (MDPI-style output puts the
    parsed number there, e.g. `n="2.1."`, and leaves the `<head>` text as a
    clean title with no number in it at all) or the heading text itself
    (IEEE-style output keeps the number inline in the text, e.g.
    "II. RELATED WORK"). Callers pass whichever one actually carries the
    number; see `parse_tei`.

    Rules, in order (amendment 1, plus the lowercase sub-subsection addition):
    - "I."                                   -> level 1
    - multi-character roman numeral + "."    -> level 1  (II., VIII., ...)
    - any other single uppercase letter + "." -> level 2  (A., B., C., ...
      including single letters that also happen to be valid roman numerals,
      e.g. "C." -- here it's a letter, not roman 100)
    - lowercase letter + ":" or ")"          -> level 3  (a:, b), ...)
    - "<digits>)"                            -> level 3
    - dotted decimal, e.g. "2.1."            -> level = number of components
    - no match                               -> level 1, heading_unnumbered=True
      (this heading's own level is still 1; whether it's *eligible* to become
      the carried-forward parent_heading is a separate question -- see
      `_is_junk_heading` and its caller)
    """
    m = _ROMAN_TOKEN_RE.match(token)
    if m:
        roman = m.group(1)
        if roman == "I" or len(roman) >= 2:
            return 1, False
        # else: single roman-look-alike letter other than "I" -- fall through
        # to be treated as a level-2 letter marker below.

    if _LETTER_TOKEN_RE.match(token):
        return 2, False

    if _LOWER_LETTER_TOKEN_RE.match(token):
        return 3, False

    if _PAREN_DIGIT_RE.match(token):
        return 3, False

    m = _DOTTED_DECIMAL_RE.match(token)
    if m:
        return m.group(1).count("."), False

    return 1, True


def _try_split_merged_heading(head_text: str) -> tuple[str, str] | None:
    """Detect and split a GROBID-merged "LEVEL1 TITLE LEVEL2 TITLE" heading
    (amendment 2), e.g.:

        "II. RELATED WORK A. VLC APPLICATIONS IN UNDERGROUND MINES"
        -> ("II. RELATED WORK", "A. VLC APPLICATIONS IN UNDERGROUND MINES")

    Only splits on an unambiguous shape: a level-1 numbering prefix at the
    very start of the string, and a level-2 marker (a lone uppercase letter +
    ".", set off by whitespace on both sides) appearing later with real title
    text after it. Returns None -- meaning "classify and use as a single
    heading" -- if either condition isn't clearly met.
    """
    m1 = _ROMAN_TOKEN_RE.match(head_text)
    if not m1:
        return None
    roman = m1.group(1)
    if not (roman == "I" or len(roman) >= 2):
        return None  # a single roman-look-alike letter is a level-2 marker, not level-1

    m2 = _EMBEDDED_LEVEL2_MARKER_RE.search(head_text, m1.end())
    if not m2:
        return None

    split_at = m2.start() + 1  # skip the whitespace the marker regex consumed
    parent = head_text[:split_at].rstrip()
    child = head_text[split_at:].lstrip()
    if not child:
        return None  # marker was at the tail end -- nothing left to own the paragraphs
    return parent, child


_JUNK_LOWERCASE_WORD_RE = re.compile(r"\b[a-z]{4,}\b")
_JUNK_FLOAT_LABEL_RE = re.compile(r"^(?:Algorithm|Figure|Fig\.|Table|Listing)\s*\d", re.IGNORECASE)


def _is_junk_heading(heading: str) -> bool:
    """Screen an *unnumbered* heading for looking like something other than a
    real section title -- a GROBID mis-parse of ordinary prose, a figure/table/
    algorithm label, or MDPI-style trailing boilerplate ("Funding:",
    "Conflicts of Interest:") -- that shouldn't become the carried-forward
    `parent_heading` for subsequent subsections.

    Junk if any of:
    - ends with one of . , ; :               (trailing-punctuation prose/labels)
    - longer than 80 characters               (a heading isn't a sentence)
    - contains a whole word of 4+ lowercase letters, and isn't title-cased
      (real headings are consistently capitalized; a stray lowercase content
      word -- "generation", "interference" -- is a sign of ordinary prose)
    - matches a figure/table/algorithm/listing label followed by a number

    A heading that's plausible (e.g. a bare "CONCLUSION") is not junk: it
    keeps its default level-1 classification *and* remains eligible to update
    `parent_heading`, same as before this screen existed.
    """
    if not heading:
        return False
    if heading[-1] in ".,;:":
        return True
    if len(heading) > 80:
        return True
    if _JUNK_LOWERCASE_WORD_RE.search(heading) and not heading.istitle():
        return True
    if _JUNK_FLOAT_LABEL_RE.match(heading):
        return True
    return False


_SECTION_TYPE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("introduction", ("introduction",)),
    ("related_work", ("related work", "prior work", "background", "literature review")),
    (
        "system_model",
        ("system model", "network model", "channel model", "system architecture", "problem formulation"),
    ),
    ("methods", ("method", "methodology", "proposed", "approach", "algorithm", "implementation")),
    ("results", ("result", "evaluation", "performance", "experiment", "simulation")),
    ("discussion", ("discussion",)),
    ("conclusion", ("conclusion", "summary", "future work")),
)


def _derive_section_type(parent_heading: str | None) -> str:
    """Coarse section_type from `parent_heading` by keyword (case-insensitive,
    substring match). Checked in the fixed order above, so e.g. a heading
    containing both "results" and "discussion" lands on "results" -- first
    match wins, there's no principled way to prefer one over the other from
    the heading text alone.
    """
    if not parent_heading:
        return "other"
    text = parent_heading.lower()
    for section_type, keywords in _SECTION_TYPE_KEYWORDS:
        if any(kw in text for kw in keywords):
            return section_type
    return "other"


# --------------------------------------------------------------------------- #
# Section context: the per-div metadata shared by every unit that div produces
# --------------------------------------------------------------------------- #
@dataclass
class _SectionContext:
    heading: str
    section_index: int
    section_level: int | None
    heading_unnumbered: bool
    heading_junk: bool
    parent_heading: str | None
    section_type: str


_NO_DIV_CONTEXT = _SectionContext(
    heading="(untitled section)",
    section_index=-1,
    section_level=None,
    heading_unnumbered=True,
    heading_junk=False,
    parent_heading=None,
    section_type="other",
)


def _make_unit(unit_type: str, elem: etree._Element, ctx: _SectionContext) -> Unit:
    raw_text = _extract_inline_text(elem)
    n_pua, n_replacement, non_ascii_ratio = _char_composition_stats(raw_text)
    return Unit(
        type=unit_type,
        section_heading=ctx.heading,
        section_index=ctx.section_index,
        section_level=ctx.section_level,
        heading_unnumbered=ctx.heading_unnumbered,
        heading_junk=ctx.heading_junk,
        parent_heading=ctx.parent_heading,
        section_type=ctx.section_type,
        raw_text=raw_text,
        contains_equation=_contains_equation(elem),
        n_pua_chars=n_pua,
        n_replacement_chars=n_replacement,
        non_ascii_ratio=non_ascii_ratio,
    )


# --------------------------------------------------------------------------- #
# Header extraction
# --------------------------------------------------------------------------- #
def _person_name(pers_name: etree._Element) -> str:
    """Join `forename`/`surname` children in document order, skipping
    `roleName` and anything else GROBID tucks into `<persName>` (e.g. "Graduate
    Student Member, IEEE" preceding the actual name) -- header fields are
    unreliable regardless, but there's no reason to also mix in obvious noise.
    """
    parts: list[str] = []
    for child in pers_name:
        local = etree.QName(child).localname
        if local in ("forename", "surname"):
            text = _clean_ws(_extract_inline_text(child))
            if text:
                parts.append(text)
    return " ".join(parts)


def _parse_year(candidate: str) -> int | None:
    """Pull a 4-digit year out of a date string/attribute like "2020",
    "2020-07", or "Jul. 2020". Returns None if nothing looks like a year.
    """
    m = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", candidate)
    return int(m.group(1)) if m else None


def _extract_abstract(root: etree._Element) -> tuple[str, bool] | None:
    """Return (raw_text, contains_equation) for `<profileDesc>/<abstract>`, or
    None if there's no abstract to extract. Multiple `<p>` inside the abstract
    are joined into the text of a single unit (the abstract is one unit, per
    PROMPT_1, however many paragraphs GROBID split it into).
    """
    abstract_elem = root.find(f".//{_q('profileDesc')}/{_q('abstract')}")
    if abstract_elem is None:
        return None
    paragraphs = abstract_elem.findall(f".//{_q('p')}")
    if not paragraphs:
        text = _extract_inline_text(abstract_elem)
        return (text, _contains_equation(abstract_elem)) if text.strip() else None
    combined = "\n".join(_extract_inline_text(p) for p in paragraphs)
    contains_eq = any(_contains_equation(p) for p in paragraphs)
    return combined, contains_eq


def _extract_header(root: etree._Element, abstract: tuple[str, bool] | None) -> Header:
    title: str | None = None
    title_elem = root.find(f".//{_q('titleStmt')}/{_q('title')}")
    if title_elem is not None:
        title = _clean_ws(_extract_inline_text(title_elem)) or None

    authors: list[str] = []
    doi: str | None = None
    venue: str | None = None
    year: int | None = None

    bibl_struct = root.find(f".//{_q('sourceDesc')}/{_q('biblStruct')}")
    if bibl_struct is not None:
        for pers_name in bibl_struct.findall(f".//{_q('analytic')}/{_q('author')}/{_q('persName')}"):
            name = _person_name(pers_name)
            if name:
                authors.append(name)

        doi_elem = bibl_struct.find(f".//{_q('idno')}[@type='DOI']")
        if doi_elem is not None and doi_elem.text:
            doi = doi_elem.text.strip() or None

        venue_elem = bibl_struct.find(f".//{_q('monogr')}/{_q('title')}")
        if venue_elem is not None:
            venue = _clean_ws(_extract_inline_text(venue_elem)) or None

        date_elem = bibl_struct.find(f".//{_q('monogr')}/{_q('imprint')}/{_q('date')}")
        if date_elem is not None:
            year = _parse_year(date_elem.get("when") or date_elem.text or "")

    if year is None:
        # Fall back to GROBID's own header date if the consolidated
        # bibliographic date (above) wasn't there.
        pub_date_elem = root.find(f".//{_q('publicationStmt')}/{_q('date')}")
        if pub_date_elem is not None:
            year = _parse_year(pub_date_elem.get("when") or pub_date_elem.text or "")

    abstract_text = _clean_ws(abstract[0]) if abstract else None

    return Header(title=title, authors=authors, year=year, venue=venue, doi=doi, abstract=abstract_text)


# --------------------------------------------------------------------------- #
# Body walking
# --------------------------------------------------------------------------- #
_EXCLUDED_HEADING_RE = re.compile(r"references|bibliography|acknowledg", re.IGNORECASE)


def _resolve_div_heading(
    head_elem: etree._Element | None,
    current_level1_heading: str | None,
    warnings: list[str],
) -> tuple[str, int, bool, bool, str | None]:
    """Work out (heading, section_level, heading_unnumbered, heading_junk,
    new_current_level1) for one `<div>`, applying amendments 1 and 2.

    `warnings` is a caller-owned accumulator: a performed merged-heading split
    or a junk-heading classification appends a plain description to it,
    instead of printing -- see the module docstring.
    """
    raw_head_text = _extract_inline_text(head_elem) if head_elem is not None else ""
    clean_head_text = _clean_ws(raw_head_text)

    if not clean_head_text:
        return "(untitled section)", 1, True, False, current_level1_heading

    split = _try_split_merged_heading(clean_head_text)
    if split is not None:
        parent_text, child_text = split
        warnings.append(
            f"split merged heading {clean_head_text!r} -> parent={parent_text!r}, child={child_text!r}"
        )
        return child_text, 2, False, False, parent_text

    n_attr = (head_elem.get("n") if head_elem is not None else None) or ""
    level, unnumbered = _classify_prefix(n_attr.strip() or clean_head_text)

    heading_junk = unnumbered and _is_junk_heading(clean_head_text)
    if heading_junk:
        warnings.append(
            f"heading classified as junk, not carried forward as parent_heading: {clean_head_text!r}"
        )

    if level == 1 and not heading_junk:
        new_current = clean_head_text
    else:
        new_current = current_level1_heading
    return clean_head_text, level, unnumbered, heading_junk, new_current


def _extract_body_units(root: etree._Element) -> tuple[list[Unit], list[str]]:
    body = root.find(f".//{_q('text')}/{_q('body')}")
    if body is None:
        return [], []

    units: list[Unit] = []
    warnings: list[str] = []
    current_level1_heading: str | None = None
    current_context: _SectionContext | None = None
    section_index = -1

    for child in body:
        local = etree.QName(child).localname

        if local == "div":
            head_elem = child.find(_q("head"))
            probe_text = _clean_ws(_extract_inline_text(head_elem)) if head_elem is not None else ""
            if child.get("type") == "acknowledgement" or (
                probe_text and _EXCLUDED_HEADING_RE.search(probe_text)
            ):
                continue  # excluded entirely: no unit, no effect on running state

            section_index += 1
            heading, level, unnumbered, heading_junk, current_level1_heading = _resolve_div_heading(
                head_elem, current_level1_heading, warnings
            )
            parent_heading = current_level1_heading
            section_type = _derive_section_type(parent_heading)
            current_context = _SectionContext(
                heading=heading,
                section_index=section_index,
                section_level=level,
                heading_unnumbered=unnumbered,
                heading_junk=heading_junk,
                parent_heading=parent_heading,
                section_type=section_type,
            )

            for p in child.findall(_q("p")):
                units.append(_make_unit("section", p, current_context))

        elif local == "figure":
            ctx = current_context or _NO_DIV_CONTEXT
            for fig_desc in child.findall(_q("figDesc")):
                units.append(_make_unit("caption", fig_desc, ctx))

        # <note> (footnotes) and anything else at the body level: skip.

    return units, warnings


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def parse_tei(xml_bytes: bytes, *, paper_id: str | None = None) -> ParsedPaper:
    """Parse one GROBID TEI document into header metadata and content units.

    `paper_id` is optional; it has no effect on parsing and is only carried
    through onto the returned `ParsedPaper.paper_id`, so a caller collecting
    many of these (e.g. across a batch run) can print each one's `warnings`
    with the right paper attached without keeping a parallel list.
    """
    root = etree.fromstring(xml_bytes)

    abstract = _extract_abstract(root)
    header = _extract_header(root, abstract)

    units: list[Unit] = []
    if abstract is not None:
        text, contains_eq = abstract
        n_pua, n_replacement, non_ascii_ratio = _char_composition_stats(text)
        units.append(
            Unit(
                type="abstract",
                section_heading="Abstract",
                section_index=-1,
                section_level=None,
                heading_unnumbered=False,
                heading_junk=False,
                parent_heading=None,
                section_type="abstract",
                raw_text=text,
                contains_equation=contains_eq,
                n_pua_chars=n_pua,
                n_replacement_chars=n_replacement,
                non_ascii_ratio=non_ascii_ratio,
            )
        )

    body_units, warnings = _extract_body_units(root)
    units.extend(body_units)

    return ParsedPaper(header=header, units=units, warnings=warnings, paper_id=paper_id)