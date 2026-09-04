"""Orchestration: PDF -> TEI (via GROBID) -> structured units (via `tei.py`)
-> authoritative metadata (via `crossref.py`) -> the canonical normalized
output that everything downstream of this stage is built on.

This is where PROMPT_1's assembly rule lives -- the one part of this whole
stage that must never silently change once an evaluation set has been
annotated against it (see `normalize.py`'s module docstring). The rule,
exactly:

1. Extract units in document order: the abstract, then every "section" unit
   in its document order, then every "caption" unit in the order its figure
   appears. `tei.py`'s `ParsedPaper.units` is already internally ordered
   *within* each type; `_ordered_units` below does the stable reorder into
   that specific three-group shape -- it is not a no-op in general, even
   though in this corpus GROBID happens to already cluster all `<figure>`s
   after the body's last `<div>` (see `tei.py`'s docstring), which would make
   it look like one.
2. Normalize each unit's text independently (`normalize.normalize`).
3. Drop units whose normalized text is under `min_unit_chars` (parsing debris).
4. Join the survivors with exactly `"\\n\\n"` (`unit_separator` in config) to
   form the canonical full text.
5. Record `char_start`/`char_end` for each surviving unit *during* that join.

`assemble_paper` (step 1-5) asserts, before ever returning, that for every
kept unit `full_text[char_start:char_end] == normalize(unit.raw_text)` --
that assertion is the entire point of this module. A silently wrong offset
here wouldn't error; it would quietly invalidate every hand-annotated
evaluation-set span built against this output, and the failure would only
ever surface, much later, as inexplicably mediocre retrieval scores. So this
raises `OffsetIntegrityError` instead of returning (and `process_paper` never
reaches the point of writing files) if the check ever fails.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ragvlc.config import get_experiment, get_paths
from ragvlc.parsing import crossref, tei
from ragvlc.parsing.grobid import fetch_tei, slugify_paper_id
from ragvlc.parsing.normalize import normalize


class OffsetIntegrityError(RuntimeError):
    """Raised when a unit's recorded (char_start, char_end) does not slice
    `full_text` back to exactly its normalized text. Should never happen if
    `assemble_paper` is correct; exists so a bug here fails loudly rather than
    silently corrupting the evaluation set's offsets.
    """


# --------------------------------------------------------------------------- #
# Assembly (steps 1-5 above) -- pure, no I/O, directly testable
# --------------------------------------------------------------------------- #
@dataclass
class AssembledUnit:
    """One surviving unit, with its offsets into `AssembledPaper.full_text`.
    Field order here is also the field order in the written JSON (see
    `process_paper`), via `dataclasses.asdict`.
    """

    unit_id: str
    type: str
    section_heading: str
    section_index: int
    section_level: int | None
    heading_unnumbered: bool
    heading_junk: bool
    parent_heading: str | None
    section_type: str
    char_start: int
    char_end: int
    contains_equation: bool
    n_pua_chars: int
    n_replacement_chars: int
    non_ascii_ratio: float


@dataclass
class AssembledPaper:
    full_text: str
    units: list[AssembledUnit] = field(default_factory=list)
    n_dropped_units: int = 0  # dropped for normalized length < min_unit_chars


def _ordered_units(units: list[tei.Unit]) -> list[tei.Unit]:
    """Stable reorder into (abstract, sections..., captions...) -- the exact
    grouping PROMPT_1 specifies for assembly. A plain filter-per-type
    preserves each group's own relative (document) order, so this is the
    whole implementation of "document order" as that spec phrase is defined.
    """
    return (
        [u for u in units if u.type == "abstract"]
        + [u for u in units if u.type == "section"]
        + [u for u in units if u.type == "caption"]
    )


def assemble_paper(units: list[tei.Unit], *, min_unit_chars: int, unit_separator: str = "\n\n") -> AssembledPaper:
    """Run the assembly rule over one paper's extracted units.

    Pure function: no file I/O, no config lookups beyond the two thresholds
    passed in, so it can be exercised directly against hand-built `tei.Unit`
    lists without touching GROBID, Crossref, or the filesystem.

    Raises `OffsetIntegrityError` (and returns nothing) if any surviving
    unit's offsets don't slice `full_text` back to its own normalized text.
    """
    kept: list[tuple[tei.Unit, str]] = []
    n_dropped = 0
    for unit in _ordered_units(units):
        normalized_text = normalize(unit.raw_text)
        if len(normalized_text) < min_unit_chars:
            n_dropped += 1
            continue
        kept.append((unit, normalized_text))

    parts: list[str] = []
    assembled: list[AssembledUnit] = []
    offset = 0
    for i, (unit, normalized_text) in enumerate(kept):
        if i > 0:
            parts.append(unit_separator)
            offset += len(unit_separator)
        char_start = offset
        parts.append(normalized_text)
        offset += len(normalized_text)
        char_end = offset

        assembled.append(
            AssembledUnit(
                unit_id=f"u{i:03d}",
                type=unit.type,
                section_heading=unit.section_heading,
                section_index=unit.section_index,
                section_level=unit.section_level,
                heading_unnumbered=unit.heading_unnumbered,
                heading_junk=unit.heading_junk,
                parent_heading=unit.parent_heading,
                section_type=unit.section_type,
                char_start=char_start,
                char_end=char_end,
                contains_equation=unit.contains_equation,
                n_pua_chars=unit.n_pua_chars,
                n_replacement_chars=unit.n_replacement_chars,
                non_ascii_ratio=unit.non_ascii_ratio,
            )
        )

    full_text = "".join(parts)

    # The offset-integrity contract, checked before this is ever handed back
    # to a caller (let alone written to disk).
    for assembled_unit, (_, normalized_text) in zip(assembled, kept):
        actual = full_text[assembled_unit.char_start : assembled_unit.char_end]
        if actual != normalized_text:
            raise OffsetIntegrityError(
                f"offset mismatch for {assembled_unit.unit_id} "
                f"(type={assembled_unit.type!r}, section_heading={assembled_unit.section_heading!r}): "
                f"full_text[{assembled_unit.char_start}:{assembled_unit.char_end}] "
                f"!= normalize(unit.raw_text)"
            )

    return AssembledPaper(full_text=full_text, units=assembled, n_dropped_units=n_dropped)


# --------------------------------------------------------------------------- #
# Full per-paper orchestration
# --------------------------------------------------------------------------- #
@dataclass
class ProcessedPaper:
    """Everything `parse_corpus.py` needs to report on one paper and build
    `manifest.csv`. `warnings` aggregates `tei.py`'s (merged-heading splits,
    junk headings) and `crossref.py`'s (missing/degraded metadata, licence
    issues) so a caller only has to look in one place.
    """

    paper_id: str
    doi: str | None
    title: str | None
    authors: list[str]
    year: int | None
    venue: str | None
    licence: str
    source_pdf: str
    n_chars: int
    text_sha256: str
    n_units: int
    n_dropped_units: int
    units: list[AssembledUnit]
    txt_path: Path
    json_path: Path
    warnings: list[str]


def process_paper(pdf_path: Path, *, force: bool = False) -> ProcessedPaper:
    """Run the full stage-1 pipeline for one PDF and write its two canonical
    output files: `data/normalized/{paper_id}.txt` and `.json`.

    `force` is passed through to both `grobid.fetch_tei` and
    `crossref.fetch_metadata`, so a full re-run can bypass both caches.

    Title/authors/year/venue prefer Crossref's authoritative values
    (`crossref.PaperMetadata`) and fall back to GROBID's own unreliable header
    extraction only where Crossref has nothing (no DOI, a 404, ...) -- some
    signal for a paper's identity is better than none, and the corresponding
    warning (already on `parsed.warnings`/`metadata.warnings`) makes clear
    that fallback happened.
    """
    paths = get_paths()
    parsing_cfg = get_experiment().parsing

    paper_id = slugify_paper_id(pdf_path.name)
    tei_path = fetch_tei(pdf_path, force=force)
    parsed = tei.parse_tei(tei_path.read_bytes(), paper_id=paper_id)

    doi = crossref.resolve_doi(paper_id, parsed.header.doi)
    metadata = crossref.fetch_metadata(paper_id, doi, force=force)

    assembled = assemble_paper(
        parsed.units,
        min_unit_chars=parsing_cfg.min_unit_chars,
        unit_separator=parsing_cfg.unit_separator,
    )

    title = metadata.title or parsed.header.title
    authors = metadata.authors or parsed.header.authors
    year = metadata.year or parsed.header.year
    venue = metadata.venue or parsed.header.venue

    # SHA-256 of the exact bytes written to the .txt file below -- the
    # tripwire `scripts/verify_corpus.py` checks before every annotation
    # session. Once a gold span is recorded as an offset into this file, any
    # change to it (even whitespace) silently invalidates that span; this is
    # what turns "silently" into "verify_corpus.py fails loudly".
    text_sha256 = hashlib.sha256(assembled.full_text.encode("utf-8")).hexdigest()

    document = {
        "paper_id": paper_id,
        "doi": metadata.doi,
        "title": title,
        "authors": authors,
        "year": year,
        "venue": venue,
        "licence": metadata.licence,
        "source_pdf": pdf_path.name,
        "n_chars": len(assembled.full_text),
        "text_sha256": text_sha256,
        "units": [asdict(unit) for unit in assembled.units],
    }

    paths.normalized.mkdir(parents=True, exist_ok=True)
    txt_path = paths.normalized / f"{paper_id}.txt"
    json_path = paths.normalized / f"{paper_id}.json"
    # newline="" disables any newline translation on write, so the bytes on
    # disk are exactly `full_text.encode("utf-8")` -- what text_sha256 above
    # is computed from -- on every platform, not just the one this happens to
    # run on today.
    txt_path.write_text(assembled.full_text, encoding="utf-8", newline="")
    json_path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return ProcessedPaper(
        paper_id=paper_id,
        doi=metadata.doi,
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        licence=metadata.licence,
        source_pdf=pdf_path.name,
        n_chars=len(assembled.full_text),
        text_sha256=text_sha256,
        n_units=len(assembled.units),
        n_dropped_units=assembled.n_dropped_units,
        units=assembled.units,
        txt_path=txt_path,
        json_path=json_path,
        warnings=[*parsed.warnings, *metadata.warnings],
    )