"""CLI: run the stage-1 parsing pipeline over the corpus (or one paper),
regenerate ``data/manifest.csv``, and print the quality-gate report this
stage's acceptance criteria are checked against.

    uv run python scripts/parse_corpus.py [--force] [--only PAPER_ID]

**Fail-soft.** One bad PDF must not abort a 36-paper run: each paper's
``pipeline.process_paper`` call is wrapped individually, a failure is
recorded and printed immediately, and processing continues. Every failure is
listed again at the end and the process exits non-zero if there were any.

**Slug collisions are checked before any processing at all.** Two filenames
differing only in punctuation (e.g. ``kim-2023.pdf`` / ``kim_2023.pdf``)
slugify to the same ``paper_id`` -- the second would silently read (and
overwrite) the first's cached TEI/Crossref/normalized output. That's a
corpus-setup problem, not a per-paper one, so it aborts the whole run rather
than being merely flagged.

This script owns all terminal output for the pipeline: ``tei.py`` and
``crossref.py`` only return warnings (see their module docstrings), so the
junk-heading list, merged-heading splits, and every other diagnostic printed
below all come from ``ProcessedPaper.warnings``, gathered here.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ragvlc.config import QualityGates, get_experiment, get_paths
from ragvlc.parsing.grobid import slugify_paper_id
from ragvlc.parsing.pipeline import ProcessedPaper, process_paper

_MANIFEST_FIELDS = ["paper_id", "doi", "title", "year", "venue", "licence", "n_chars"]

# The heading-structure table's "merged-splits" column has no dedicated field
# to count -- tei.py records a split as a plain string on ParsedPaper.warnings
# (see its module docstring on why: library code returns, it doesn't print,
# and a plain string list is what it returns). Matching this fixed prefix is
# the only way to pick those back out; if tei.py's message wording ever
# changes, this needs to change with it.
_SPLIT_WARNING_PREFIX = "split merged heading"


@dataclass
class Failure:
    paper_id: str
    pdf_path: Path
    error: BaseException


# --------------------------------------------------------------------------- #
# Discovery and the slug-collision guard
# --------------------------------------------------------------------------- #
def _discover_pdfs(pdfs_dir: Path) -> list[Path]:
    return sorted(pdfs_dir.glob("*.pdf"))


def _paper_id_map(pdf_paths: list[Path]) -> dict[str, Path]:
    """Map paper_id -> pdf_path, aborting the whole run if two filenames
    collide. Must be called with *every* discovered PDF, before any
    ``--only`` filtering -- a collision is a corpus problem regardless of
    which paper this invocation is actually processing.
    """
    by_id: dict[str, list[Path]] = defaultdict(list)
    for pdf_path in pdf_paths:
        by_id[slugify_paper_id(pdf_path.name)].append(pdf_path)

    collisions = {paper_id: paths for paper_id, paths in by_id.items() if len(paths) > 1}
    if collisions:
        print("parse_corpus.py: ABORTING -- filename slug collisions detected:", file=sys.stderr)
        for paper_id, paths in collisions.items():
            print(f"  {paper_id!r}: {', '.join(p.name for p in paths)}", file=sys.stderr)
        sys.exit(1)

    return {paper_id: paths[0] for paper_id, paths in by_id.items()}


# --------------------------------------------------------------------------- #
# Per-paper processing (fail-soft)
# --------------------------------------------------------------------------- #
def _process_all(
    pdf_by_id: dict[str, Path], *, force: bool, only: str | None
) -> tuple[list[ProcessedPaper], list[Failure]]:
    if only is not None:
        if only not in pdf_by_id:
            print(f"parse_corpus.py: --only {only!r} matches no PDF's paper_id", file=sys.stderr)
            print(f"  known paper_ids: {', '.join(sorted(pdf_by_id))}", file=sys.stderr)
            sys.exit(1)
        targets = {only: pdf_by_id[only]}
    else:
        targets = pdf_by_id

    results: list[ProcessedPaper] = []
    failures: list[Failure] = []
    ordered = sorted(targets.items())
    for i, (paper_id, pdf_path) in enumerate(ordered, start=1):
        try:
            result = process_paper(pdf_path, force=force)
        except Exception as exc:  # noqa: BLE001 -- fail-soft is the point: one bad PDF must not abort the run
            failures.append(Failure(paper_id=paper_id, pdf_path=pdf_path, error=exc))
            print(f"[{i}/{len(ordered)}] {paper_id}: FAILED -- {type(exc).__name__}: {exc}")
            continue
        results.append(result)
        print(f"[{i}/{len(ordered)}] {paper_id}: ok ({result.n_units} units, {result.n_chars} chars)")

    return results, failures


# --------------------------------------------------------------------------- #
# Per-paper stats derived from a ProcessedPaper for the tables/flags below
# --------------------------------------------------------------------------- #
def _section_summaries(result: ProcessedPaper) -> dict[int, tuple[int | None, bool, bool]]:
    """One (section_level, heading_unnumbered, heading_junk) per distinct
    section_index among kept "section" units -- these three fields are
    properties of the *section* (the div), shared by every paragraph unit
    inside it, so counting per-unit would double-count multi-paragraph
    sections.
    """
    summaries: dict[int, tuple[int | None, bool, bool]] = {}
    for unit in result.units:
        if unit.type != "section":
            continue
        summaries.setdefault(unit.section_index, (unit.section_level, unit.heading_unnumbered, unit.heading_junk))
    return summaries


def _heading_level_counts(result: ProcessedPaper) -> tuple[int, int, int, int, int]:
    """(n_level1, n_level2, n_level3, n_unnumbered_plausible, n_junk).

    Not a partition: an unnumbered heading defaults to section_level=1 (see
    tei.py's _classify_prefix), so it's counted in both L1 and one of the
    last two columns. That's intentional -- these are six independent lenses
    on the same section set, not six mutually exclusive buckets.
    """
    summaries = _section_summaries(result).values()
    n1 = sum(1 for level, _, _ in summaries if level == 1)
    n2 = sum(1 for level, _, _ in summaries if level == 2)
    n3 = sum(1 for level, _, _ in summaries if level == 3)
    n_unnumbered_plausible = sum(1 for _, unnumbered, junk in summaries if unnumbered and not junk)
    n_junk = sum(1 for _, _, junk in summaries if junk)
    return n1, n2, n3, n_unnumbered_plausible, n_junk


def _n_merged_splits(result: ProcessedPaper) -> int:
    return sum(1 for w in result.warnings if w.startswith(_SPLIT_WARNING_PREFIX))


def _n_captions(result: ProcessedPaper) -> int:
    return sum(1 for unit in result.units if unit.type == "caption")


def _n_equation_units(result: ProcessedPaper) -> int:
    return sum(1 for unit in result.units if unit.contains_equation)


def _mean_unit_chars(result: ProcessedPaper) -> float:
    if not result.units:
        return 0.0
    return sum(unit.char_end - unit.char_start for unit in result.units) / len(result.units)


def _read_full_text(result: ProcessedPaper) -> str:
    """The actual persisted canonical text, read back from disk rather than
    reconstructed. Several flags below need this rather than per-unit stats
    (e.g. accurately weighting several per-unit non_ascii_ratio values by
    their original, pre-normalize lengths would need data AssembledUnit
    doesn't carry) -- and re-reading it here doubles as a check that the file
    was actually written correctly.
    """
    return result.txt_path.read_text(encoding="utf-8")


def _text_ratios(text: str) -> tuple[float, float]:
    """(alphabetic_ratio, non_ascii_ratio) over the given text."""
    if not text:
        return 0.0, 0.0
    n_alpha = sum(1 for ch in text if ch.isalpha())
    n_non_ascii = sum(1 for ch in text if ord(ch) > 127)
    return n_alpha / len(text), n_non_ascii / len(text)


# tei.py strips numeric citation-bracket residue (untagged ranges like
# "[59]-[65]", orphan range-dashes) at extraction time -- see its module
# docstring, item 4. This should therefore find exactly zero occurrences on
# every paper; a hit means that fix missed a shape, and it's cheap enough to
# check on all 36 papers automatically rather than by re-reading them by eye.
_CITATION_RESIDUE_RE = re.compile(r"\[\d+\]")


def _n_citation_residue(text: str) -> int:
    return len(_CITATION_RESIDUE_RE.findall(text))


def _flags_for(result: ProcessedPaper, gates: QualityGates) -> list[str]:
    """Every quality-gate flag PROMPT_1 lists, for one paper. Erring toward
    over-flagging is the explicit design goal here -- a false alarm costs a
    minute of review, a missed broken paper costs much more later.
    """
    flags: list[str] = []

    n_sections = len(_section_summaries(result))
    if n_sections < gates.min_sections:
        flags.append(f"fewer than {gates.min_sections} sections ({n_sections})")

    if result.n_chars < gates.min_chars:
        flags.append(f"under {gates.min_chars:,} characters ({result.n_chars:,})")

    mean_chars = _mean_unit_chars(result)
    if mean_chars < gates.min_mean_unit_chars:
        flags.append(f"mean unit length under {gates.min_mean_unit_chars:.0f} chars ({mean_chars:.1f})")

    text = _read_full_text(result)

    alpha_ratio, non_ascii_ratio = _text_ratios(text)
    if alpha_ratio < gates.min_alpha_ratio:
        flags.append(f"alphabetic ratio below {gates.min_alpha_ratio:.0%} ({alpha_ratio:.1%})")
    if non_ascii_ratio > gates.max_non_ascii_ratio:
        flags.append(f"non-ASCII ratio above {gates.max_non_ascii_ratio:.0%} ({non_ascii_ratio:.1%})")

    n_residue = _n_citation_residue(text)
    if n_residue > 0:
        flags.append(f"{n_residue} untagged citation-bracket residue (e.g. '[12]') left in the text")

    n_pua = sum(unit.n_pua_chars for unit in result.units)
    if n_pua > 0:
        flags.append(f"{n_pua} Private Use Area character(s)")
    n_replacement = sum(unit.n_replacement_chars for unit in result.units)
    if n_replacement > 0:
        flags.append(f"{n_replacement} U+FFFD replacement character(s)")

    if result.doi is None:
        flags.append("missing DOI")
    if not result.licence.upper().startswith("CC"):
        flags.append(f"licence is not a CC variant: {result.licence!r}")

    return flags


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def render(cells: list[str]) -> str:
        return " | ".join(cell.ljust(width) for cell, width in zip(cells, widths))

    print(render(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(render(row))


def _print_summary_table(results: list[ProcessedPaper]) -> None:
    print("\n=== Summary ===\n")
    headers = ["paper_id", "n_units", "n_chars", "n_sections", "n_captions", "eqn units", "licence", "DOI?"]
    rows = [
        [
            r.paper_id,
            str(r.n_units),
            str(r.n_chars),
            str(len(_section_summaries(r))),
            str(_n_captions(r)),
            str(_n_equation_units(r)),
            r.licence,
            "Y" if r.doi else "N",
        ]
        for r in sorted(results, key=lambda r: r.paper_id)
    ]
    _print_table(headers, rows)


def _print_heading_table(results: list[ProcessedPaper]) -> None:
    print("\n=== Heading structure ===\n")
    headers = ["paper_id", "L1", "L2", "L3", "unnumbered-plausible", "junk", "merged-splits"]
    rows = []
    for r in sorted(results, key=lambda r: r.paper_id):
        n1, n2, n3, n_unnumbered, n_junk = _heading_level_counts(r)
        rows.append([r.paper_id, str(n1), str(n2), str(n3), str(n_unnumbered), str(n_junk), str(_n_merged_splits(r))])
    _print_table(headers, rows)


def _print_flags(results: list[ProcessedPaper], gates: QualityGates) -> None:
    print("\n=== Flags ===\n")
    printed_any = False
    for r in sorted(results, key=lambda r: r.paper_id):
        items = _flags_for(r, gates) + list(r.warnings)
        if not items:
            continue
        printed_any = True
        print(f"{r.paper_id}:")
        for item in items:
            print(f"  - {item}")
    if not printed_any:
        print("(none)")


def _print_failures(failures: list[Failure]) -> None:
    if not failures:
        return
    print("\n=== Failures ===\n")
    for f in sorted(failures, key=lambda f: f.paper_id):
        print(f"{f.paper_id} ({f.pdf_path.name}): {type(f.error).__name__}: {f.error}")


# --------------------------------------------------------------------------- #
# manifest.csv
# --------------------------------------------------------------------------- #
def _regenerate_manifest(normalized_dir: Path, manifest_path: Path) -> int:
    """Rebuild manifest.csv from every ``data/normalized/*.json`` currently on
    disk -- not just the papers *this* invocation processed. A naive "write
    only what this run touched" approach would make ``--only PAPER_ID`` wipe
    out the other 35 papers' rows; reading the persisted JSON files back
    instead makes manifest.csv self-healing and correct regardless of which
    subset of the corpus was just (re)processed. Returns the row count.
    """
    rows = []
    for json_path in sorted(normalized_dir.glob("*.json")):
        doc = json.loads(json_path.read_text(encoding="utf-8"))
        rows.append({field: doc.get(field) for field in _MANIFEST_FIELDS})

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the stage-1 parsing pipeline over data/pdfs/.")
    parser.add_argument("--force", action="store_true", help="bypass GROBID/Crossref caches and re-fetch everything")
    parser.add_argument("--only", metavar="PAPER_ID", help="process only the PDF whose slug is PAPER_ID")
    args = parser.parse_args(argv)

    paths = get_paths()
    gates = get_experiment().parsing.quality_gates

    pdf_paths = _discover_pdfs(paths.pdfs)
    if not pdf_paths:
        print(f"parse_corpus.py: no PDFs found in {paths.pdfs}", file=sys.stderr)
        return 1

    pdf_by_id = _paper_id_map(pdf_paths)  # aborts (sys.exit) on a slug collision
    results, failures = _process_all(pdf_by_id, force=args.force, only=args.only)

    if results:
        _print_summary_table(results)
        _print_heading_table(results)
        _print_flags(results, gates)

    _print_failures(failures)

    n_manifest_rows = _regenerate_manifest(paths.normalized, paths.manifest_csv)
    print(f"\nWrote {paths.manifest_csv} ({n_manifest_rows} row(s)).")

    if failures:
        print(f"\n{len(failures)} paper(s) FAILED -- see Failures above.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())