"""CLI: the annotation helper -- find a copied sentence in the normalized
corpus and print offsets ready to paste into `data/eval/questions.jsonl`.

    uv run python scripts/locate.py --paper kim-2023-indoor-vlc --text "the sentence I copied"
    uv run python scripts/locate.py --text "a distinctive sentence"     # searches all papers

The search text is run through `normalize()` before matching, so pasted text
with different line-wrapping, spacing, or a stray non-breaking space still
matches exactly what's in the canonical `.txt` file.

Before searching a paper, this runs `verify_corpus.py`'s per-paper check
(hash, `n_chars`, and the structural offset checks) on it and refuses to
search -- and refuses to return any offsets for -- a paper that fails. An
offset out of a drifted file is worse than no offset at all: it would look
like a normal gold span right up until it silently pointed at the wrong text.

Output depends on how many matches turn up:

- **Exactly one**: `paper_id`, `char_start`, `char_end`, the enclosing unit's
  `section_heading`/`parent_heading`/`type`, ~100 characters of context on
  each side, and a ready-to-paste JSON fragment (`doi`, `paper_id`,
  `section_heading`, `gold_span`, `gold_char_start`, `gold_char_end`) printed
  as a single compact line, since the destination is JSON**L**.
- **Zero**: says so, then uses `difflib` (`SequenceMatcher.find_longest_match`,
  `autojunk=False` -- see the comment above `_nearest_passage`) to find and
  show the longest matching passage anywhere in the searched text, so a typo
  or a search string that happens to span a `\\n\\n` unit boundary is visible
  rather than just "not found".
- **More than one**: every match, each with its own context, to disambiguate.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from dataclasses import dataclass

import verify_corpus

from ragvlc.config import get_experiment, get_paths
from ragvlc.parsing.normalize import normalize

_CONTEXT_RADIUS = 100


@dataclass
class LoadedPaper:
    paper_id: str
    doc: dict
    text: str


@dataclass
class Match:
    paper_id: str
    char_start: int
    char_end: int


# --------------------------------------------------------------------------- #
# Verification gate -- reuses verify_corpus.py rather than re-checking hashes
# itself, so there is exactly one place that knows what "still trustworthy"
# means for a normalized-output file.
# --------------------------------------------------------------------------- #
def _verify_candidates(paper_ids: list[str], unit_separator: str, paths) -> tuple[list[str], dict[str, list[str]]]:
    ok: list[str] = []
    failed: dict[str, list[str]] = {}
    for paper_id in paper_ids:
        json_path = paths.normalized / f"{paper_id}.json"
        issues = verify_corpus._verify_one(json_path, unit_separator)
        if issues.problems:
            failed[paper_id] = issues.problems
        else:
            ok.append(paper_id)
    return ok, failed


def _load_paper(paper_id: str, paths) -> LoadedPaper:
    json_path = paths.normalized / f"{paper_id}.json"
    txt_path = paths.normalized / f"{paper_id}.txt"
    doc = json.loads(json_path.read_text(encoding="utf-8"))
    text = txt_path.read_text(encoding="utf-8")
    return LoadedPaper(paper_id=paper_id, doc=doc, text=text)


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
def _find_matches(papers: dict[str, LoadedPaper], search_text: str) -> list[Match]:
    matches: list[Match] = []
    for paper_id, paper in papers.items():
        start = 0
        while True:
            idx = paper.text.find(search_text, start)
            if idx == -1:
                break
            matches.append(Match(paper_id=paper_id, char_start=idx, char_end=idx + len(search_text)))
            start = idx + 1
    matches.sort(key=lambda m: (m.paper_id, m.char_start))
    return matches


def _enclosing_unit(doc: dict, char_start: int) -> dict | None:
    for unit in doc.get("units", []):
        if unit["char_start"] <= char_start < unit["char_end"]:
            return unit
    return None


def _context(text: str, start: int, end: int, radius: int = _CONTEXT_RADIUS) -> str:
    before_from = max(0, start - radius)
    after_to = min(len(text), end + radius)
    prefix = "..." if before_from > 0 else ""
    suffix = "..." if after_to < len(text) else ""
    return f"{prefix}{text[before_from:start]}[[{text[start:end]}]]{text[end:after_to]}{suffix}"


# --------------------------------------------------------------------------- #
# Zero matches: nearest passage via difflib
# --------------------------------------------------------------------------- #
def _nearest_passage(papers: dict[str, LoadedPaper], search_text: str) -> tuple[str, int, int] | None:
    """(paper_id, start, size) of the single longest contiguous passage,
    across all searched papers, that matches part of `search_text`.

    `autojunk=False` is not optional here: SequenceMatcher's default
    autojunk heuristic discards any character that recurs often enough in a
    long sequence as "popular", which for ordinary prose (long stretches of
    the same few dozen letters and spaces) would gut real matches rather
    than just filtering noise -- it's meant for things like source code, not
    natural-language text.
    """
    best: tuple[int, str, int] | None = None  # (size, paper_id, start)
    for paper_id, paper in papers.items():
        matcher = difflib.SequenceMatcher(None, paper.text, search_text, autojunk=False)
        m = matcher.find_longest_match(0, len(paper.text), 0, len(search_text))
        if m.size > 0 and (best is None or m.size > best[0]):
            best = (m.size, paper_id, m.a)
    if best is None:
        return None
    size, paper_id, start = best
    return paper_id, start, size


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _print_unit_info(unit: dict | None) -> None:
    if unit is None:
        print("(no single recorded unit contains this match -- it may span a unit boundary)")
        return
    print(f"section_heading: {unit.get('section_heading')}")
    print(f"parent_heading:  {unit.get('parent_heading')}")
    print(f"type:            {unit.get('type')}")


def _print_single_match(papers: dict[str, LoadedPaper], match: Match) -> None:
    paper = papers[match.paper_id]
    unit = _enclosing_unit(paper.doc, match.char_start)

    print("Exactly one match.\n")
    print(f"paper_id:        {match.paper_id}")
    print(f"char_start:      {match.char_start}")
    print(f"char_end:        {match.char_end}")
    _print_unit_info(unit)
    print()
    print(_context(paper.text, match.char_start, match.char_end))

    fragment = {
        "doi": paper.doc.get("doi"),
        "paper_id": match.paper_id,
        "section_heading": unit.get("section_heading") if unit else None,
        "gold_span": paper.text[match.char_start : match.char_end],
        "gold_char_start": match.char_start,
        "gold_char_end": match.char_end,
    }
    print("\nReady to paste into data/eval/questions.jsonl:\n")
    # One line, no indent: the destination is JSON*L*, one record per line.
    print(json.dumps(fragment, ensure_ascii=False))


def _print_multiple_matches(papers: dict[str, LoadedPaper], matches: list[Match]) -> None:
    print(f"{len(matches)} matches -- disambiguate with --paper or a more specific --text.\n")
    for i, match in enumerate(matches, start=1):
        paper = papers[match.paper_id]
        unit = _enclosing_unit(paper.doc, match.char_start)
        heading = unit.get("section_heading") if unit else "?"
        print(f"[{i}] {match.paper_id}  char_start={match.char_start} char_end={match.char_end}  section={heading!r}")
        print(f"    {_context(paper.text, match.char_start, match.char_end, radius=60)}\n")


def _print_no_match(papers: dict[str, LoadedPaper], search_text: str) -> None:
    print("No exact match found.\n")
    nearest = _nearest_passage(papers, search_text)
    if nearest is None:
        print("No similar passage found either -- the text may not be in the corpus at all.")
        return
    paper_id, start, size = nearest
    paper = papers[paper_id]
    print(f"Nearest passage found ({size} matching character(s), via difflib) in {paper_id}:\n")
    print(_context(paper.text, start, start + size))
    print("\nCheck for a typo, or whether the text spans a \\n\\n unit boundary.")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Find a copied passage in the normalized corpus.")
    parser.add_argument("--paper", metavar="PAPER_ID", help="restrict the search to one paper")
    parser.add_argument("--text", required=True, help="the text to locate (any spacing -- it's normalized first)")
    args = parser.parse_args(argv)

    paths = get_paths()
    unit_separator = get_experiment().parsing.unit_separator

    search_text = normalize(args.text)
    if not search_text:
        print("locate.py: --text normalizes to an empty string -- nothing to search for", file=sys.stderr)
        return 1

    if args.paper:
        if not (paths.normalized / f"{args.paper}.json").is_file():
            print(f"locate.py: no such paper {args.paper!r} in {paths.normalized}", file=sys.stderr)
            return 1
        candidate_ids = [args.paper]
    else:
        candidate_ids = sorted(p.stem for p in paths.normalized.glob("*.json"))
        if not candidate_ids:
            print(f"locate.py: no papers found in {paths.normalized}", file=sys.stderr)
            return 1

    ok_ids, failed = _verify_candidates(candidate_ids, unit_separator, paths)
    for paper_id, problems in failed.items():
        print(f"locate.py: REFUSING {paper_id} -- verify_corpus check failed:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(file=sys.stderr)

    if args.paper and args.paper in failed:
        print(f"locate.py: {args.paper} failed verification -- refusing to search it or return offsets.", file=sys.stderr)
        return 1

    if not ok_ids:
        print("locate.py: no verified papers left to search.", file=sys.stderr)
        return 1

    papers = {paper_id: _load_paper(paper_id, paths) for paper_id in ok_ids}
    matches = _find_matches(papers, search_text)

    if len(matches) == 1:
        _print_single_match(papers, matches[0])
    elif len(matches) == 0:
        _print_no_match(papers, search_text)
    else:
        _print_multiple_matches(papers, matches)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
