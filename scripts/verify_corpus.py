"""CLI: verify that every paper in data/normalized/ still matches what was
written -- the tripwire against normalization drift once character offsets
have been hand-annotated into an evaluation set.

    uv run python scripts/verify_corpus.py

For every paper (each `data/normalized/{paper_id}.json` + its matching
`.txt`):

1. Re-hash the `.txt` file and compare against `text_sha256` in the JSON.
2. Check `n_chars` in the JSON matches the actual text length.
3. Check every unit's `char_start`/`char_end` still slices to a string of
   exactly `char_end - char_start` characters. Python slicing never raises
   for an out-of-range end index -- it just silently truncates -- so a
   length mismatch is the only reliable way to catch a unit whose range now
   runs past the end of the text.
4. Structural checks over the units, in document order, that pin every
   offset absolutely rather than just its length (a length-only check can't
   tell a correct offset from a same-length one shifted to somewhere else
   entirely valid in the same file):
     - `units[0].char_start == 0`
     - `units[-1].char_end == n_chars`
     - every unit's `char_end > char_start`
     - consecutive units satisfy `next.char_start == prev.char_end +
       len(unit_separator)`
     - the text at each inter-unit gap is exactly `unit_separator`

It then checks the evaluation set (`data/eval/questions.v2.jsonl`) against
the same normalized text:

5. Every `qid` is unique.
6. Every non-null `doi` resolves to exactly one paper in `data/manifest.csv`,
   and any `paper_id` already filled in on the question agrees with that
   resolution (so a bad `eval_set.py backfill` is caught here).
7. For every entry in a question's `gold_spans`, the text at
   `[char_start:char_end]` in that paper's `.txt` equals the span's `text`
   field exactly. This is the eval-set equivalent of check 3: a gold span is
   just an offset into the frozen text, and relevance in Phase F is judged by
   character overlap, so an offset that has drifted off its text corrupts
   every metric computed against it.

Run this before every annotation session and at the start of every
evaluation run: once a gold span is recorded as an offset into one of these
files, any change to that file -- even whitespace -- silently invalidates
every span built against it. This script is what turns "silently" into
"this command fails, loudly, right now."

This is deliberately independent of the rest of the pipeline: it re-derives
nothing from `tei.py`/`normalize.py`/`pipeline.py`, doesn't touch GROBID or
Crossref, and only ever reads the two files already on disk per paper -- a
drift check that itself depended on the pipeline being correct wouldn't be
much of a tripwire.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ragvlc.config import get_experiment, get_paths


@dataclass
class PaperIssues:
    paper_id: str
    problems: list[str] = field(default_factory=list)


def _verify_one(json_path: Path, unit_separator: str) -> PaperIssues:
    paper_id = json_path.stem
    issues = PaperIssues(paper_id=paper_id)

    try:
        doc = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        issues.problems.append(f"could not read/parse {json_path.name}: {exc}")
        return issues

    txt_path = json_path.with_suffix(".txt")
    if not txt_path.is_file():
        issues.problems.append(f"missing {txt_path.name}")
        return issues
    raw_bytes = txt_path.read_bytes()

    expected_hash = doc.get("text_sha256")
    actual_hash = hashlib.sha256(raw_bytes).hexdigest()
    if expected_hash is None:
        issues.problems.append("no text_sha256 in the JSON (needs a re-run of parse_corpus.py to populate)")
    elif actual_hash != expected_hash:
        issues.problems.append(f"text_sha256 mismatch: file hashes to {actual_hash}, JSON says {expected_hash}")

    # Decoded from the *same* bytes just hashed, so the length/slice checks
    # below can never disagree with the hash check over what "the text" is.
    text = raw_bytes.decode("utf-8")

    n_chars = doc.get("n_chars")
    if n_chars != len(text):
        issues.problems.append(f"n_chars mismatch: JSON says {n_chars}, actual text has {len(text)} character(s)")

    units = doc.get("units", [])
    last_index = len(units) - 1
    prev_end: int | None = None
    prev_label: str | None = None

    for i, unit in enumerate(units):
        label = f"unit[{i}] ({unit.get('unit_id', '<unknown>')})"
        start, end = unit.get("char_start"), unit.get("char_end")
        if start is None or end is None:
            issues.problems.append(f"{label}: missing char_start/char_end")
            prev_end, prev_label = None, None  # nothing valid to chain the next gap check off
            continue

        # Slice length: catches an out-of-range end (Python truncates rather
        # than raising, so length is the only signal).
        expected_len = end - start
        actual_len = len(text[start:end])
        if actual_len != expected_len:
            issues.problems.append(
                f"{label}: char_start={start}, char_end={end} slices to {actual_len} "
                f"character(s), expected {expected_len}"
            )

        # Structural checks: these pin the offset absolutely, so a
        # shifted-but-in-bounds char_end (same length, wrong position) fails
        # here even though it would pass the length check above.
        if end <= start:
            issues.problems.append(f"{label}: char_end ({end}) is not greater than char_start ({start})")

        if i == 0 and start != 0:
            issues.problems.append(f"{label}: is the first unit but char_start={start}, expected 0")

        if i == last_index and n_chars is not None and end != n_chars:
            issues.problems.append(f"{label}: is the last unit but char_end={end}, expected n_chars={n_chars}")

        if prev_end is not None:
            expected_start = prev_end + len(unit_separator)
            if start != expected_start:
                issues.problems.append(
                    f"{label}: char_start={start} does not follow {prev_label} (char_end={prev_end}) by "
                    f"len(unit_separator)={len(unit_separator)}; expected char_start={expected_start}"
                )
            gap = text[prev_end:start]
            if gap != unit_separator:
                issues.problems.append(
                    f"{label}: text between {prev_label} and this unit is {gap!r}, "
                    f"expected the separator {unit_separator!r}"
                )

        prev_end, prev_label = end, label

    return issues


# --------------------------------------------------------------------------- #
# Eval-set checks (data/eval/questions.v2.jsonl)
#
# Deliberately self-contained, like the rest of this file: the DOI
# normalization below is a three-line copy of scripts/eval_set.py's
# `normalize_doi` rather than an import, so the tripwire keeps working even if
# eval_set.py is mid-edit or broken.
# --------------------------------------------------------------------------- #
_DOI_URL_PREFIXES: tuple[str, ...] = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi.org/",
    "doi:",
)


def _normalize_doi(raw: str) -> str:
    lowered = raw.strip().lower()
    for prefix in _DOI_URL_PREFIXES:
        if lowered.startswith(prefix):
            return lowered[len(prefix) :].strip()
    return lowered


def _load_manifest_doi_index(manifest_csv: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            doi, paper_id = row.get("doi"), row.get("paper_id")
            if doi and paper_id:
                index[_normalize_doi(doi)] = paper_id
    return index


def _verify_eval_set(questions_jsonl: Path, manifest_csv: Path, normalized_dir: Path) -> list[str]:
    """Return a flat list of problems with the eval set (empty == clean)."""
    problems: list[str] = []

    if not questions_jsonl.is_file():
        return [f"missing eval set: {questions_jsonl}"]
    if not manifest_csv.is_file():
        return [f"missing manifest: {manifest_csv}"]

    doi_index = _load_manifest_doi_index(manifest_csv)

    text_cache: dict[str, str] = {}

    def load_text(paper_id: str) -> str | None:
        if paper_id not in text_cache:
            txt_path = normalized_dir / f"{paper_id}.txt"
            if not txt_path.is_file():
                return None
            text_cache[paper_id] = txt_path.read_text(encoding="utf-8")
        return text_cache[paper_id]

    seen_qids: set[str] = set()

    with questions_jsonl.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                q = json.loads(stripped)
            except json.JSONDecodeError as exc:
                problems.append(f"{questions_jsonl.name}:{lineno}: invalid JSON: {exc}")
                continue

            qid = str(q.get("qid", f"<line {lineno}>"))
            if qid in seen_qids:
                problems.append(f"{qid}: duplicate qid")
            seen_qids.add(qid)

            doi = q.get("doi")
            resolved_paper_id: str | None = None
            if doi is not None and str(doi).strip() != "":
                resolved_paper_id = doi_index.get(_normalize_doi(str(doi)))
                if resolved_paper_id is None:
                    problems.append(f"{qid}: doi {doi!r} does not resolve to any manifest paper")

            stated_paper_id = q.get("paper_id")
            if (
                stated_paper_id is not None
                and resolved_paper_id is not None
                and stated_paper_id != resolved_paper_id
            ):
                problems.append(
                    f"{qid}: paper_id {stated_paper_id!r} disagrees with doi-resolved {resolved_paper_id!r}"
                )

            gold_spans = q.get("gold_spans") or []
            if not gold_spans:
                continue

            paper_id = resolved_paper_id or stated_paper_id
            if paper_id is None:
                problems.append(f"{qid}: has {len(gold_spans)} gold_span(s) but no resolvable paper")
                continue
            text = load_text(paper_id)
            if text is None:
                problems.append(f"{qid}: gold_spans reference paper {paper_id!r} with no .txt in {normalized_dir}")
                continue

            for i, span in enumerate(gold_spans):
                label = f"{qid}: gold_spans[{i}]"
                start, end, span_text = span.get("char_start"), span.get("char_end"), span.get("text")
                if not isinstance(start, int) or not isinstance(end, int) or not isinstance(span_text, str):
                    problems.append(f"{label}: missing/invalid char_start, char_end, or text")
                    continue
                if not (0 <= start < end <= len(text)):
                    problems.append(
                        f"{label}: char range [{start}:{end}] is out of bounds for {paper_id!r} "
                        f"(len {len(text)})"
                    )
                    continue
                actual = text[start:end]
                if actual != span_text:
                    problems.append(
                        f"{label}: text at [{start}:{end}] does not match the recorded span\n"
                        f"      recorded: {span_text!r}\n"
                        f"      on disk:  {actual!r}"
                    )

    return problems


def main(argv: list[str] | None = None) -> int:
    paths = get_paths()
    unit_separator = get_experiment().parsing.unit_separator

    json_paths = sorted(paths.normalized.glob("*.json"))
    if not json_paths:
        print(f"verify_corpus.py: no papers found in {paths.normalized}", file=sys.stderr)
        return 1

    all_issues = [issues for issues in (_verify_one(p, unit_separator) for p in json_paths) if issues.problems]
    eval_problems = _verify_eval_set(paths.questions_jsonl, paths.manifest_csv, paths.normalized)

    if not all_issues and not eval_problems:
        print(
            f"verify_corpus.py: OK -- {len(json_paths)} paper(s) verified and the eval set "
            f"({paths.questions_jsonl.name}) is consistent, no drift detected."
        )
        return 0

    if all_issues:
        print(
            f"verify_corpus.py: DRIFT DETECTED in {len(all_issues)} of {len(json_paths)} paper(s):\n",
            file=sys.stderr,
        )
        for issues in all_issues:
            print(f"{issues.paper_id}:", file=sys.stderr)
            for problem in issues.problems:
                print(f"  - {problem}", file=sys.stderr)

    if eval_problems:
        print(f"\nverify_corpus.py: {len(eval_problems)} problem(s) in {paths.questions_jsonl.name}:\n", file=sys.stderr)
        for problem in eval_problems:
            print(f"  - {problem}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
