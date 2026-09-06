"""CLI: wire up the evaluation set (`data/eval/questions.v2.jsonl`).

    uv run python scripts/eval_set.py backfill
    uv run python scripts/eval_set.py backfill --init-has-answer

The questions file is hand-written: each entry has a `question`, a `doi`
(null for the expected-negative questions), and hand-found `gold_spans`. Two
fields are derived rather than typed by hand, and this script fills them in:

* **`backfill`** joins each question's `doi` to `data/manifest.csv` to populate
  `paper_id`. The DOI is matched case-insensitively and with any
  `https://doi.org/` URL prefix stripped, because DOIs get pasted in both
  forms. A DOI that does not resolve is reported and left as `null` -- we never
  guess a paper.

* **`--init-has-answer`** sets `has_answer` from the qid convention: `q...`
  questions were written to be answerable from the corpus, `n...` questions
  were written to have no answer. This only records the prior held at
  question-writing time; Phase E's pooled judging revises it (a `q` question
  whose pool turns up nothing relevant flips to `false`).

Library functions here return result objects; `main` decides what to print and
what exit code to use. Rewrites are atomic (temp file + `os.replace`) so an
interrupted run cannot truncate the eval set, and preserve JSON key order and
non-ASCII text exactly.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ragvlc.config import get_paths

# DOI URL prefixes seen when a DOI is pasted from a browser rather than typed.
# Order matters: the longest match is stripped, so check specific before bare.
_DOI_URL_PREFIXES: tuple[str, ...] = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi.org/",
    "doi:",
)


def normalize_doi(raw: str) -> str:
    """Canonical form of a DOI for equality checks: lower-cased, whitespace
    stripped, and with any resolver-URL prefix removed.

    DOIs are officially case-insensitive, so comparing them requires case
    folding on both sides.
    """
    value = raw.strip()
    lowered = value.lower()
    for prefix in _DOI_URL_PREFIXES:
        if lowered.startswith(prefix):
            return lowered[len(prefix) :].strip()
    return lowered


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_manifest_doi_index(manifest_csv: Path) -> dict[str, str]:
    """Map normalized DOI -> paper_id from `data/manifest.csv`.

    Raises `ValueError` if two rows share a normalized DOI, since that would
    make the backfill join ambiguous.
    """
    index: dict[str, str] = {}
    with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            doi = row.get("doi")
            paper_id = row.get("paper_id")
            if not doi or not paper_id:
                continue
            key = normalize_doi(doi)
            if key in index and index[key] != paper_id:
                raise ValueError(
                    f"manifest has two paper_ids for DOI {key!r}: "
                    f"{index[key]!r} and {paper_id!r}"
                )
            index[key] = paper_id
    return index


def read_questions(questions_jsonl: Path) -> list[dict]:
    """Parse the JSONL eval set into a list of dicts, preserving key order."""
    rows: list[dict] = []
    with questions_jsonl.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{questions_jsonl.name}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def write_questions(questions_jsonl: Path, rows: list[dict]) -> None:
    """Rewrite the JSONL eval set atomically.

    `ensure_ascii=False` keeps the corpus's non-ASCII characters readable and
    byte-identical to how they were typed; the default separators match the
    existing file (one record per line, `", "` / `": "`).
    """
    tmp_path = questions_jsonl.with_suffix(questions_jsonl.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    os.replace(tmp_path, questions_jsonl)


# --------------------------------------------------------------------------- #
# backfill
# --------------------------------------------------------------------------- #
@dataclass
class BackfillResult:
    resolved: int = 0            # questions whose paper_id was set or confirmed
    changed: int = 0             # of those, ones whose paper_id actually changed
    skipped_null_doi: int = 0    # expected-negative questions with no DOI
    unresolved: list[tuple[str, str]] = field(default_factory=list)  # (qid, doi)
    conflicts: list[str] = field(default_factory=list)               # human-readable

    @property
    def ok(self) -> bool:
        return not self.unresolved and not self.conflicts


def backfill_paper_ids(rows: list[dict], doi_index: dict[str, str]) -> BackfillResult:
    """Populate each question's `paper_id` from its `doi`. Mutates `rows`.

    A question with no `doi` (the expected negatives) is left untouched. A DOI
    that is not in the manifest is recorded in `unresolved` and its `paper_id`
    is left as-is -- we do not guess.
    """
    result = BackfillResult()
    for row in rows:
        qid = str(row.get("qid", "<no qid>"))
        doi = row.get("doi")
        if doi is None or str(doi).strip() == "":
            result.skipped_null_doi += 1
            continue

        paper_id = doi_index.get(normalize_doi(str(doi)))
        if paper_id is None:
            result.unresolved.append((qid, str(doi)))
            continue

        previous = row.get("paper_id")
        if previous is not None and previous != paper_id:
            # Not fatal on its own, but worth surfacing: a hand-typed paper_id
            # that disagrees with the DOI join means one of the two is wrong.
            result.conflicts.append(
                f"{qid}: existing paper_id {previous!r} != DOI-resolved {paper_id!r} (overwriting)"
            )
        if previous != paper_id:
            result.changed += 1
        row["paper_id"] = paper_id
        result.resolved += 1
    return result


# --------------------------------------------------------------------------- #
# --init-has-answer
# --------------------------------------------------------------------------- #
@dataclass
class HasAnswerResult:
    set_true: int = 0
    set_false: int = 0
    unknown_prefix: list[str] = field(default_factory=list)  # qids

    @property
    def ok(self) -> bool:
        return not self.unknown_prefix


def init_has_answer(rows: list[dict]) -> HasAnswerResult:
    """Set `has_answer` from the qid prefix: `q...` -> True, `n...` -> False.

    Mutates `rows`. A qid with any other prefix is left untouched and reported.
    """
    result = HasAnswerResult()
    for row in rows:
        qid = str(row.get("qid", ""))
        if qid.startswith("q"):
            row["has_answer"] = True
            result.set_true += 1
        elif qid.startswith("n"):
            row["has_answer"] = False
            result.set_false += 1
        else:
            result.unknown_prefix.append(qid or "<no qid>")
    return result


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _report_backfill(result: BackfillResult) -> None:
    print(
        f"backfill: {result.resolved} question(s) resolved to a paper "
        f"({result.changed} changed), {result.skipped_null_doi} skipped (no DOI)."
    )
    for message in result.conflicts:
        print(f"  conflict: {message}", file=sys.stderr)
    if result.unresolved:
        print(f"  {len(result.unresolved)} DOI(s) did not resolve to any manifest paper:", file=sys.stderr)
        for qid, doi in result.unresolved:
            print(f"    - {qid}: {doi}", file=sys.stderr)


def _report_has_answer(result: HasAnswerResult) -> None:
    print(
        f"init-has-answer: {result.set_true} set to true (q...), "
        f"{result.set_false} set to false (n...)."
    )
    if result.unknown_prefix:
        print(f"  {len(result.unknown_prefix)} qid(s) with an unrecognised prefix (left untouched):", file=sys.stderr)
        for qid in result.unknown_prefix:
            print(f"    - {qid}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wire up the evaluation-set file.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    backfill_parser = subparsers.add_parser(
        "backfill",
        help="populate each question's paper_id from its doi via data/manifest.csv",
    )
    backfill_parser.add_argument(
        "--init-has-answer",
        action="store_true",
        help="also set has_answer from the qid prefix (q -> true, n -> false)",
    )

    args = parser.parse_args(argv)
    paths = get_paths()

    if not paths.questions_jsonl.is_file():
        print(f"eval_set.py: no eval set at {paths.questions_jsonl}", file=sys.stderr)
        return 1
    if not paths.manifest_csv.is_file():
        print(f"eval_set.py: no manifest at {paths.manifest_csv}", file=sys.stderr)
        return 1

    try:
        doi_index = load_manifest_doi_index(paths.manifest_csv)
        rows = read_questions(paths.questions_jsonl)
    except (OSError, ValueError) as exc:
        print(f"eval_set.py: {exc}", file=sys.stderr)
        return 1

    if args.command != "backfill":  # pragma: no cover -- argparse enforces this
        parser.error(f"unknown command {args.command!r}")

    backfill_result = backfill_paper_ids(rows, doi_index)
    has_answer_result: HasAnswerResult | None = None
    if args.init_has_answer:
        has_answer_result = init_has_answer(rows)

    try:
        write_questions(paths.questions_jsonl, rows)
    except OSError as exc:
        print(f"eval_set.py: failed to write {paths.questions_jsonl}: {exc}", file=sys.stderr)
        return 1

    _report_backfill(backfill_result)
    if has_answer_result is not None:
        _report_has_answer(has_answer_result)

    everything_ok = backfill_result.ok and (has_answer_result is None or has_answer_result.ok)
    if not everything_ok:
        print("\neval_set.py: finished with unresolved items (see above).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
