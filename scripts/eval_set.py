"""CLI: wire up and reconcile the evaluation set (`data/eval/questions.v2.jsonl`).

    uv run python scripts/eval_set.py backfill [--init-has-answer]
    uv run python scripts/eval_set.py reconcile

* **`backfill`** joins each question's `doi` to `data/manifest.csv` to populate
  `paper_id` (case-insensitive, resolver-URL prefix stripped; unresolved DOIs
  are reported, never guessed). `--init-has-answer` additionally sets
  `has_answer` from the qid convention (`q...` -> true, `n...` -> false) --
  the prior held when the questions were written.

* **`reconcile`** runs after Phase E judging. Two directions, the first
  unconditional: any question (`q` or `n`) that has a recorded relevant
  gold_span is corrected to `has_answer: true` -- this both fixes an `n`
  question that turned out answerable and self-heals a `q` question a
  previous run flipped to `false` before a manual span was added, no matter
  how it got into that state. Second, a `q` question with no gold_spans whose
  entire pool has been judged has `has_answer` flipped to `false`. A `q`
  question with un-judged pool items is left alone and flagged.

Library functions return result objects; `main` decides what to print and what
exit code to use. Rewrites are atomic and preserve JSON key order / non-ASCII
text exactly.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ragvlc.config import get_paths
from ragvlc.eval.reconcile import ReconcileResult, reconcile_has_answer
from ragvlc.eval.store import read_jsonl, read_questions, write_questions

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


# `read_questions` / `write_questions` come from ragvlc.eval.store (same atomic,
# order-preserving JSONL I/O the pooling and judging scripts use).


# --------------------------------------------------------------------------- #
# backfill
# --------------------------------------------------------------------------- #
@dataclass
class BackfillResult:
    resolved: int = 0            # questions whose paper_id was set or confirmed
    changed: int = 0             # of those, ones whose paper_id actually changed
    skipped_null_doi: int = 0    # expected-negative questions with no DOI
    spans_stamped: int = 0       # gold_spans that had paper_id filled in
    unresolved: list[tuple[str, str]] = field(default_factory=list)  # (qid, doi)
    conflicts: list[str] = field(default_factory=list)               # human-readable

    @property
    def ok(self) -> bool:
        return not self.unresolved and not self.conflicts


def backfill_paper_ids(rows: list[dict], doi_index: dict[str, str]) -> BackfillResult:
    """Populate each question's `paper_id` from its `doi`, and stamp that
    `paper_id` onto every `gold_span` that lacks one. Mutates `rows`.

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

        spans = row.get("gold_spans", [])
        for j, span in enumerate(spans):
            if span.get("paper_id") == paper_id:
                continue
            if span.get("paper_id") is not None:
                result.conflicts.append(
                    f"{qid}: gold_span paper_id {span['paper_id']!r} != question paper {paper_id!r}"
                )
                continue
            # Rebuild so paper_id sits in the same position judge.py writes it
            # (after char_end), rather than tacked on at the end.
            spans[j] = {
                "text": span.get("text"),
                "char_start": span.get("char_start"),
                "char_end": span.get("char_end"),
                "paper_id": paper_id,
                **{k: v for k, v in span.items() if k not in ("text", "char_start", "char_end", "paper_id")},
            }
            result.spans_stamped += 1
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
        f"({result.changed} changed), {result.skipped_null_doi} skipped (no DOI); "
        f"{result.spans_stamped} gold_span(s) stamped with paper_id."
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


# --------------------------------------------------------------------------- #
# reconcile  (logic lives in ragvlc.eval.reconcile)
# --------------------------------------------------------------------------- #
def _report_reconcile(result: ReconcileResult) -> None:
    print(
        f"reconcile: {result.confirmed_true} answerable + {result.confirmed_false} "
        f"not-answerable already consistent."
    )
    if result.corrected_to_true:
        print(f"  {len(result.corrected_to_true)} corrected to has_answer=true (had a gold_span, prior said otherwise):")
        for qid in result.corrected_to_true:
            print(f"    - {qid}")
    if result.flipped_to_false:
        print(f"  {len(result.flipped_to_false)} flipped to has_answer=false (pool judged, nothing relevant):")
        for qid in result.flipped_to_false:
            print(f"    - {qid}")
    if result.incomplete:
        print(f"  {len(result.incomplete)} question(s) still have un-judged pool items -- not flipped:", file=sys.stderr)
        for qid, n in result.incomplete:
            print(f"    - {qid}: {n} item(s) left to judge", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _cmd_backfill(args: argparse.Namespace, paths) -> int:
    if not paths.manifest_csv.is_file():
        print(f"eval_set.py: no manifest at {paths.manifest_csv}", file=sys.stderr)
        return 1
    try:
        doi_index = load_manifest_doi_index(paths.manifest_csv)
        rows = read_questions(paths.questions_jsonl)
    except (OSError, ValueError) as exc:
        print(f"eval_set.py: {exc}", file=sys.stderr)
        return 1

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

    if backfill_result.ok and (has_answer_result is None or has_answer_result.ok):
        return 0
    print("\neval_set.py: finished with unresolved items (see above).", file=sys.stderr)
    return 1


def _cmd_reconcile(args: argparse.Namespace, paths) -> int:
    del args
    if not paths.pool_jsonl.is_file():
        print(f"eval_set.py: no pool at {paths.pool_jsonl} -- run scripts/build_pool.py and judge first", file=sys.stderr)
        return 1
    try:
        rows = read_questions(paths.questions_jsonl)
        pool_rows = read_jsonl(paths.pool_jsonl)
        negatives = read_jsonl(paths.judgments_jsonl)
    except (OSError, ValueError) as exc:
        print(f"eval_set.py: {exc}", file=sys.stderr)
        return 1

    result = reconcile_has_answer(rows, pool_rows, negatives)

    if result.corrected_to_true or result.flipped_to_false:
        try:
            write_questions(paths.questions_jsonl, rows)
        except OSError as exc:
            print(f"eval_set.py: failed to write {paths.questions_jsonl}: {exc}", file=sys.stderr)
            return 1

    _report_reconcile(result)
    return 0 if result.ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wire up and reconcile the evaluation set.")
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
    subparsers.add_parser(
        "reconcile",
        help="after judging: sync has_answer to gold_spans (true if any exist) and to fully-judged empty pools (false)",
    )

    args = parser.parse_args(argv)
    paths = get_paths()

    if not paths.questions_jsonl.is_file():
        print(f"eval_set.py: no eval set at {paths.questions_jsonl}", file=sys.stderr)
        return 1

    if args.command == "backfill":
        return _cmd_backfill(args, paths)
    if args.command == "reconcile":
        return _cmd_reconcile(args, paths)
    parser.error(f"unknown command {args.command!r}")  # pragma: no cover
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
