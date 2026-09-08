"""CLI: build the judging pool.

    uv run python scripts/build_pool.py

Runs all 8 retrieval configurations (2 chunkers x 4 modes) over every question,
takes top-k per config, dedupes, seeds the manual gold spans, and writes
``data/eval/pool.jsonl`` -- one record per (question, pooled item), each
carrying the item's text, char range, section heading, and the configs that
retrieved it (stored, never shown during judging).

Prints a size distribution at the end: total items, items per question, and the
raw-vs-deduped ratio -- the judging burden, before committing to it.

Needs the Qdrant collections ingested (scripts/ingest.py).
"""

from __future__ import annotations

import statistics
import sys

from qdrant_client import QdrantClient

from ragvlc.config import get_experiment, get_paths, get_settings
from ragvlc.eval.pool import CONFIGS, build_pool
from ragvlc.eval.store import read_questions, write_jsonl
from ragvlc.retrieval import Searcher


def _percentile(values: list[int], pct: float) -> float:
    """``pct``-th percentile (0-100), linear interpolation. stdlib only."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (pct / 100) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def main(argv: list[str] | None = None) -> int:
    _ = argv  # no options yet
    paths = get_paths()
    retrieval = get_experiment().retrieval
    settings = get_settings()

    if not paths.questions_jsonl.is_file():
        print(f"build_pool.py: no eval set at {paths.questions_jsonl}", file=sys.stderr)
        return 1

    questions = read_questions(paths.questions_jsonl)
    collections = retrieval.qdrant.collections

    client = QdrantClient(url=settings.qdrant_url)
    try:
        missing = [name for name in collections.values() if not client.collection_exists(name)]
    except Exception as exc:  # broad on purpose: turn any connection failure into a hint
        print(f"build_pool.py: cannot reach Qdrant at {settings.qdrant_url}: {exc}", file=sys.stderr)
        return 1
    if missing:
        print(f"build_pool.py: collections not ingested: {missing} -- run scripts/ingest.py", file=sys.stderr)
        return 1

    searcher = Searcher.from_config()

    def _progress(done: int, total: int) -> None:
        if done % len(CONFIGS) == 0 or done == total:  # once per question
            print(f"  searched {done:>4}/{total}", end="\r", flush=True)

    print(f"pooling {len(questions)} questions x {len(CONFIGS)} configs...")
    pool, stats = build_pool(searcher, questions, collections, top_k=10, on_progress=_progress)
    print()

    write_jsonl(paths.pool_jsonl, [item.to_json_dict() for item in pool])

    per_q = stats.per_question
    reduction = stats.raw_items / stats.pooled_items if stats.pooled_items else 0.0
    removed_pct = 100 * (1 - stats.pooled_items / stats.raw_items) if stats.raw_items else 0.0
    n_manual = sum(1 for item in pool if item.seed == "manual")

    answerable_qids = {q["qid"] for q in questions if q.get("has_answer")}
    n_answerable_items = sum(1 for item in pool if item.qid in answerable_qids)
    n_negative_items = stats.pooled_items - n_answerable_items

    print()
    print(f"pool written: {paths.pool_jsonl.relative_to(paths.root)}")
    print(f"  questions:          {stats.n_questions}")
    print(f"  pooled items:       {stats.pooled_items}   ({n_manual} manual seeds retained)")
    print(f"  raw -> deduped:     {stats.raw_items} -> {stats.pooled_items}   "
          f"({reduction:.2f}x, {removed_pct:.0f}% removed)")
    print(f"    exact (paper,char) dups: {stats.exact_dups_removed}")
    print(f"    overlap-merged (>80%):   {stats.overlap_merged}")
    print(f"  items/question:     min {min(per_q)} | median {statistics.median(per_q):g} | "
          f"p95 {_percentile(per_q, 95):g} | max {max(per_q)}")
    print(f"  by question type:   {n_answerable_items} items across answerable q  |  "
          f"{n_negative_items} across negatives")
    print(f"  judgments to make:  ~{stats.pooled_items - n_manual}  "
          f"(manual seeds already ground truth, auto-skipped)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
