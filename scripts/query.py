"""CLI: run one retrieval query and print the ranked chunks, for eyeballing
before Phase E pools anything.

    uv run python scripts/query.py "how does RMS delay spread behave in mining VLC?"
    uv run python scripts/query.py "color shift keying" --mode sparse --collection fixed
    uv run python scripts/query.py "IRS placement optimisation" --mode hybrid_dbsf --year-min 2023

``--mode`` is one of dense | sparse | hybrid_rrf | hybrid_dbsf. ``--collection``
takes the alias ``fixed`` or ``section``. The per-stage timing line at the end
is the same ``Timings`` object the eval runner records.
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from ragvlc.config import get_experiment
from ragvlc.retrieval import SEARCH_MODES, Searcher, build_filter

_COLLECTION_ALIASES = {"fixed": "fixed", "section": "section_aware"}
_SNIPPET_CHARS = 240


def _snippet(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) > _SNIPPET_CHARS:
        collapsed = collapsed[:_SNIPPET_CHARS].rstrip() + "..."
    return collapsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one retrieval query against Qdrant.")
    parser.add_argument("query", help="the query text")
    parser.add_argument("--mode", choices=SEARCH_MODES, default="hybrid_rrf")
    parser.add_argument("--collection", choices=list(_COLLECTION_ALIASES), default="section")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--year-min", type=int, default=None)
    parser.add_argument("--year-max", type=int, default=None)
    parser.add_argument("--venue", default=None)
    parser.add_argument("--section-type", default=None)
    args = parser.parse_args(argv)

    chunker = _COLLECTION_ALIASES[args.collection]
    collection = get_experiment().retrieval.qdrant.collections[chunker]

    filters = build_filter(
        year_min=args.year_min,
        year_max=args.year_max,
        venue=args.venue,
        section_type=args.section_type,
    )
    filter_desc = ", ".join(
        f"{name}={value}"
        for name, value in (
            ("year_min", args.year_min), ("year_max", args.year_max),
            ("venue", args.venue), ("section_type", args.section_type),
        )
        if value is not None
    )

    searcher = Searcher.from_config()
    try:
        hits, timings = searcher.search(collection, args.query, args.mode, limit=args.limit, filters=filters)
    except Exception as exc:  # broad on purpose: turn any Qdrant/embedding failure into one line
        print(f"query.py: search failed: {exc}", file=sys.stderr)
        return 1

    print(f'query:      {args.query!r}')
    print(f"mode:       {args.mode}   collection: {collection}" + (f"   filter: {filter_desc}" if filter_desc else ""))
    print()
    if not hits:
        print("(no results)")
        return 0

    for rank, hit in enumerate(hits, start=1):
        p = hit.payload
        heading = p.get("section_heading") or p.get("parent_heading") or "-"
        print(f"[{rank:>2}] score={hit.score:.4f}  {p.get('paper_id')}  ({p.get('year')})")
        print(f"     {p.get('title')}")
        print(f"     section: {heading}   chars [{p.get('char_start')}:{p.get('char_end')}]  n_tokens={p.get('n_tokens')}")
        for line in textwrap.wrap(_snippet(p.get("text", "")), width=100):
            print(f"     {line}")
        print()

    print(
        f"timings (ms): dense_embed={timings.dense_embed:.1f}  sparse_embed={timings.sparse_embed:.1f}  "
        f"qdrant_search={timings.qdrant_search:.1f}  total={timings.total:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
