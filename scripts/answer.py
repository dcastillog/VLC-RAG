"""CLI: ask one grounded question and print the answer, its citations, and timings.

    uv run python scripts/answer.py "What limits the data rate of OCC compared to photodiode-based VLC?"
    uv run python scripts/answer.py "color shift keying receivers" --mode hybrid_rrf --collection fixed
    uv run python scripts/answer.py "IRS placement optimisation" --top-k 8

``--mode`` is one of dense | sparse | hybrid_rrf | hybrid_dbsf (default
``hybrid_dbsf`` -- the best MRR@10 in the retrieval eval). ``--collection`` takes
the alias ``fixed`` or ``section``. ``--top-k`` defaults to ``generation.top_k``
in config/default.yaml.

Needs Qdrant up with the collections ingested, and an OpenAI-compatible LLM
endpoint reachable at ``LLM_BASE_URL`` serving ``LLM_MODEL`` (see .env.example).
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from ragvlc.config import get_experiment
from ragvlc.generation import GenerationError, LLMClient, answer
from ragvlc.retrieval import SEARCH_MODES, Searcher

_COLLECTION_ALIASES = {"fixed": "fixed", "section": "section_aware"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask one grounded question against the corpus.")
    parser.add_argument("query", help="the question")
    parser.add_argument("--mode", choices=SEARCH_MODES, default="hybrid_dbsf")
    parser.add_argument("--collection", choices=list(_COLLECTION_ALIASES), default="section")
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="chunks to retrieve and cite (default: generation.top_k in config)",
    )
    args = parser.parse_args(argv)

    experiment = get_experiment()
    chunker = _COLLECTION_ALIASES[args.collection]
    collection = experiment.retrieval.qdrant.collections[chunker]
    top_k = args.top_k if args.top_k is not None else experiment.generation.top_k

    searcher = Searcher.from_config()
    client = LLMClient.from_config()

    try:
        result = answer(args.query, collection, args.mode, top_k, searcher=searcher, client=client)
    except GenerationError as exc:
        print(f"answer.py: generation failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # broad on purpose: one line for any Qdrant/embedding failure
        print(f"answer.py: retrieval failed: {exc}", file=sys.stderr)
        return 1

    print(f"query:      {args.query!r}")
    print(
        f"mode:       {args.mode}   collection: {collection}   top_k: {top_k}   "
        f"model: {client.model}"
    )
    print(f"abstained:  {result.abstained}")
    print()
    for line in textwrap.wrap(result.text, width=100) or [""]:
        print(line)
    print()

    if result.citations:
        print("citations:")
        for c in result.citations:
            doi = c.doi or "(no DOI)"
            print(f"  [{c.index}] {c.paper_id}  chars [{c.char_start}:{c.char_end}]  {doi}")
            print(f"       {c.title or 'Untitled'}")
    else:
        print("citations:  (none)")

    if result.invalid_citations:
        print(f"invalid citation indices (no such source block): {list(result.invalid_citations)}")

    t = result.timings
    print()
    print(
        "timings (ms): "
        f"dense_embed={t['dense_embed_ms']:.1f}  sparse_embed={t['sparse_embed_ms']:.1f}  "
        f"qdrant_search={t['qdrant_search_ms']:.1f}  retrieval_total={t['retrieval_total_ms']:.1f}  "
        f"generation={t['generation_ms']:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
