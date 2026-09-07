"""CLI: embed a chunk set and load it into its Qdrant collection.

    uv run python scripts/ingest.py                 # both chunkers
    uv run python scripts/ingest.py --chunker fixed
    uv run python scripts/ingest.py --recreate      # drop + rebuild first

Reads ``data/chunks/{chunker}.jsonl`` (from scripts/build_chunks.py), joins
each chunk back to its paper's normalized JSON for the payload, embeds dense
(bge-small, no prefix) and sparse (bm25), and upserts. Afterwards it prints
``collection_info``.

The dense HNSW index is **not** built at this corpus size, so dense search is
exact (brute force), not approximate. Qdrant only builds a dense HNSW segment
once the dense vector data passes ``indexing_threshold`` (10 MB on this
server); ~4.7k chunks x 384 dims x 4 bytes is a few MB, well under it. This is
expected -- Phase G measures what forcing HNSW would cost here.

Note that ``collection_info.indexed_vectors_count`` is still non-zero: it
equals ``points_count`` because the *sparse* / BM25 inverted index is always
fully built. It is not evidence of a dense HNSW index -- the per-report line
below computes the dense situation directly from the vector count.
"""

from __future__ import annotations

import argparse
import sys

from qdrant_client import QdrantClient

from ragvlc.chunking import Paper
from ragvlc.config import get_experiment, get_paths, get_settings
from ragvlc.retrieval import Embedder, ensure_collection, ingest, load_chunk_records

# CLI alias -> chunker name (used for the chunk filename, chunk_id, and payload).
_CHUNKER_ALIASES = {"fixed": "fixed", "section": "section_aware"}


def _progress(done: int, total: int) -> None:
    print(f"    {done:>5}/{total} upserted", end="\r", flush=True)


def _load_papers(paper_ids: set[str], normalized_dir) -> dict[str, Paper]:
    papers: dict[str, Paper] = {}
    for paper_id in sorted(paper_ids):
        json_path = normalized_dir / f"{paper_id}.json"
        if not json_path.is_file():
            raise FileNotFoundError(f"chunk references {paper_id!r} but {json_path} is missing")
        papers[paper_id] = Paper.from_normalized(json_path)
    return papers


def _report_collection(client: QdrantClient, name: str, n_points: int, dense_dim: int) -> None:
    info = client.get_collection(name)
    threshold_kb = info.config.optimizer_config.indexing_threshold
    dense_kb = n_points * dense_dim * 4 / 1024

    print(f"  collection_info[{name}]:")
    print(f"    status:                 {info.status}")
    print(f"    points_count:           {info.points_count}")
    print(f"    indexed_vectors_count:  {info.indexed_vectors_count}  (sparse/BM25 index; always on)")
    print(
        f"    dense vectors:          {n_points} x {dense_dim}d ~= {dense_kb:,.0f} KB"
        + (f" vs indexing_threshold {threshold_kb:,} KB" if threshold_kb is not None else "")
    )
    if threshold_kb is None or dense_kb < threshold_kb:
        print(
            "      -> dense HNSW NOT built: dense data is below indexing_threshold, so dense\n"
            "         search is EXACT (brute force). Expected at this size; Phase G measures\n"
            "         what forcing HNSW would cost."
        )
    else:
        print("      -> dense data exceeds indexing_threshold: dense HNSW is being built.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Embed and ingest a chunk set into Qdrant.")
    parser.add_argument("--chunker", choices=["fixed", "section", "all"], default="all")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="drop each target collection and rebuild it before ingesting",
    )
    args = parser.parse_args(argv)

    experiment = get_experiment()
    retrieval = experiment.retrieval
    paths = get_paths()
    settings = get_settings()

    aliases = list(_CHUNKER_ALIASES) if args.chunker == "all" else [args.chunker]
    chunkers = [_CHUNKER_ALIASES[a] for a in aliases]

    chunks_dir = paths.root / "data" / "chunks"
    missing = [c for c in chunkers if not (chunks_dir / f"{c}.jsonl").is_file()]
    if missing:
        print(
            f"ingest.py: no chunk file for {missing} -- run scripts/build_chunks.py first",
            file=sys.stderr,
        )
        return 1

    client = QdrantClient(url=settings.qdrant_url)
    try:
        client.get_collections()
    except Exception as exc:  # broad on purpose: a preflight that turns any failure into a hint
        print(
            f"ingest.py: cannot reach Qdrant at {settings.qdrant_url}: {exc}\n"
            "  start it with:  docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant",
            file=sys.stderr,
        )
        return 1

    embedder = Embedder(
        dense_model=retrieval.dense_model,
        sparse_model=retrieval.sparse_model,
        query_prefix=retrieval.query_prefix,
    )

    for chunker in chunkers:
        collection = retrieval.qdrant.collections[chunker]
        print(f"[{chunker} -> {collection}]")

        records = load_chunk_records(chunks_dir / f"{chunker}.jsonl")
        papers = _load_papers({r["paper_id"] for r in records}, paths.normalized)

        created = ensure_collection(
            client,
            collection,
            embedder.dense_dim,
            recreate=args.recreate,
            indexing_threshold_kb=retrieval.qdrant.indexing_threshold_kb,
        )
        print(f"  collection {'created' if created else 'kept (upserting into existing)'}")
        print(f"  embedding + upserting {len(records)} chunk(s) from {len(papers)} paper(s)...")

        stats = ingest(
            client,
            collection,
            records,
            papers,
            chunker,
            embedder,
            batch_size=retrieval.qdrant.upsert_batch_size,
            on_progress=_progress,
        )
        print(f"    {stats.n_upserted:>5}/{stats.n_chunks} upserted")
        if stats.chunks_without_unit:
            print(
                f"  WARNING: {len(stats.chunks_without_unit)} chunk(s) overlapped no unit "
                f"(payload section fields null): {stats.chunks_without_unit[:5]}...",
                file=sys.stderr,
            )

        _report_collection(client, collection, stats.n_upserted, embedder.dense_dim)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
