"""Embed a chunk set and upsert it into its Qdrant collection.

Library side of ``scripts/ingest.py``: it returns counts, the script decides
what to print. Point ids are ``uuid5(chunk_id)`` -- deterministic, so
re-running without ``--recreate`` overwrites points in place rather than
duplicating them.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from qdrant_client import QdrantClient, models

from ragvlc.chunking import Paper
from ragvlc.retrieval.collections import DENSE_VECTOR, SPARSE_VECTOR
from ragvlc.retrieval.embed import Embedder
from ragvlc.retrieval.payload import build_payload, overlapping_units

# Fixed namespace so a chunk_id always maps to the same point id across runs
# and machines.
_POINT_NAMESPACE = uuid.UUID("6f1b2e2a-0d1e-4c8b-9a3a-2c7d5e9f00c3")


@dataclass
class IngestStats:
    collection: str
    chunker: str
    n_chunks: int = 0
    n_upserted: int = 0
    chunks_without_unit: list[str] = field(default_factory=list)  # chunk_ids with no overlapping unit


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


def load_chunk_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _batches(records: list[dict], size: int) -> Iterator[list[dict]]:
    for start in range(0, len(records), size):
        yield records[start : start + size]


def ingest(
    client: QdrantClient,
    collection: str,
    records: list[dict],
    papers: dict[str, Paper],
    chunker: str,
    embedder: Embedder,
    *,
    batch_size: int = 128,
    on_progress: Callable[[int, int], None] = lambda done, total: None,
) -> IngestStats:
    """Embed every record's text and upsert it. ``papers`` must contain every
    ``paper_id`` referenced by ``records``."""
    stats = IngestStats(collection=collection, chunker=chunker, n_chunks=len(records))

    for batch in _batches(records, batch_size):
        embeddings = embedder.embed_documents([r["text"] for r in batch], batch_size=batch_size)
        points: list[models.PointStruct] = []
        for record, embedding in zip(batch, embeddings):
            paper = papers[record["paper_id"]]
            payload = build_payload(record, paper, chunker)
            if not overlapping_units(paper.units, record["char_start"], record["char_end"]):
                # No unit overlaps this chunk -- should not happen for chunks
                # derived from unit text, but record it rather than hide it.
                stats.chunks_without_unit.append(record["chunk_id"])
            points.append(
                models.PointStruct(
                    id=point_id(record["chunk_id"]),
                    vector={
                        DENSE_VECTOR: embedding.dense,
                        SPARSE_VECTOR: models.SparseVector(
                            indices=embedding.sparse.indices,
                            values=embedding.sparse.values,
                        ),
                    },
                    payload=payload,
                )
            )
        client.upsert(collection_name=collection, points=points, wait=True)
        stats.n_upserted += len(points)
        on_progress(stats.n_upserted, stats.n_chunks)

    return stats
