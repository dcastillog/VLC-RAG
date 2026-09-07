"""Stage 3: embedding, Qdrant collections, and (Phase D) search.

Phase C surface:

* :class:`~ragvlc.retrieval.embed.Embedder` -- dense + sparse embedding, with
  the BGE query/document prefix asymmetry confined to one method.
* :func:`~ragvlc.retrieval.collections.ensure_collection` -- create the
  ``dense``/``sparse`` named-vector collections (sparse carries
  ``Modifier.IDF``) and their payload indexes.
* :func:`~ragvlc.retrieval.ingest.ingest` -- embed a chunk set and upsert it.
"""

from __future__ import annotations

from ragvlc.retrieval.collections import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    ensure_collection,
)
from ragvlc.retrieval.embed import Embedder, Embedding, SparseVec
from ragvlc.retrieval.ingest import IngestStats, ingest, load_chunk_records, point_id
from ragvlc.retrieval.payload import PAYLOAD_FIELDS, build_payload
from ragvlc.retrieval.search import (
    SEARCH_MODES,
    Hit,
    Searcher,
    SearchMode,
    Timings,
    build_filter,
)

__all__ = [
    "DENSE_VECTOR",
    "SPARSE_VECTOR",
    "ensure_collection",
    "Embedder",
    "Embedding",
    "SparseVec",
    "IngestStats",
    "ingest",
    "load_chunk_records",
    "point_id",
    "PAYLOAD_FIELDS",
    "build_payload",
    "Searcher",
    "SearchMode",
    "SEARCH_MODES",
    "Hit",
    "Timings",
    "build_filter",
]
