"""Qdrant collection layout for the two chunk sets.

Both collections have the same shape -- one named dense vector and one named
sparse vector -- so a retrieval mode written against ``vlc_fixed`` runs
unchanged against ``vlc_section``. That symmetry is the whole point: it keeps
the chunker comparison from being confounded by collection differences.
"""

from __future__ import annotations

from qdrant_client import QdrantClient, models

# Named-vector keys. Referenced by ingestion and by every retrieval mode, so
# they live here as constants rather than as string literals scattered around.
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

DENSE_DISTANCE = models.Distance.COSINE  # bge-small vectors are meant to be compared by cosine

# Payload fields we filter on in Phase D/G. Qdrant can filter without an index
# (linear scan), but an explicit index is what makes ``year >= 2022`` cheap --
# and Phase G measures exactly that difference.
_PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    "paper_id": models.PayloadSchemaType.KEYWORD,
    "venue": models.PayloadSchemaType.KEYWORD,
    "section_type": models.PayloadSchemaType.KEYWORD,
    "year": models.PayloadSchemaType.INTEGER,
}


def ensure_collection(
    client: QdrantClient,
    name: str,
    dense_dim: int,
    *,
    recreate: bool,
    indexing_threshold_kb: int | None = None,
) -> bool:
    """Make sure collection ``name`` exists with the right vectors and payload
    indexes. Returns ``True`` if it was (re)created, ``False`` if left as-is.

    ``recreate`` drops any existing collection first -- use it when the chunk
    set or the embedding model changed. Without it, an existing collection is
    kept and ingestion upserts into it.
    """
    exists = client.collection_exists(name)
    if exists and not recreate:
        return False
    if exists:
        client.delete_collection(name)

    optimizers = None
    if indexing_threshold_kb is not None:
        optimizers = models.OptimizersConfigDiff(indexing_threshold=indexing_threshold_kb)

    client.create_collection(
        collection_name=name,
        vectors_config={
            DENSE_VECTOR: models.VectorParams(size=dense_dim, distance=DENSE_DISTANCE),
        },
        sparse_vectors_config={
            # modifier=IDF makes Qdrant compute inverse document frequency
            # server-side over this collection. FastEmbed's Qdrant/bm25 only
            # emits term-frequency weights; without IDF here, ranking would
            # ignore how common a term is across the corpus -- i.e. it would
            # not actually be BM25 -- and retrieval quality would drop with
            # nothing in the query path to explain it.
            SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF),
        },
        optimizers_config=optimizers,
    )

    for field_name, schema in _PAYLOAD_INDEXES.items():
        client.create_payload_index(name, field_name=field_name, field_schema=schema)

    return True
