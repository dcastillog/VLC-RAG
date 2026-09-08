"""Phase D: the retrieval entry point.

Four modes behind one call (:meth:`Searcher.search`):

* ``dense``       -- bge-small vectors, cosine distance (exact at this corpus size)
* ``sparse``      -- BM25, with inverse document frequency applied by Qdrant
* ``hybrid_rrf``  -- Reciprocal Rank Fusion of dense + sparse
* ``hybrid_dbsf`` -- Distribution-Based Score Fusion of dense + sparse

Hybrid modes use Qdrant's Query API: two ``prefetch`` branches, each returning
``prefetch_limit`` candidates, combined by a fusion query. **Fusion only
reorders the union of what the branches returned** -- a chunk that neither
branch places in its top ``prefetch_limit`` cannot be retrieved no matter how
large ``limit`` is. So ``prefetch_limit``, not ``limit``, is the recall
ceiling for hybrid retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal, get_args

from qdrant_client import QdrantClient, models

from ragvlc.config import get_experiment, get_settings
from ragvlc.retrieval.collections import DENSE_VECTOR, SPARSE_VECTOR
from ragvlc.retrieval.embed import Embedder

SearchMode = Literal["dense", "sparse", "hybrid_rrf", "hybrid_dbsf"]
SEARCH_MODES: tuple[SearchMode, ...] = get_args(SearchMode)

_HYBRID_MODES = ("hybrid_rrf", "hybrid_dbsf")


@dataclass(frozen=True)
class Hit:
    point_id: str
    score: float
    payload: dict  # the full chunk payload (see ragvlc.retrieval.payload.PAYLOAD_FIELDS)


@dataclass(frozen=True)
class Timings:
    """Wall-clock milliseconds for one :meth:`Searcher.search` call.

    ``dense_embed`` / ``sparse_embed`` are 0.0 for a mode that does not use
    that vector -- the work is genuinely skipped, not just untimed.
    """

    dense_embed: float
    sparse_embed: float
    qdrant_search: float
    total: float


def _ms(since: float) -> float:
    return (perf_counter() - since) * 1000.0


def build_filter(
    *,
    paper_id: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    venue: str | None = None,
    section_type: str | None = None,
) -> models.Filter | None:
    """Assemble a Qdrant filter from the payload fields we index. Returns
    ``None`` when nothing is constrained (so callers can pass it straight
    through)."""
    must: list[models.FieldCondition] = []
    if paper_id is not None:
        must.append(models.FieldCondition(key="paper_id", match=models.MatchValue(value=paper_id)))
    if year_min is not None or year_max is not None:
        must.append(
            models.FieldCondition(key="year", range=models.Range(gte=year_min, lte=year_max))
        )
    if venue is not None:
        must.append(models.FieldCondition(key="venue", match=models.MatchValue(value=venue)))
    if section_type is not None:
        must.append(
            models.FieldCondition(key="section_type", match=models.MatchValue(value=section_type))
        )
    return models.Filter(must=must) if must else None


class Searcher:
    """Holds the Qdrant client and embedding models so the hot path (Phase E
    runs ~650 searches, Phase G many more) reuses one of each."""

    def __init__(
        self,
        client: QdrantClient,
        embedder: Embedder,
        *,
        prefetch_limit: int = 50,
        rrf_k: int = 2,
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._prefetch_limit = prefetch_limit
        self._rrf_k = rrf_k

    @classmethod
    def from_config(cls) -> Searcher:
        settings = get_settings()
        retrieval = get_experiment().retrieval
        return cls(
            QdrantClient(url=settings.qdrant_url),
            Embedder(retrieval.dense_model, retrieval.sparse_model, retrieval.query_prefix),
            prefetch_limit=retrieval.prefetch_limit,
            rrf_k=retrieval.rrf_k,
        )

    # ------------------------------------------------------------------ #
    def search(
        self,
        collection: str,
        query: str,
        mode: SearchMode,
        limit: int = 10,
        filters: models.Filter | None = None,
        *,
        rrf_k: int | None = None,
    ) -> tuple[list[Hit], Timings]:
        """``rrf_k`` overrides the instance's RRF rank constant for this one
        call (mode ``hybrid_rrf`` only) -- Phase G sweeps it without needing a
        new ``Searcher`` per value."""
        if mode not in SEARCH_MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {SEARCH_MODES}")

        started = perf_counter()
        dense_ms = sparse_ms = 0.0
        dense_vec: list[float] | None = None
        sparse_vec: models.SparseVector | None = None

        if mode == "dense" or mode in _HYBRID_MODES:
            embed_started = perf_counter()
            dense_vec = self._embedder.embed_query_dense(query)
            dense_ms = _ms(embed_started)
        if mode == "sparse" or mode in _HYBRID_MODES:
            embed_started = perf_counter()
            raw = self._embedder.embed_query_sparse(query)
            sparse_vec = models.SparseVector(indices=raw.indices, values=raw.values)
            sparse_ms = _ms(embed_started)

        search_started = perf_counter()
        if mode == "dense":
            response = self._client.query_points(
                collection, query=dense_vec, using=DENSE_VECTOR,
                limit=limit, query_filter=filters, with_payload=True,
            )
        elif mode == "sparse":
            response = self._client.query_points(
                collection, query=sparse_vec, using=SPARSE_VECTOR,
                limit=limit, query_filter=filters, with_payload=True,
            )
        else:
            # filter goes on each prefetch branch: that is where it bounds the
            # candidate set (and therefore recall), not on the fusion step.
            prefetch = [
                models.Prefetch(query=dense_vec, using=DENSE_VECTOR, limit=self._prefetch_limit, filter=filters),
                models.Prefetch(query=sparse_vec, using=SPARSE_VECTOR, limit=self._prefetch_limit, filter=filters),
            ]
            if mode == "hybrid_rrf":
                # RrfQuery, not FusionQuery(RRF), so k is configurable. Qdrant's
                # default k is 2; the original RRF paper uses 60. Phase G sweeps it.
                fusion = models.RrfQuery(rrf=models.Rrf(k=rrf_k if rrf_k is not None else self._rrf_k))
            else:
                fusion = models.FusionQuery(fusion=models.Fusion.DBSF)
            response = self._client.query_points(
                collection, prefetch=prefetch, query=fusion, limit=limit, with_payload=True,
            )
        qdrant_ms = _ms(search_started)

        hits = [
            Hit(point_id=str(point.id), score=float(point.score), payload=dict(point.payload or {}))
            for point in response.points
        ]
        # Score ties are common (RRF scores are a small set of rational
        # numbers; exact-search cosine ties happen too) and Qdrant does not
        # promise a stable order among them -- it can vary run to run with the
        # same data (segment iteration order, thread scheduling). Re-sorting
        # by (-score, chunk_id) makes the result order a pure function of the
        # collection's contents, not of how Qdrant happened to merge segments
        # this time -- required for run_eval.py to be reproducible.
        hits.sort(key=lambda h: (-h.score, h.payload.get("chunk_id", "")))
        timings = Timings(
            dense_embed=dense_ms,
            sparse_embed=sparse_ms,
            qdrant_search=qdrant_ms,
            total=_ms(started),
        )
        return hits, timings
