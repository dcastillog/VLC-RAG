"""Tests for Phase D retrieval.

``build_filter`` and mode validation are pure and always run. The four search
modes are exercised against the *real* ingested collections (mocking a vector
database only tests the mock); those tests skip cleanly when Qdrant is not up
or the collections have not been ingested.
"""

from __future__ import annotations

import pytest

from ragvlc.config import get_experiment
from ragvlc.retrieval import SEARCH_MODES, Searcher, build_filter

# --------------------------------------------------------------------------- #
# Pure
# --------------------------------------------------------------------------- #
def test_build_filter_returns_none_when_unconstrained():
    assert build_filter() is None


def test_build_filter_builds_one_condition_per_constraint():
    f = build_filter(year_min=2022, year_max=2025, venue="Sensors", section_type="methods")
    assert f is not None
    keys = [c.key for c in f.must]
    assert keys == ["year", "venue", "section_type"]  # year_min/year_max collapse to one range
    year = next(c for c in f.must if c.key == "year")
    assert year.range.gte == 2022 and year.range.lte == 2025


def test_search_rejects_unknown_mode():
    # fails before any client/embedder work, so None deps are fine
    with pytest.raises(ValueError, match="unknown mode"):
        Searcher(None, None).search("vlc_section", "q", "bm25")


class _FakePoint:
    def __init__(self, point_id: str, score: float, payload: dict) -> None:
        self.id = point_id
        self.score = score
        self.payload = payload


class _FakeResponse:
    def __init__(self, points: list[_FakePoint]) -> None:
        self.points = points


class _FakeClient:
    """A plain stub of the one method `search` calls -- not a mock of Qdrant
    itself, just fixed data to prove the tie-break sort runs regardless of
    what order the points arrived in."""

    def __init__(self, points: list[_FakePoint]) -> None:
        self._points = points

    def query_points(self, *args, **kwargs):  # noqa: ARG002
        return _FakeResponse(self._points)


class _FakeEmbedder:
    def embed_query_dense(self, text: str) -> list[float]:  # noqa: ARG002
        return [0.0]


def test_search_breaks_score_ties_deterministically_by_chunk_id():
    """Qdrant does not promise a stable order among tied scores -- it can
    vary run to run with the same data. Feed points back tied and
    out-of-chunk_id-order; `search` must still return them sorted by
    (-score, chunk_id), never Qdrant's arrival order."""
    points = [
        _FakePoint("id-b", 0.9, {"chunk_id": "paper:fixed:0002"}),
        _FakePoint("id-a", 0.9, {"chunk_id": "paper:fixed:0001"}),  # tied with id-b, arrives second
        _FakePoint("id-c", 0.5, {"chunk_id": "paper:fixed:0000"}),  # lower score but "smaller" chunk_id
    ]
    searcher = Searcher(_FakeClient(points), _FakeEmbedder())
    hits, _ = searcher.search("coll", "q", "dense")
    assert [h.payload["chunk_id"] for h in hits] == [
        "paper:fixed:0001",  # tied at 0.9, wins the chunk_id tiebreak
        "paper:fixed:0002",  # tied at 0.9
        "paper:fixed:0000",  # lower score always loses regardless of chunk_id
    ]


# --------------------------------------------------------------------------- #
# Live -- needs Qdrant + ingested collections
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def live_searcher() -> Searcher:
    from qdrant_client import QdrantClient

    collections = get_experiment().retrieval.qdrant.collections
    try:
        probe = QdrantClient(url="http://localhost:6333", timeout=2)
        missing = [name for name in collections.values() if not probe.collection_exists(name)]
    except Exception as exc:  # any connection failure means "skip", not "fail"
        pytest.skip(f"Qdrant not reachable: {exc}")
    if missing:
        pytest.skip(f"collections not ingested: {missing} (run scripts/ingest.py)")
    return Searcher.from_config()


_QUERY = "RMS delay spread in underground mining visible light communication channels"


@pytest.mark.parametrize("mode", SEARCH_MODES)
def test_every_mode_returns_ranked_hits(live_searcher: Searcher, mode: str):
    collection = get_experiment().retrieval.qdrant.collections["section_aware"]
    hits, timings = live_searcher.search(collection, _QUERY, mode, limit=10)

    assert 0 < len(hits) <= 10
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    for hit in hits:
        assert {"paper_id", "char_start", "char_end", "text"} <= hit.payload.keys()

    assert timings.total >= timings.qdrant_search > 0
    assert (timings.dense_embed > 0) == (mode in ("dense", "hybrid_rrf", "hybrid_dbsf"))
    assert (timings.sparse_embed > 0) == (mode in ("sparse", "hybrid_rrf", "hybrid_dbsf"))


def test_filter_restricts_results(live_searcher: Searcher):
    collection = get_experiment().retrieval.qdrant.collections["section_aware"]
    hits, _ = live_searcher.search(
        collection, _QUERY, "dense", limit=10, filters=build_filter(year_min=2023)
    )
    assert hits
    assert all(h.payload["year"] >= 2023 for h in hits)


def test_prefetch_limit_bounds_hybrid_recall(live_searcher: Searcher):
    """A hybrid search cannot return more than 2 * prefetch_limit distinct
    chunks, however large `limit` is -- fusion only reorders the prefetch
    union."""
    collection = get_experiment().retrieval.qdrant.collections["fixed"]
    tight = Searcher(live_searcher._client, live_searcher._embedder, prefetch_limit=3, rrf_k=2)
    hits, _ = tight.search(collection, _QUERY, "hybrid_rrf", limit=50)
    assert len(hits) <= 6
