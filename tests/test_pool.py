"""Tests for pool construction -- the dedup rules that decide the judging
burden and, through it, the ground truth every later metric rests on.

``build_pool`` is exercised with a stub searcher (a plain function boundary in
our own code -- not the vector DB), so the retrieval half is fixed and the
pooling logic is what's under test.
"""

from __future__ import annotations

from ragvlc.eval.pool import PoolItem, _dedupe_question, build_pool
from ragvlc.retrieval import Hit, Timings


def _item(qid: str, start: int, end: int, *, configs: set[str], seed: str = "pooled") -> PoolItem:
    return PoolItem(
        qid=qid,
        paper_id="p1",
        char_start=start,
        char_end=end,
        text=f"chars {start}..{end}",
        section_heading="S",
        configs=set(configs),
        seed=seed,
    )


# --------------------------------------------------------------------------- #
# dedup
# --------------------------------------------------------------------------- #
def test_exact_duplicates_collapse_and_union_configs():
    raw = [
        _item("q1", 100, 500, configs={"fixed/dense"}),
        _item("q1", 100, 500, configs={"section_aware/sparse"}),
        _item("q1", 100, 500, configs={"fixed/hybrid_rrf"}),
    ]
    kept, exact_removed, overlap_merged = _dedupe_question(raw)
    assert len(kept) == 1
    assert exact_removed == 2
    assert overlap_merged == 0
    assert kept[0].configs == {"fixed/dense", "section_aware/sparse", "fixed/hybrid_rrf"}


def test_overlap_above_80pct_keeps_only_the_shorter():
    raw = [
        _item("q1", 100, 1100, configs={"fixed/dense"}),          # 1000 chars
        _item("q1", 150, 1050, configs={"section_aware/dense"}),  # 900 chars, wholly inside
    ]
    kept, _, overlap_merged = _dedupe_question(raw)
    assert len(kept) == 1
    assert kept[0].length == 900  # the tighter span survives
    assert overlap_merged == 1
    assert kept[0].configs == {"fixed/dense", "section_aware/dense"}


def test_overlap_below_threshold_keeps_both():
    # shorter is 400 chars, intersection is 300 -> 0.75 < 0.80
    raw = [
        _item("q1", 0, 500, configs={"fixed/dense"}),
        _item("q1", 200, 600, configs={"fixed/sparse"}),
    ]
    kept, _, overlap_merged = _dedupe_question(raw)
    assert len(kept) == 2
    assert overlap_merged == 0


def test_manual_seed_survives_when_shorter_than_a_covering_chunk():
    raw = [
        _item("q1", 500, 700, configs={"manual"}, seed="manual"),   # 200 chars
        _item("q1", 0, 4000, configs={"fixed/dense"}),              # covers the manual span 100%
    ]
    kept, _, overlap_merged = _dedupe_question(raw)
    assert len(kept) == 1
    assert kept[0].seed == "manual"
    assert kept[0].length == 200
    assert kept[0].configs == {"manual", "fixed/dense"}
    assert overlap_merged == 1


def test_manual_seed_wins_even_when_it_is_the_longer_item():
    raw = [
        _item("q1", 0, 4000, configs={"manual"}, seed="manual"),   # long, but manual
        _item("q1", 500, 700, configs={"fixed/dense"}),            # shorter, wholly inside
    ]
    kept, _, overlap_merged = _dedupe_question(raw)
    assert len(kept) == 1
    assert kept[0].seed == "manual"
    assert kept[0].length == 4000
    assert overlap_merged == 1


# --------------------------------------------------------------------------- #
# build_pool
# --------------------------------------------------------------------------- #
class _StubSearcher:
    """Returns two fixed hits per call and records the filter it was passed."""

    def __init__(self) -> None:
        self.filters_seen: list[object] = []

    def search(self, collection, query, mode, limit=10, filters=None):  # noqa: ARG002
        self.filters_seen.append(filters)
        payload_a = {"paper_id": "p1", "char_start": 0, "char_end": 400, "text": "A", "section_heading": "Intro"}
        payload_b = {"paper_id": "p1", "char_start": 1000, "char_end": 1400, "text": "B", "section_heading": "Method"}
        hits = [Hit("id-a", 0.9, payload_a), Hit("id-b", 0.8, payload_b)]
        return hits, Timings(0.0, 0.0, 0.0, 0.0)


def test_build_pool_dedupes_across_configs_and_seeds_manual_spans():
    questions = [
        {
            "qid": "q1",
            "question": "what?",
            "paper_id": "p1",
            "has_answer": True,
            "gold_spans": [
                {"text": "manual", "char_start": 5000, "char_end": 5100,
                 "section_heading": "Results", "source": "manual"},
            ],
        }
    ]
    searcher = _StubSearcher()
    pool, stats = build_pool(searcher, questions, {"fixed": "c_fixed", "section_aware": "c_sec"}, top_k=10)

    # 8 configs x 2 identical hits -> 2 unique spans, + 1 manual seed
    assert stats.raw_items == 8 * 2 + 1
    assert stats.pooled_items == 3
    assert stats.exact_dups_removed == 14
    by_span = {(it.char_start, it.char_end): it for it in pool}
    assert set(by_span) == {(0, 400), (1000, 1400), (5000, 5100)}
    assert by_span[(5000, 5100)].seed == "manual"
    assert len(by_span[(0, 400)].configs) == 8  # every config retrieved it

    # answerable question -> every search was filtered (to the source paper)
    assert all(f is not None for f in searcher.filters_seen)


def test_build_pool_does_not_filter_negative_questions():
    questions = [{"qid": "n1", "question": "no answer", "paper_id": None, "has_answer": False, "gold_spans": []}]
    searcher = _StubSearcher()
    build_pool(searcher, questions, {"fixed": "c_fixed", "section_aware": "c_sec"}, top_k=10)
    assert all(f is None for f in searcher.filters_seen)
