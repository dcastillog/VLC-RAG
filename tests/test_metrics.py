"""Tests for the relevance rule (ragvlc.eval.relevance) and the per-question
retrieval metrics built on it (ragvlc.eval.metrics): MRR (the primary metric),
success@k, and recall@k.
"""

from __future__ import annotations

import pytest

from ragvlc.eval.metrics import reciprocal_rank, recall_at_k, success_at_k
from ragvlc.eval.relevance import chunk_matches_any, is_relevant, relevance_ratio

# A gold span occupying characters [1000, 2000) -- length 1000.
GOLD_START, GOLD_END = 1000, 2000


@pytest.mark.parametrize(
    ("chunk_start", "chunk_end", "expected_ratio", "expected_relevant"),
    [
        (1000, 2000, 1.0, True),    # exact
        (1000, 3000, 1.0, True),    # chunk covers 100% of the span (span is shorter)
        (1400, 2000, 1.0, True),    # 600-char chunk fully inside the span -> 600/600
        (1000, 1600, 1.0, True),    # 600-char chunk, all inside -> 1.0
        (1000, 1500, 1.0, True),    # 500 overlap / min(500, 1000) = 1.0
        (400, 1500, 500 / 1000, True),   # 1100-char chunk, 500 overlap, shorter is the span
        (1600, 2000, 1.0, True),    # 400 in / 400
        (1400, 2400, 600 / 1000, True),  # 1000-char chunk, 600 overlap
        (1800, 2200, 200 / 400, True),   # 400-char chunk, 200 overlap -> 0.5 exactly
        (1850, 2200, 150 / 350, False),  # 350-char chunk, 150 overlap -> ~0.43
        (0, 1000, 0.0, False),      # touches at the boundary, no overlap
        (0, 500, 0.0, False),       # disjoint
    ],
)
def test_symmetric_overlap_rule(chunk_start, chunk_end, expected_ratio, expected_relevant):
    ratio = relevance_ratio(chunk_start, chunk_end, GOLD_START, GOLD_END)
    assert ratio == pytest.approx(expected_ratio)
    assert is_relevant(chunk_start, chunk_end, GOLD_START, GOLD_END, threshold=0.5) is expected_relevant


def test_short_chunk_fully_inside_long_span_is_relevant():
    """The case the dedup fix protects: a 500-char section_aware chunk sitting
    inside a 2000-char span still scores 1.0 -- it is not penalised for being
    the tighter unit."""
    assert is_relevant(1200, 1700, GOLD_START, GOLD_END, threshold=0.5) is True
    # ...and the asymmetric rule (overlap / gold-span length) would have given
    # 500 / 1000 = 0.5 here, but only 500 / 2000 = 0.25 for a 2000-char span,
    # which is the unfairness we removed.


def test_long_chunk_covering_short_span_is_relevant():
    assert is_relevant(0, 5000, 2400, 2600, threshold=0.5) is True  # 200/200


def test_chunk_matches_any_respects_paper_id():
    gold = [
        {"paper_id": "paperA", "char_start": 1000, "char_end": 2000},
        {"paper_id": "paperB", "char_start": 0, "char_end": 100},
    ]
    # same offsets, wrong paper -> not a match
    assert chunk_matches_any("paperB", 1000, 2000, gold) is False
    assert chunk_matches_any("paperA", 1000, 2000, gold) is True
    assert chunk_matches_any("paperB", 0, 100, gold) is True


def test_threshold_is_applied():
    # chunk [1700, 2700): 1000 chars, 300 overlap with the span -> ratio 0.3
    assert relevance_ratio(1700, 2700, GOLD_START, GOLD_END) == pytest.approx(0.3)
    assert is_relevant(1700, 2700, GOLD_START, GOLD_END, threshold=0.5) is False
    assert is_relevant(1700, 2700, GOLD_START, GOLD_END, threshold=0.25) is True


def test_overlap_rule_against_100_60_40_0_percent_of_the_span():
    """The literal case from the spec: chunks containing 100%, 60%, 40%, and
    0% of a span. Every chunk here is at least as long as the span, so
    relevance_ratio reduces to intersection / span_length -- "the chunk
    contains X% of the span" -- and only the >=50% ones are relevant."""
    assert relevance_ratio(500, 3500, GOLD_START, GOLD_END) == pytest.approx(1.0)
    assert is_relevant(500, 3500, GOLD_START, GOLD_END) is True

    assert relevance_ratio(600, 1600, GOLD_START, GOLD_END) == pytest.approx(0.6)
    assert is_relevant(600, 1600, GOLD_START, GOLD_END) is True

    assert relevance_ratio(1600, 3000, GOLD_START, GOLD_END) == pytest.approx(0.4)
    assert is_relevant(1600, 3000, GOLD_START, GOLD_END) is False

    assert relevance_ratio(2000, 3000, GOLD_START, GOLD_END) == pytest.approx(0.0)
    assert is_relevant(2000, 3000, GOLD_START, GOLD_END) is False


# --------------------------------------------------------------------------- #
# Per-question metrics: MRR (primary), success@k, recall@k
# --------------------------------------------------------------------------- #
GOLD_SPANS = [{"paper_id": "p1", "char_start": GOLD_START, "char_end": GOLD_END}]
RELEVANT = ("p1", GOLD_START, GOLD_END)
IRRELEVANT = ("p1", 9000, 9100)


def test_reciprocal_rank_hand_computed_fixtures():
    # gold at rank 1 -> MRR 1.0
    assert reciprocal_rank([RELEVANT, IRRELEVANT, IRRELEVANT], GOLD_SPANS) == pytest.approx(1.0)
    # gold at rank 3 -> MRR 0.333...
    assert reciprocal_rank([IRRELEVANT, IRRELEVANT, RELEVANT], GOLD_SPANS) == pytest.approx(1 / 3)
    # gold absent -> MRR 0.0
    assert reciprocal_rank([IRRELEVANT, IRRELEVANT, IRRELEVANT], GOLD_SPANS) == pytest.approx(0.0)
    # no hits at all -> 0.0, not an error
    assert reciprocal_rank([], GOLD_SPANS) == pytest.approx(0.0)


def test_success_at_k_is_binary_and_position_insensitive():
    hits = [IRRELEVANT, IRRELEVANT, RELEVANT, IRRELEVANT]
    assert success_at_k(hits, GOLD_SPANS, k=1) is False
    assert success_at_k(hits, GOLD_SPANS, k=3) is True
    assert success_at_k(hits, GOLD_SPANS, k=4) is True
    assert success_at_k([IRRELEVANT] * 5, GOLD_SPANS, k=5) is False


def test_recall_at_k_counts_distinct_gold_spans_found():
    gold = [
        {"paper_id": "p1", "char_start": 0, "char_end": 100},
        {"paper_id": "p1", "char_start": 1000, "char_end": 1100},
        {"paper_id": "p1", "char_start": 2000, "char_end": 2100},
    ]
    hits = [("p1", 0, 100), ("p1", 1000, 1100), ("p1", 9000, 9100)]  # 2 of 3 spans covered
    assert recall_at_k(hits, gold, k=3) == pytest.approx(2 / 3)
    assert recall_at_k(hits, gold, k=1) == pytest.approx(1 / 3)  # only the first hit counts


def test_recall_at_k_does_not_double_count_two_hits_on_the_same_span():
    gold = [{"paper_id": "p1", "char_start": 1000, "char_end": 2000}]
    hits = [("p1", 1000, 1500), ("p1", 1500, 2000)]  # both relevant to the one span
    assert recall_at_k(hits, gold, k=2) == pytest.approx(1.0)


def test_metrics_ignore_hits_from_the_wrong_paper():
    hits = [("other-paper", GOLD_START, GOLD_END)]
    assert reciprocal_rank(hits, GOLD_SPANS) == pytest.approx(0.0)
    assert success_at_k(hits, GOLD_SPANS, k=1) is False
    assert recall_at_k(hits, GOLD_SPANS, k=1) == pytest.approx(0.0)
