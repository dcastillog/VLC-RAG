"""Tests for success@budget (ragvlc.eval.budget) -- pure, no Qdrant, no model."""

from __future__ import annotations

import math

import pytest

from ragvlc.eval.budget import (
    cumulative_tokens_to_first_relevant,
    first_relevant_rank,
    success_at_budget_curve,
    token_lookup,
)


def test_token_lookup_keys_by_paper_char_start_char_end():
    records = [
        {"paper_id": "p1", "char_start": 0, "char_end": 100, "n_tokens": 42, "text": "..."},
        {"paper_id": "p1", "char_start": 100, "char_end": 300, "n_tokens": 88, "text": "..."},
    ]
    lookup = token_lookup(records)
    assert lookup[("p1", 0, 100)] == 42
    assert lookup[("p1", 100, 300)] == 88


@pytest.mark.parametrize(
    ("rr", "expected_rank"),
    [(1.0, 1), (0.5, 2), (1 / 3, 3), (0.1, 10), (0.0, None), (-0.0, None)],
)
def test_first_relevant_rank_inverts_reciprocal_rank(rr, expected_rank):
    assert first_relevant_rank(rr) == expected_rank


def test_cumulative_tokens_sums_up_to_and_including_the_first_relevant_hit():
    hits = [("p1", 0, 100), ("p1", 100, 200), ("p1", 200, 300)]
    tokens = {("p1", 0, 100): 50, ("p1", 100, 200): 75, ("p1", 200, 300): 400}
    # relevant at rank 2 -> only the first two hits' tokens count
    assert cumulative_tokens_to_first_relevant(hits, rr=0.5, tokens=tokens) == 125


def test_cumulative_tokens_is_infinite_when_nothing_was_relevant():
    hits = [("p1", 0, 100), ("p1", 100, 200)]
    tokens = {("p1", 0, 100): 50, ("p1", 100, 200): 75}
    assert cumulative_tokens_to_first_relevant(hits, rr=0.0, tokens=tokens) == math.inf


def test_cumulative_tokens_handles_a_missing_lookup_entry_gracefully():
    hits = [("p1", 0, 100)]
    assert cumulative_tokens_to_first_relevant(hits, rr=1.0, tokens={}) == 0


def test_success_at_budget_curve_is_monotonic_nondecreasing():
    first_relevant = [50.0, 200.0, math.inf, 800.0, 100.0]
    budgets = [0, 100, 400, 800, 10_000]
    curve = success_at_budget_curve(first_relevant, budgets)
    assert curve == sorted(curve)
    assert curve[0] == 0.0  # budget 0 can never reach any hit
    assert curve[-1] == pytest.approx(4 / 5)  # the math.inf question never succeeds, however large the budget


def test_success_at_budget_curve_matches_hand_computed_fractions():
    first_relevant = [100.0, 100.0, 500.0, math.inf]
    curve = success_at_budget_curve(first_relevant, budgets=[50, 100, 500])
    assert curve == [0.0, 0.5, 0.75]


def test_success_at_budget_curve_empty_input():
    assert success_at_budget_curve([], budgets=[100, 200]) == [0.0, 0.0]


def test_fixed_returns_more_text_at_matched_rank_than_section_aware():
    """The motivating case: at k=3, `fixed`'s uniform ~400-token chunks cost
    far more budget than `section_aware`'s smaller ones for the same rank,
    even when both find the relevant hit at the same position."""
    fixed_hits = [("p1", 0, 400), ("p1", 400, 800), ("p1", 800, 1200)]
    fixed_tokens = {("p1", 0, 400): 400, ("p1", 400, 800): 400, ("p1", 800, 1200): 400}
    section_hits = [("p1", 0, 90), ("p1", 90, 250), ("p1", 250, 400)]
    section_tokens = {("p1", 0, 90): 90, ("p1", 90, 250): 160, ("p1", 250, 400): 150}

    # both find the answer at rank 3 (rr = 1/3)
    fixed_cost = cumulative_tokens_to_first_relevant(fixed_hits, 1 / 3, fixed_tokens)
    section_cost = cumulative_tokens_to_first_relevant(section_hits, 1 / 3, section_tokens)
    assert fixed_cost == 1200
    assert section_cost == 400
    assert fixed_cost > section_cost
