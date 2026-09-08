"""Tests for the pure half of the IDF-overlap regression (ragvlc.eval.idf):
the IDF formula and the overlap score, given token-id sets directly. The
tokenizer itself (text -> token-id set) needs the bm25 model and is exercised
indirectly through scripts/run_eval.py, not unit-tested here.
"""

from __future__ import annotations

import math

import pytest

from ragvlc.eval.idf import compute_idf, overlap_score


def test_idf_is_higher_for_rarer_tokens():
    # token 1 in every doc, token 2 in one of three, token 3 in none
    docs = [{1, 2}, {1}, {1}]
    idf = compute_idf(docs)
    assert idf.get(1) < idf.get(2)
    assert idf.get(2) < idf.get(3)  # token 3 never appears -> the "unseen" default


def test_idf_unseen_token_gets_the_maximal_default():
    docs = [{1}, {1}, {1}]
    idf = compute_idf(docs)
    unseen = idf.get(999)
    assert unseen == pytest.approx(idf.unseen_idf)
    # matches the df=0 case of the same formula
    assert unseen == pytest.approx(math.log(1 + (3 + 0.5) / 0.5))


def test_idf_of_a_token_in_every_document_is_small_but_not_negative():
    docs = [{1}, {1}, {1}, {1}]
    idf = compute_idf(docs)
    assert idf.get(1) >= 0


def test_overlap_score_is_full_when_all_question_tokens_are_shared():
    docs = [{1, 2, 3}, {1}, {2}]
    idf = compute_idf(docs)
    assert overlap_score(idf, question_tokens={1, 2}, answer_tokens={1, 2, 9}) == pytest.approx(1.0)


def test_overlap_score_weights_by_idf_not_just_token_count():
    # token 1 is common (low idf), token 2 is rare (high idf)
    docs = [{1}] * 9 + [{1, 2}]
    idf = compute_idf(docs)
    question = {1, 2}
    # sharing only the rare token 2 scores higher than sharing only common token 1
    only_common = overlap_score(idf, question, answer_tokens={1})
    only_rare = overlap_score(idf, question, answer_tokens={2})
    assert only_rare > only_common


def test_overlap_score_is_zero_for_no_shared_tokens():
    docs = [{1, 2}]
    idf = compute_idf(docs)
    assert overlap_score(idf, question_tokens={1, 2}, answer_tokens={3, 4}) == 0.0


def test_overlap_score_is_zero_for_an_empty_question():
    docs = [{1, 2}]
    idf = compute_idf(docs)
    assert overlap_score(idf, question_tokens=set(), answer_tokens={1, 2}) == 0.0


def test_overlap_score_is_bounded_zero_to_one():
    docs = [{1, 2, 3, 4, 5}, {1}, {2, 3}]
    idf = compute_idf(docs)
    for answer in [set(), {1}, {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5, 6, 7}]:
        score = overlap_score(idf, question_tokens={1, 2, 3}, answer_tokens=answer)
        assert 0.0 <= score <= 1.0
