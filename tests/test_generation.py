"""Tests for Phase A generation.

All pure: context-block formatting, abstention-marker detection, citation
extraction and verification, and the response parser. The HTTP path in
``client.complete`` is not unit-tested here (the repo doesn't mock httpx for
GROBID/Crossref either); it is exercised end to end by scripts/answer.py and
the Phase B abstention check against a live endpoint.
"""

from __future__ import annotations

import pytest

from ragvlc.generation import (
    ABSTAIN_MARKER,
    build_context_block,
    detect_abstention,
    parse_citations,
    resolve_citations,
    strip_reasoning_block,
)
from ragvlc.generation.client import GenerationError, _parse_completion
from ragvlc.retrieval import Hit


def _hit(i: int, **overrides) -> Hit:
    payload = {
        "paper_id": f"paper-{i}",
        "title": f"Title {i}",
        "doi": f"10.0/{i}",
        "year": 2020 + i,
        "section_heading": f"Section {i}",
        "parent_heading": None,
        "char_start": 100 * i,
        "char_end": 100 * i + 50,
        "text": f"body text {i}",
    }
    payload.update(overrides)
    return Hit(point_id=f"id-{i}", score=1.0 / i, payload=payload)


# --------------------------------------------------------------------------- #
# Context block
# --------------------------------------------------------------------------- #
def test_context_block_numbers_from_one_with_title_year_heading():
    block = build_context_block([_hit(1), _hit(2)])
    assert "[1] Title 1 (2021) -- Section 1\nbody text 1" in block
    assert "[2] Title 2 (2022) -- Section 2\nbody text 2" in block


def test_context_block_falls_back_for_missing_metadata():
    block = build_context_block([_hit(1, title=None, year=None, section_heading=None, parent_heading="Parent")])
    assert "[1] Untitled (n.d.) -- Parent" in block


# --------------------------------------------------------------------------- #
# Abstention
# --------------------------------------------------------------------------- #
def test_detect_abstention_strips_leading_marker():
    abstained, text = detect_abstention(f"  {ABSTAIN_MARKER}: the sources do not cover this.")
    assert abstained is True
    assert text == "the sources do not cover this."


def test_detect_abstention_false_for_a_real_answer():
    abstained, text = detect_abstention("The data rate is limited by the frame rate [1].")
    assert abstained is False
    assert text == "The data rate is limited by the frame rate [1]."


def test_marker_only_mid_sentence_is_not_an_abstention():
    abstained, _ = detect_abstention(f"The answer is {ABSTAIN_MARKER} somehow")
    assert abstained is False


def test_strip_reasoning_block_removes_leading_think_tags():
    assert strip_reasoning_block("<think>weighing it up</think>\nThe answer is X [1].") == "The answer is X [1]."
    assert strip_reasoning_block("No think block here [1].") == "No think block here [1]."


def test_abstention_survives_a_leading_think_block():
    raw = f"<think>nothing relevant</think>\n{ABSTAIN_MARKER} the sources do not cover this."
    abstained, text = detect_abstention(strip_reasoning_block(raw))
    assert abstained is True
    assert text == "the sources do not cover this."


# --------------------------------------------------------------------------- #
# Citation extraction + verification
# --------------------------------------------------------------------------- #
def test_parse_citations_splits_valid_and_out_of_range():
    valid, invalid = parse_citations("Claim one [1][2]. Claim two [2]. Bogus [7].", n_hits=3)
    assert valid == [1, 2]
    assert invalid == [7]


def test_parse_citations_deduplicates_by_first_appearance():
    valid, invalid = parse_citations("[3] then [1] then [3] again", n_hits=3)
    assert valid == [3, 1]
    assert invalid == []


def test_resolve_citations_maps_index_to_chunk_payload():
    hits = (_hit(1), _hit(2), _hit(3))
    citations = resolve_citations([3, 1], hits)
    assert [c.index for c in citations] == [1, 3]  # sorted by index
    first = citations[0]
    assert (first.paper_id, first.doi, first.char_start, first.char_end) == ("paper-1", "10.0/1", 100, 150)


# --------------------------------------------------------------------------- #
# Response parser
# --------------------------------------------------------------------------- #
def test_parse_completion_extracts_content_model_and_finish_reason():
    body = {
        "model": "qwen3:4b",
        "choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
    }
    completion = _parse_completion(body, fallback_model="fallback")
    assert completion.text == "hello"
    assert completion.model == "qwen3:4b"
    assert completion.finish_reason == "stop"


def test_parse_completion_uses_fallback_model_when_absent():
    body = {"choices": [{"message": {"content": "hi"}}]}
    assert _parse_completion(body, fallback_model="fallback").model == "fallback"


@pytest.mark.parametrize("body", [{}, {"choices": []}, {"choices": [{"message": {}}]}, {"choices": [{"message": {"content": 5}}]}])
def test_parse_completion_rejects_malformed_bodies(body):
    with pytest.raises((KeyError, IndexError, TypeError)):
        _parse_completion(body, fallback_model="fallback")


def test_generation_error_is_a_runtime_error():
    assert issubclass(GenerationError, RuntimeError)
