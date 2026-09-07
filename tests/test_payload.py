"""Tests for the chunk -> Qdrant payload join (``ragvlc.retrieval.payload``)
and the deterministic point id. Both are pure -- no Qdrant, no embedding.

The join is the one place section metadata is attached to a chunk, and it has
to behave differently for the two chunkers: a ``section_aware`` chunk sits in
exactly one unit, a ``fixed`` chunk can straddle several.
"""

from __future__ import annotations

from ragvlc.chunking import Paper, Unit
from ragvlc.retrieval import PAYLOAD_FIELDS, build_payload, point_id


def _unit(unit_id: str, heading: str, section_type: str, start: int, end: int, *, equation: bool = False) -> Unit:
    return Unit(
        unit_id=unit_id,
        type="section",
        section_heading=heading,
        parent_heading=heading,
        section_type=section_type,
        contains_equation=equation,
        char_start=start,
        char_end=end,
    )


def _paper() -> Paper:
    # text layout: [0:100) Introduction, sep, [102:210) Method (has equation)
    text = "I" * 100 + "\n\n" + "M" * 108
    return Paper(
        paper_id="p1",
        text=text,
        units=(
            _unit("u0", "Introduction", "introduction", 0, 100),
            _unit("u1", "Method", "methods", 102, 210, equation=True),
        ),
        doi="10.1/x",
        title="A Paper",
        year=2021,
        venue="Sensors",
        licence="CC-BY-4.0",
    )


def _record(chunk_id: str, start: int, end: int, n_tokens: int = 42) -> dict:
    paper_id = chunk_id.split(":")[0]
    return {
        "chunk_id": chunk_id,
        "paper_id": paper_id,
        "char_start": start,
        "char_end": end,
        "n_tokens": n_tokens,
        "text": _paper().text[start:end],
    }


def test_payload_has_exactly_the_expected_fields():
    payload = build_payload(_record("p1:section_aware:0003", 10, 90), _paper(), "section_aware")
    assert set(payload) == set(PAYLOAD_FIELDS)


def test_section_aware_chunk_inside_one_unit_gets_that_units_metadata():
    payload = build_payload(_record("p1:section_aware:0000", 10, 90), _paper(), "section_aware")
    assert payload["section_heading"] == "Introduction"
    assert payload["section_type"] == "introduction"
    assert payload["contains_equation"] is False
    assert payload["chunk_index"] == 0
    assert (payload["char_start"], payload["char_end"]) == (10, 90)
    assert payload["chunker"] == "section_aware"
    # paper-level fields carried through
    assert payload["year"] == 2021 and payload["venue"] == "Sensors" and payload["doi"] == "10.1/x"


def test_fixed_chunk_spanning_two_units_takes_the_dominant_one():
    # 30 chars in Introduction (70..100), 78 chars in Method (102..180) -> Method dominates
    payload = build_payload(_record("p1:fixed:0007", 70, 180), _paper(), "fixed")
    assert payload["section_heading"] == "Method"
    assert payload["section_type"] == "methods"
    # equation OR across every overlapped unit
    assert payload["contains_equation"] is True
    assert payload["chunk_index"] == 7


def test_chunk_overlapping_no_unit_yields_null_section_fields():
    payload = build_payload(_record("p1:fixed:0001", 100, 102), _paper(), "fixed")  # the "\n\n" gap
    assert payload["section_heading"] is None
    assert payload["section_type"] is None
    assert payload["contains_equation"] is False


def test_point_id_is_deterministic_and_unique_per_chunk():
    assert point_id("p1:fixed:0000") == point_id("p1:fixed:0000")
    assert point_id("p1:fixed:0000") != point_id("p1:fixed:0001")
    assert point_id("p1:fixed:0000") != point_id("p2:fixed:0000")
    # a UUID string
    assert len(point_id("p1:fixed:0000")) == 36
