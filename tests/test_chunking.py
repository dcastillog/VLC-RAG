"""Tests for the two chunking strategies.

The corpus-wide guarantee is offset integrity: ``chunk.text`` must equal
``normalized_text[char_start:char_end]`` for every chunk of every strategy,
because Phase F judges relevance by character overlap and a drifted offset
corrupts the metric silently. These tests pin that plus the structural
promises each strategy makes.

They build synthetic ``Paper`` objects (``data/normalized/`` is gitignored and
not present in CI) but use the *real* ``BAAI/bge-small-en-v1.5`` tokenizer via
a session-scoped ``TokenCounter`` -- token budgets measured any other way
would not be the budgets the model actually enforces.
"""

from __future__ import annotations

import pytest

from ragvlc.chunking import Paper, TokenCounter, Unit
from ragvlc.chunking.base import ChunkOffsetError, make_chunk
from ragvlc.chunking.fixed import FixedChunker
from ragvlc.chunking.section_aware import SectionAwareChunker

MAX_TOKENS = 400
OVERLAP_TOKENS = 60
MIN_CHUNK_TOKENS = 50
UNIT_SEPARATOR = "\n\n"  # matches config.parsing.unit_separator


@pytest.fixture(scope="session")
def counter() -> TokenCounter:
    return TokenCounter("BAAI/bge-small-en-v1.5")


def _make_paper(paper_id: str, specs: list[tuple[str, str, str]]) -> Paper:
    """Assemble a ``Paper`` from ``(unit_type, section_heading, body)`` triples,
    joined by the canonical unit separator, with exact offsets."""
    bodies = [body for _, _, body in specs]
    text = UNIT_SEPARATOR.join(bodies)

    units: list[Unit] = []
    cursor = 0
    for i, (unit_type, heading, body) in enumerate(specs):
        units.append(
            Unit(
                unit_id=f"u{i:03d}",
                type=unit_type,
                section_heading=heading,
                parent_heading=heading,
                section_type="other",
                contains_equation=False,
                char_start=cursor,
                char_end=cursor + len(body),
            )
        )
        cursor += len(body) + len(UNIT_SEPARATOR)

    return Paper(paper_id=paper_id, text=text, units=tuple(units))


_SENTENCE = (
    "Visible light communication uses rapid intensity modulation of light emitting diodes "
    "to carry data across a short indoor free space optical channel to a photodiode receiver. "
)


def _prose(n_sentences: int) -> str:
    return (_SENTENCE * n_sentences).strip()


@pytest.fixture(scope="session")
def sample_paper() -> Paper:
    """A paper with an abstract, several ordinary sections, one unit far over
    the token budget, a run of tiny same-heading units, a tiny unit sandwiched
    between different headings, and captions of various lengths."""
    return _make_paper(
        "synthetic-vlc-paper",
        [
            ("abstract", "Abstract", _prose(4)),
            ("section", "Introduction", _prose(6)),
            ("section", "Introduction", "A short follow-up remark on the motivation."),
            ("section", "System Model", _prose(60)),  # well over 400 tokens -> split
            ("caption", "System Model", "Figure 1. Block diagram of the link."),
            ("section", "Results", "First tiny result."),
            ("section", "Results", "Second tiny result."),
            ("section", "Results", "Third tiny result."),
            ("caption", "Results", "Table 1. " + _prose(70)),  # over-budget caption
            ("section", "Discussion", "An isolated one-liner under its own heading."),
            ("section", "Conclusion", _prose(5)),
            ("caption", "Conclusion", "Figure 3. Summary."),  # a few tokens -> below the index floor
        ],
    )


INDEX_FLOOR = 15


@pytest.fixture(scope="session")
def fixed_chunks(counter: TokenCounter, sample_paper: Paper):
    chunker = FixedChunker(counter, max_tokens=MAX_TOKENS, overlap_tokens=OVERLAP_TOKENS)
    return chunker.chunk(sample_paper)


@pytest.fixture(scope="session")
def section_chunks(counter: TokenCounter, sample_paper: Paper):
    chunker = SectionAwareChunker(counter, max_tokens=MAX_TOKENS, min_chunk_tokens=MIN_CHUNK_TOKENS)
    return chunker.chunk(sample_paper)


# --------------------------------------------------------------------------- #
# Offset integrity -- the one guarantee both strategies must never break.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fixture_name", ["fixed_chunks", "section_chunks"])
def test_offsets_round_trip(fixture_name: str, sample_paper: Paper, request: pytest.FixtureRequest):
    chunks = request.getfixturevalue(fixture_name)
    assert chunks, "expected a non-empty chunk set"
    for chunk in chunks:
        assert chunk.text == sample_paper.text[chunk.char_start : chunk.char_end]
        assert chunk.paper_id == sample_paper.paper_id
        assert chunk.char_start < chunk.char_end


@pytest.mark.parametrize("fixture_name", ["fixed_chunks", "section_chunks"])
def test_no_chunk_exceeds_the_token_budget(fixture_name: str, counter: TokenCounter, request: pytest.FixtureRequest):
    chunks = request.getfixturevalue(fixture_name)
    for chunk in chunks:
        assert chunk.n_tokens <= MAX_TOKENS
        assert chunk.n_tokens == counter.count(chunk.text)  # recorded count is honest


def test_chunk_ids_are_unique_and_ordered(fixed_chunks, section_chunks):
    for chunks, name in [(fixed_chunks, "fixed"), (section_chunks, "section_aware")]:
        ids = [c.chunk_id for c in chunks]
        assert ids == sorted(ids)
        assert len(ids) == len(set(ids))
        assert ids[0] == f"synthetic-vlc-paper:{name}:0000"


# --------------------------------------------------------------------------- #
# fixed: the overlap has to actually be there.
# --------------------------------------------------------------------------- #
def test_fixed_overlap_is_present(fixed_chunks, sample_paper: Paper, counter: TokenCounter):
    assert len(fixed_chunks) >= 3, "sample paper should need several fixed windows"
    for earlier, later in zip(fixed_chunks, fixed_chunks[1:]):
        # windows share characters...
        assert later.char_start < earlier.char_end
        # ...and the shared slice carries ~the configured token overlap
        shared = sample_paper.text[later.char_start : earlier.char_end]
        assert counter.count_content(shared) >= OVERLAP_TOKENS


def test_fixed_chunks_cover_the_text_left_to_right(fixed_chunks):
    for earlier, later in zip(fixed_chunks, fixed_chunks[1:]):
        assert later.char_start > earlier.char_start
        assert later.char_end > earlier.char_end


# --------------------------------------------------------------------------- #
# section_aware: structural contract.
# --------------------------------------------------------------------------- #
def _overlapping_units(paper: Paper, char_start: int, char_end: int) -> list[Unit]:
    return [
        u for u in paper.units
        if not (u.char_end <= char_start or u.char_start >= char_end)
    ]


def test_section_aware_never_partially_spans_two_units(section_chunks, sample_paper: Paper):
    """A chunk is either inside one unit or an exact concatenation of whole
    consecutive units -- never mid-A-to-mid-B. This is the operational meaning
    of "never merge across units"."""
    for chunk in section_chunks:
        touched = _overlapping_units(sample_paper, chunk.char_start, chunk.char_end)
        assert touched
        if len(touched) == 1:
            unit = touched[0]
            assert unit.char_start <= chunk.char_start < chunk.char_end <= unit.char_end
        else:
            # whole-unit concatenation: boundaries coincide with unit edges,
            # units are consecutive, and they share one section_heading
            assert chunk.char_start == touched[0].char_start
            assert chunk.char_end == touched[-1].char_end
            indices = [sample_paper.units.index(u) for u in touched]
            assert indices == list(range(indices[0], indices[0] + len(indices)))
            assert len({u.section_heading for u in touched}) == 1


def test_section_aware_never_merges_a_caption(section_chunks, sample_paper: Paper):
    caption_units = [u for u in sample_paper.units if u.type == "caption"]
    assert caption_units
    for caption in caption_units:
        covering = [
            c for c in section_chunks
            if not (c.char_end <= caption.char_start or c.char_start >= caption.char_end)
        ]
        # every chunk touching a caption stays strictly within that caption's
        # character range -- nothing else is ever glued on
        for chunk in covering:
            assert caption.char_start <= chunk.char_start < chunk.char_end <= caption.char_end


def test_section_aware_splits_oversized_units(section_chunks, sample_paper: Paper, counter: TokenCounter):
    big_unit = sample_paper.units[3]  # the _prose(60) "System Model" section
    assert counter.count(sample_paper.text[big_unit.char_start : big_unit.char_end]) > MAX_TOKENS
    pieces = [
        c for c in section_chunks
        if c.char_start >= big_unit.char_start and c.char_end <= big_unit.char_end
    ]
    assert len(pieces) >= 2
    # the split is a partition of the unit (contiguous, gapless, in order)
    assert pieces[0].char_start == big_unit.char_start
    assert pieces[-1].char_end == big_unit.char_end
    for earlier, later in zip(pieces, pieces[1:]):
        assert later.char_start == earlier.char_end


def test_section_aware_merges_tiny_same_heading_units(section_chunks, sample_paper: Paper):
    """The three tiny "Results" text units collapse into fewer chunks than
    there were units."""
    results_text_units = [
        u for u in sample_paper.units if u.section_heading == "Results" and u.type == "section"
    ]
    assert len(results_text_units) == 3
    span_start = min(u.char_start for u in results_text_units)
    span_end = max(u.char_end for u in results_text_units)
    covering = [
        c for c in section_chunks
        if c.char_start >= span_start and c.char_end <= span_end
    ]
    assert 0 < len(covering) < len(results_text_units)


def test_section_aware_drops_sub_floor_chunks(counter: TokenCounter, sample_paper: Paper):
    """The tiny "Figure 3. Summary." caption is below the floor: it is returned
    by chunk_with_drops as dropped, absent from chunk(), and the kept set stays
    contiguously indexed. fixed never sees this path."""
    chunker = SectionAwareChunker(
        counter,
        max_tokens=MAX_TOKENS,
        min_chunk_tokens=MIN_CHUNK_TOKENS,
        min_indexed_tokens=INDEX_FLOOR,
    )
    kept, dropped = chunker.chunk_with_drops(sample_paper)

    assert dropped, "the sample paper has a sub-floor caption"
    assert all(c.n_tokens < INDEX_FLOOR for c in dropped)
    assert all(c.n_tokens >= INDEX_FLOOR for c in kept)

    # chunk() is exactly the kept set...
    assert [c.chunk_id for c in chunker.chunk(sample_paper)] == [c.chunk_id for c in kept]
    # ...contiguously re-indexed, no gaps from the removed chunks
    assert [c.chunk_id for c in kept] == [
        f"synthetic-vlc-paper:section_aware:{i:04d}" for i in range(len(kept))
    ]
    # dropped chunks are still real spans -- they just are not indexed
    for chunk in dropped:
        assert chunk.text == sample_paper.text[chunk.char_start : chunk.char_end]

    # the dropped caption's text is nowhere in the indexed set
    dropped_text = {c.text for c in dropped}
    assert not any(c.text in dropped_text for c in kept)


def test_isolated_tiny_unit_under_its_own_heading_survives_unmerged(section_chunks, sample_paper: Paper):
    lonely = next(u for u in sample_paper.units if u.section_heading == "Discussion")
    covering = [
        c for c in section_chunks
        if not (c.char_end <= lonely.char_start or c.char_start >= lonely.char_end)
    ]
    assert len(covering) == 1
    assert covering[0].char_start == lonely.char_start
    assert covering[0].char_end == lonely.char_end


# --------------------------------------------------------------------------- #
# The offset tripwire in make_chunk fires.
# --------------------------------------------------------------------------- #
def test_make_chunk_raises_on_out_of_bounds(counter: TokenCounter):
    paper = _make_paper("p", [("section", "S", "Some ordinary text here.")])
    with pytest.raises(ChunkOffsetError):
        make_chunk(paper, "fixed", 0, 0, len(paper.text) + 5, counter)


def test_make_chunk_raises_when_assembled_text_disagrees_with_slice(counter: TokenCounter):
    paper = _make_paper("p", [("section", "S", "Some ordinary text here.")])
    with pytest.raises(ChunkOffsetError):
        make_chunk(paper, "fixed", 0, 0, 4, counter, expected_text="Some other thing")
