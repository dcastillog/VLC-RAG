"""Shared types and helpers for the two chunking strategies.

Both chunkers implement the same interface -- ``chunk(paper) -> list[Chunk]``
-- so the Phase E pool builder and the eval runner can iterate over them
without knowing which is which.

The one invariant every chunk must satisfy, enforced by :func:`make_chunk`:

    chunk.text == paper.text[chunk.char_start:chunk.char_end]

This is the same guarantee ``pipeline.py`` makes for units. Retrieval
relevance in Phase F is judged purely by character overlap between a chunk's
``[char_start, char_end]`` and a gold span, so an offset that has drifted off
its text does not fail loudly -- it silently corrupts every metric. We raise
instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ragvlc.chunking.tokenizer import TokenCounter


class ChunkOffsetError(RuntimeError):
    """A chunk's recorded text does not match its char range in the source."""


@dataclass(frozen=True)
class Unit:
    """The parsing-stage unit fields the chunkers actually use.

    A thin projection of one entry in ``data/normalized/{paper_id}.json`` --
    the chunkers never touch the rest (PUA counts, non-ASCII ratios, ...).
    """

    unit_id: str
    type: str
    section_heading: str | None
    parent_heading: str | None
    section_type: str | None
    contains_equation: bool
    char_start: int
    char_end: int


@dataclass(frozen=True)
class Paper:
    """A parsed paper: the frozen canonical text plus its unit boundaries."""

    paper_id: str
    text: str
    units: tuple[Unit, ...]

    @classmethod
    def from_normalized(cls, json_path: Path) -> Paper:
        """Load a paper from its ``{paper_id}.json`` (and sibling ``.txt``)."""
        doc = json.loads(json_path.read_text(encoding="utf-8"))
        txt_path = json_path.with_suffix(".txt")
        text = txt_path.read_text(encoding="utf-8")
        units = tuple(
            Unit(
                unit_id=u["unit_id"],
                type=u["type"],
                section_heading=u.get("section_heading"),
                parent_heading=u.get("parent_heading"),
                section_type=u.get("section_type"),
                contains_equation=bool(u.get("contains_equation", False)),
                char_start=u["char_start"],
                char_end=u["char_end"],
            )
            for u in doc.get("units", [])
        )
        return cls(paper_id=doc["paper_id"], text=text, units=units)


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit of text. Exactly the fields Phase C will embed."""

    chunk_id: str  # f"{paper_id}:{chunker}:{index:04d}"
    paper_id: str
    text: str  # exactly paper.text[char_start:char_end]
    char_start: int
    char_end: int
    n_tokens: int  # full encoded length incl. [CLS]/[SEP] -- what 512 caps

    def to_json_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "n_tokens": self.n_tokens,
            "text": self.text,
        }


class Chunker(Protocol):
    """The common interface. ``name`` is what ends up in the chunk_id and the
    Qdrant ``chunker`` payload field."""

    name: str

    def chunk(self, paper: Paper) -> list[Chunk]: ...


def make_chunk(
    paper: Paper,
    chunker: str,
    index: int,
    char_start: int,
    char_end: int,
    counter: TokenCounter,
    *,
    expected_text: str | None = None,
) -> Chunk:
    """Build a :class:`Chunk`, raising :class:`ChunkOffsetError` on any
    offset inconsistency rather than returning a corrupt chunk.

    ``expected_text``, when a chunker assembled the span from pieces, is
    cross-checked against the slice -- that is where an off-by-one in a
    chunker's span arithmetic actually surfaces.
    """
    if not (0 <= char_start < char_end <= len(paper.text)):
        raise ChunkOffsetError(
            f"{paper.paper_id}:{chunker}:{index:04d}: char range [{char_start}:{char_end}] "
            f"is out of bounds for text of length {len(paper.text)}"
        )

    text = paper.text[char_start:char_end]

    # Both chunkers define a chunk purely as an offset pair and let `text` be
    # the slice, so this identity holds by construction -- it is kept as an
    # explicit, cheap tripwire against a future chunker that assembles `text`
    # some other way (the same guarantee pipeline.py makes for units).
    if text != paper.text[char_start:char_end]:  # pragma: no cover - structural
        raise ChunkOffsetError(
            f"{paper.paper_id}:{chunker}:{index:04d}: text is not the [{char_start}:{char_end}] slice"
        )

    # Python slicing truncates a past-the-end stop silently, so a length check
    # is the reliable signal (same reasoning as verify_corpus.py).
    if len(text) != char_end - char_start:
        raise ChunkOffsetError(
            f"{paper.paper_id}:{chunker}:{index:04d}: slice [{char_start}:{char_end}] yielded "
            f"{len(text)} chars, expected {char_end - char_start}"
        )

    if expected_text is not None and expected_text != text:
        raise ChunkOffsetError(
            f"{paper.paper_id}:{chunker}:{index:04d}: assembled text does not match the slice\n"
            f"  assembled: {expected_text!r}\n"
            f"  slice:     {text!r}"
        )

    return Chunk(
        chunk_id=f"{paper.paper_id}:{chunker}:{index:04d}",
        paper_id=paper.paper_id,
        text=text,
        char_start=char_start,
        char_end=char_end,
        n_tokens=counter.count(text),
    )


def pack_spans(
    spans: list[tuple[int, int]],
    span_tokens: list[int],
    content_budget: int,
    overlap_tokens: int = 0,
) -> list[tuple[int, int]]:
    """Greedily pack consecutive ``spans`` (each a ``(char_start, char_end)``
    pair, in order) into windows whose summed token count stays within
    ``content_budget``.

    WordPiece tokenizes each whitespace-separated group independently, so the
    token count of a concatenation equals the sum of its parts' counts -- which
    is what lets this pack by a running sum instead of re-encoding every
    candidate window.

    ``overlap_tokens`` > 0 makes each window start ``overlap_tokens`` back from
    where the previous one ended (used by ``fixed``). Progress is always at
    least one span, so a single span larger than the budget becomes its own
    (over-budget) window rather than looping forever -- callers that cannot
    tolerate that (sentence packing) must split such a span further themselves.
    """
    if not spans:
        return []

    windows: list[tuple[int, int]] = []
    i = 0
    n = len(spans)
    while i < n:
        j = i
        total = 0
        while j < n and (j == i or total + span_tokens[j] <= content_budget):
            total += span_tokens[j]
            j += 1
        windows.append((spans[i][0], spans[j - 1][1]))
        if j >= n:
            break
        if overlap_tokens <= 0:
            i = j
            continue
        # Step the start back until we have re-covered ~overlap_tokens, but
        # never past i + 1 -- that guarantees forward progress.
        back = 0
        k = j
        while k > i + 1 and back < overlap_tokens:
            k -= 1
            back += span_tokens[k]
        i = k
    return windows
