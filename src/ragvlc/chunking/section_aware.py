"""``section_aware`` -- chunking that respects the parsed structure.

Contract, in priority order:

1. **Chunk boundaries never fall inside two different units.** A chunk is
   either a sub-span of one unit (a unit too big for the budget, split up) or
   the exact concatenation of one or more whole consecutive units (small units
   merged). It never runs from the middle of unit A into the middle of unit B.
   This is what "never merge across units" means operationally, and it is what
   makes the two chunkers comparable: every section_aware chunk lines up with
   the structure, every fixed chunk ignores it.

2. **Captions stand alone.** A ``type == "caption"`` unit is never merged with
   anything, in either direction, regardless of how short it is. (A caption
   over the 512-token model limit is still split -- into caption-only
   sub-chunks -- so its tail is not silently truncated at embed time. That is
   about the model limit, not about merging.)

3. **Oversized units are split** at sentence boundaries, and a single
   sentence still over budget is split again at whitespace.

4. **Undersized units merge** with an adjacent unit, but only when both carry
   the same ``section_heading`` and the merged chunk still fits the budget.
   Merging a two-sentence aside into the paragraph next to it is good;
   gluing the last line of one section onto the first line of the next is
   exactly the cross-section bleed contract 1 forbids.

5. **Sub-floor chunks are dropped.** After 1-4, a chunk with fewer than
   ``min_indexed_tokens`` tokens (e.g. a bare "Figure 9. Optical wireless
   communication types.") is not indexed. It carries almost no retrievable
   signal yet still occupies a top-k slot and costs a pool judgment. This is
   an indexing decision, not a parsing failure, and ``fixed`` is deliberately
   exempt -- its short chunks are one-per-paper tails of real content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ragvlc.chunking.base import Chunk, Paper, Unit, make_chunk, pack_spans
from ragvlc.chunking.tokenizer import TokenCounter

_WORD_RE = re.compile(r"\S+")

# A sentence boundary: sentence-ending punctuation, then whitespace, then
# something that looks like the start of a new sentence. Deliberately
# conservative -- it only has to be good enough to keep oversized units under
# budget, and a sentence that slips through still hits the whitespace fallback.
_SENT_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z(\["“])')

# Tokens that end in "." without ending a sentence. Checked lower-cased with a
# trailing "." stripped, so "Fig." matches "fig".
_ABBREVIATIONS: frozenset[str] = frozenset(
    {
        "fig", "figs", "eq", "eqs", "tab", "tabs", "ref", "refs", "sec", "secs",
        "al", "cf", "vs", "no", "nos", "etc", "approx", "e.g", "i.e", "resp",
        "dr", "mr", "mrs", "ms", "prof", "inc", "ltd", "co", "vol", "pp",
    }
)


@dataclass
class _Segment:
    """A candidate chunk before the merge pass. ``kind`` tracks how it was
    formed so the merge rules can keep contract 1 and 2."""

    char_start: int
    char_end: int
    section_heading: str | None
    last_unit_id: str
    kind: str  # "whole" | "partial" | "merged" | "caption"


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Contiguous ``(start, end)`` spans covering all of ``text``, one per
    sentence. ``spans[k][1] == spans[k + 1][0]`` always, so packing a run of
    them yields an exact slice of the original."""
    if not text:
        return []

    starts = [0]
    for match in _SENT_BOUNDARY.finditer(text):
        preceding = text[starts[-1] : match.start()].split()
        last_token = preceding[-1].rstrip(".").lower() if preceding else ""
        if last_token in _ABBREVIATIONS:
            continue
        starts.append(match.end())

    return [
        (start, starts[i + 1] if i + 1 < len(starts) else len(text))
        for i, start in enumerate(starts)
    ]


class SectionAwareChunker:
    name = "section_aware"

    def __init__(
        self,
        counter: TokenCounter,
        max_tokens: int,
        min_chunk_tokens: int,
        min_indexed_tokens: int = 0,
    ) -> None:
        self._counter = counter
        self._max_tokens = max_tokens
        self._min_chunk_tokens = min_chunk_tokens
        self._min_indexed_tokens = min_indexed_tokens
        self._content_budget = max_tokens - counter.special_token_overhead()

    @property
    def min_indexed_tokens(self) -> int:
        """Token floor below which a chunk is dropped rather than indexed
        (0 disables). Exposed so build_chunks.py can label its report."""
        return self._min_indexed_tokens

    # ------------------------------------------------------------------ #
    # Splitting an oversized span (used for both oversized units and, rarely,
    # an oversized caption). Never crosses the [start, end) it is given, so the
    # results stay inside one unit.
    # ------------------------------------------------------------------ #
    def _split_span(self, text: str, start: int, end: int) -> list[tuple[int, int]]:
        local = text[start:end]
        sentences = _sentence_spans(local)
        sentence_tokens = [self._counter.count_content(local[a:b]) for a, b in sentences]
        packed = pack_spans(sentences, sentence_tokens, self._content_budget)

        result: list[tuple[int, int]] = []
        for a, b in packed:
            if self._counter.count(local[a:b]) <= self._max_tokens:
                result.append((start + a, start + b))
                continue
            # A lone sentence over budget: fall back to whitespace packing.
            words = [(m.start(), m.end()) for m in _WORD_RE.finditer(local[a:b])]
            word_tokens = [self._counter.count_content(local[a:b][ws:we]) for ws, we in words]
            for wa, wb in pack_spans(words, word_tokens, self._content_budget):
                result.append((start + a + wa, start + a + wb))
        return result

    # ------------------------------------------------------------------ #
    # Pass 1: one unit -> one or more segments.
    # ------------------------------------------------------------------ #
    def _segment_unit(self, paper: Paper, unit: Unit) -> list[_Segment]:
        unit_tokens = self._counter.count(paper.text[unit.char_start : unit.char_end])
        is_caption = unit.type == "caption"

        if unit_tokens <= self._max_tokens:
            spans = [(unit.char_start, unit.char_end)]
        else:
            spans = self._split_span(paper.text, unit.char_start, unit.char_end)

        kind = "caption" if is_caption else ("whole" if len(spans) == 1 else "partial")
        return [
            _Segment(
                char_start=a,
                char_end=b,
                section_heading=unit.section_heading,
                last_unit_id=unit.unit_id,
                kind=kind,
            )
            for a, b in spans
        ]

    # ------------------------------------------------------------------ #
    # Pass 2: merge undersized segments.
    # ------------------------------------------------------------------ #
    def _should_merge(self, paper: Paper, prev: _Segment, nxt: _Segment) -> bool:
        if prev.kind == "caption" or nxt.kind == "caption":
            return False  # contract 2

        prev_tokens = self._counter.count(paper.text[prev.char_start : prev.char_end])
        next_tokens = self._counter.count(paper.text[nxt.char_start : nxt.char_end])
        if prev_tokens >= self._min_chunk_tokens and next_tokens >= self._min_chunk_tokens:
            return False  # the merge rule is only for undersized segments

        combined = self._counter.count(paper.text[prev.char_start : nxt.char_end])
        if combined > self._max_tokens:
            return False  # contract would be violated the other way

        # Sub-parts of the same oversized unit: merging stays inside that unit,
        # so contract 1 holds regardless of heading.
        if nxt.last_unit_id == prev.last_unit_id:
            return True

        # Different units: only whole units, only under one heading (contract 1).
        prev_whole = prev.kind in ("whole", "merged")
        return (
            prev_whole
            and nxt.kind == "whole"
            and prev.section_heading is not None
            and prev.section_heading == nxt.section_heading
        )

    def _merge(self, prev: _Segment, nxt: _Segment) -> _Segment:
        same_unit = nxt.last_unit_id == prev.last_unit_id
        return _Segment(
            char_start=prev.char_start,
            char_end=nxt.char_end,
            section_heading=prev.section_heading,
            last_unit_id=nxt.last_unit_id,
            kind="partial" if (same_unit and prev.kind == "partial") else "merged",
        )

    # ------------------------------------------------------------------ #
    def _merged_segments(self, paper: Paper) -> list[_Segment]:
        segments: list[_Segment] = []
        for unit in paper.units:
            segments.extend(self._segment_unit(paper, unit))

        merged: list[_Segment] = []
        for segment in segments:
            if merged and self._should_merge(paper, merged[-1], segment):
                merged[-1] = self._merge(merged[-1], segment)
            else:
                merged.append(segment)
        return merged

    def chunk(self, paper: Paper) -> list[Chunk]:
        kept, _dropped = self.chunk_with_drops(paper)
        return kept

    def chunk_with_drops(self, paper: Paper) -> tuple[list[Chunk], list[Chunk]]:
        """``(indexed_chunks, dropped_chunks)``.

        Dropped chunks are fully-formed and offset-checked -- they are just
        below ``min_indexed_tokens`` and will not be embedded. Returned so
        build_chunks.py can report exactly what the floor removed. Each list is
        indexed 0..n-1 independently; only ``indexed_chunks`` ids reach Qdrant.
        """
        scored = [
            (seg, self._counter.count(paper.text[seg.char_start : seg.char_end]))
            for seg in self._merged_segments(paper)
        ]
        kept_spans = [seg for seg, n_tokens in scored if n_tokens >= self._min_indexed_tokens]
        dropped_spans = [seg for seg, n_tokens in scored if n_tokens < self._min_indexed_tokens]

        kept = [
            make_chunk(paper, self.name, i, seg.char_start, seg.char_end, self._counter)
            for i, seg in enumerate(kept_spans)
        ]
        dropped = [
            make_chunk(paper, self.name, i, seg.char_start, seg.char_end, self._counter)
            for i, seg in enumerate(dropped_spans)
        ]
        return kept, dropped
