"""The one relevance rule, used everywhere a chunk is scored against a gold span.

A retrieved chunk and a judged gold span are both character ranges in the same
paper's frozen normalized text. The chunk is **relevant** to the span when

    overlap_chars / min(len(chunk), len(gold_span)) >= threshold      (default 0.5)

Dividing by the *shorter* of the two lengths makes the rule symmetric, which
matters because the two chunkers produce very different span sizes and a
judged gold span may have come from either:

* a short ``section_aware`` chunk fully inside a long retrieved chunk -> 1.0
* a long retrieved chunk fully covering a short gold span            -> 1.0

Dividing by the gold span length instead (the asymmetric rule) would make a
``section_aware`` chunk unable to ever reach 0.5 against a gold span that was
judged from a ~1800-char ``fixed`` chunk, so ``section_aware`` would score
near-zero for reasons unrelated to chunking quality.

Different papers' offsets all start at zero, so overlap is only computed
between ranges in the **same paper** -- callers must check ``paper_id`` first
(see :func:`chunk_matches_any`).
"""

from __future__ import annotations

from collections.abc import Iterable


def overlap_chars(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Character overlap of ``[a_start, a_end)`` and ``[b_start, b_end)``."""
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def relevance_ratio(
    chunk_start: int, chunk_end: int, gold_start: int, gold_end: int
) -> float:
    """Overlap as a fraction of the shorter range's length (0.0..1.0)."""
    shorter = min(chunk_end - chunk_start, gold_end - gold_start)
    if shorter <= 0:
        return 0.0
    return overlap_chars(chunk_start, chunk_end, gold_start, gold_end) / shorter


def is_relevant(
    chunk_start: int,
    chunk_end: int,
    gold_start: int,
    gold_end: int,
    threshold: float = 0.5,
) -> bool:
    """Whether a chunk is relevant to a single gold span (same paper assumed)."""
    return relevance_ratio(chunk_start, chunk_end, gold_start, gold_end) >= threshold


def chunk_matches_any(
    chunk_paper_id: str,
    chunk_start: int,
    chunk_end: int,
    gold_spans: Iterable[dict],
    threshold: float = 0.5,
) -> bool:
    """Whether a chunk is relevant to *any* of a question's gold spans.

    Each gold span dict carries ``paper_id`` (Phase E onwards), ``char_start``
    and ``char_end``. A span in a different paper never matches.
    """
    for span in gold_spans:
        if span.get("paper_id") != chunk_paper_id:
            continue
        if is_relevant(chunk_start, chunk_end, span["char_start"], span["char_end"], threshold):
            return True
    return False
