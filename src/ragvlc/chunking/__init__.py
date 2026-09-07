"""Stage 2 chunking: turn a frozen normalized paper into retrievable chunks.

Two strategies, one interface (:class:`~ragvlc.chunking.base.Chunker`):

* ``fixed`` -- structure-blind sliding window (the naive baseline).
* ``section_aware`` -- respects unit and section boundaries.

Callers get them via :func:`get_chunkers`, which wires each one from
``config.chunking`` and a shared :class:`~ragvlc.chunking.tokenizer.TokenCounter`
so the whole run measures tokens against one tokenizer instance.
"""

from __future__ import annotations

from ragvlc.chunking.base import Chunk, Chunker, ChunkOffsetError, Paper, Unit
from ragvlc.chunking.fixed import FixedChunker
from ragvlc.chunking.section_aware import SectionAwareChunker
from ragvlc.chunking.tokenizer import TokenCounter
from ragvlc.config import ChunkingConfig

__all__ = [
    "Chunk",
    "Chunker",
    "ChunkOffsetError",
    "Paper",
    "Unit",
    "TokenCounter",
    "FixedChunker",
    "SectionAwareChunker",
    "get_chunkers",
]


def get_chunkers(config: ChunkingConfig, counter: TokenCounter | None = None) -> list[Chunker]:
    """Return both chunkers, configured. Order is stable but callers must not
    depend on it -- the eval runner treats them as an unlabelled set."""
    counter = counter or TokenCounter(config.tokenizer_model)
    return [
        FixedChunker(
            counter,
            max_tokens=config.max_tokens,
            overlap_tokens=config.fixed.overlap_tokens,
        ),
        SectionAwareChunker(
            counter,
            max_tokens=config.max_tokens,
            min_chunk_tokens=config.section_aware.min_chunk_tokens,
            min_indexed_tokens=config.section_aware.min_indexed_tokens,
        ),
    ]
