"""``fixed`` -- the structure-blind baseline chunker.

A sliding window over the whole normalized text: ``max_tokens`` per chunk,
``overlap_tokens`` shared with the next one, cut only at whitespace so no
chunk starts or ends mid-word. It ignores sections, captions, everything --
this is deliberately the naive approach most RAG projects ship, and the point
of having it is to measure what section awareness actually buys.

The overlap exists so a passage that straddles a window boundary still lands
whole inside at least one chunk; without it, exactly the sentences at the cut
points would be unretrievable.
"""

from __future__ import annotations

import re

from ragvlc.chunking.base import Chunk, Paper, make_chunk, pack_spans
from ragvlc.chunking.tokenizer import TokenCounter

# A "word" is any run of non-whitespace. Using match spans (not str.split)
# keeps every chunk boundary on a real character offset in the source text.
_WORD_RE = re.compile(r"\S+")


class FixedChunker:
    name = "fixed"

    def __init__(self, counter: TokenCounter, max_tokens: int, overlap_tokens: int) -> None:
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        self._counter = counter
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens

    def chunk(self, paper: Paper) -> list[Chunk]:
        words = [(m.start(), m.end()) for m in _WORD_RE.finditer(paper.text)]
        if not words:
            return []

        # Per-word content-token counts; summing these is exact for WordPiece
        # (see pack_spans). Budget leaves room for [CLS]/[SEP] so the final
        # encoded length -- recorded as n_tokens -- stays within max_tokens.
        word_tokens = [self._counter.count_content(paper.text[s:e]) for s, e in words]
        content_budget = self._max_tokens - self._counter.special_token_overhead()

        windows = pack_spans(
            words,
            word_tokens,
            content_budget=content_budget,
            overlap_tokens=self._overlap_tokens,
        )

        return [
            make_chunk(paper, self.name, i, start, end, self._counter)
            for i, (start, end) in enumerate(windows)
        ]
