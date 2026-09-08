"""Stage 4: a grounded answer layer on top of retrieval.

* :class:`~ragvlc.generation.client.LLMClient` -- a thin POST to an
  OpenAI-compatible ``/chat/completions`` endpoint (local Ollama by default;
  a hosted provider is a ``.env`` change).
* :func:`~ragvlc.generation.answer.answer` -- retrieve via the existing
  ``Searcher``, stuff a numbered context block, generate, and verify the
  citations the model produced.
"""

from __future__ import annotations

from ragvlc.generation.answer import (
    ABSTAIN_MARKER,
    Answer,
    Citation,
    answer,
    build_context_block,
    detect_abstention,
    parse_citations,
    resolve_citations,
    strip_reasoning_block,
)
from ragvlc.generation.client import ChatCompletion, GenerationError, LLMClient

__all__ = [
    "ABSTAIN_MARKER",
    "Answer",
    "Citation",
    "answer",
    "build_context_block",
    "detect_abstention",
    "parse_citations",
    "resolve_citations",
    "strip_reasoning_block",
    "ChatCompletion",
    "GenerationError",
    "LLMClient",
]
