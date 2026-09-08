"""Grounded answering: retrieve, stuff a numbered context block, generate, verify.

The retrieval half is **not** reimplemented here -- :func:`answer` calls the
existing :class:`ragvlc.retrieval.Searcher`. This module owns only what sits on
top of retrieval:

* build a context block numbering each chunk ``[1]``..``[k]``, each headed by
  its paper title, year and section heading;
* a system prompt that forbids outside knowledge, requires inline ``[n]``
  citations, and asks the model to prefix a fixed marker when it cannot answer;
* post-processing: pull the cited indices back out of the response and check
  each one refers to a context block that was actually supplied. A cited index
  with no matching block is recorded on the result (``invalid_citations``),
  never silently dropped.

Abstention is detected by the marker token, not by string-matching English
refusal phrasing -- that phrasing varies too much across models to rely on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from time import perf_counter

from ragvlc.config import get_experiment
from ragvlc.generation.client import ChatMessage, LLMClient
from ragvlc.retrieval import Hit, Searcher, SearchMode

# The model is told to begin its reply with this exact token when the context
# does not contain the answer. Distinctive enough not to collide with prose.
ABSTAIN_MARKER = "NO_ANSWER_IN_CONTEXT"

_CITATION_RE = re.compile(r"\[(\d+)\]")

# Some OpenAI-compatible servers inline a reasoning model's chain of thought in
# the message content as a leading <think>...</think> block instead of a
# separate field. Strip it so it never reaches the answer text or the citation
# check. (The local Ollama used in development puts it in message.reasoning,
# which we already ignore -- this is the belt-and-braces case.)
_THINK_BLOCK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)

_SYSTEM_PROMPT = f"""\
You answer questions about visible light communication (VLC) using only a set \
of numbered source passages provided by the user.

Rules:
- Use only the numbered passages. Do not add facts from your own knowledge, \
even if you are confident they are correct.
- After each claim, cite the passage(s) it came from inline, like [1] or [2][3].
- Distinguish values the authors measured or observed from values they assumed, \
selected, or set as simulation parameters. Where a passage presents a number as \
a chosen assumption, a worst-case value, or a simulation input, say so \
explicitly rather than reporting it as a measured or established result.
- If the passages do not contain enough information to answer, reply with the \
single token {ABSTAIN_MARKER} on its own, followed by one sentence saying the \
provided sources do not cover the question. Do not guess.
"""


@dataclass(frozen=True)
class Citation:
    """A resolved inline citation: the ``[index]`` the model wrote, mapped back
    to the chunk that occupied that slot in the context block."""

    index: int
    paper_id: str
    title: str | None
    doi: str | None
    char_start: int
    char_end: int


@dataclass(frozen=True)
class Answer:
    text: str  # the model's reply, with the abstention marker stripped if present
    citations: tuple[Citation, ...]  # valid cited indices, resolved, ordered by index
    invalid_citations: tuple[int, ...]  # cited indices with no matching context block
    abstained: bool
    hits: tuple[Hit, ...]  # the retrieved context, in the order it was numbered
    timings: dict  # retrieval Timings fields (ms) + generation_ms


def _heading(payload: dict) -> str:
    return payload.get("section_heading") or payload.get("parent_heading") or "n/a"


def build_context_block(hits: tuple[Hit, ...] | list[Hit]) -> str:
    """Number the hits ``[1]``..``[k]``, each preceded by title / year / section."""
    blocks: list[str] = []
    for i, hit in enumerate(hits, start=1):
        p = hit.payload
        title = p.get("title") or "Untitled"
        year = p.get("year") or "n.d."
        blocks.append(f"[{i}] {title} ({year}) -- {_heading(p)}\n{p.get('text', '')}")
    return "\n\n".join(blocks)


def _build_messages(query: str, context_block: str) -> list[ChatMessage]:
    user = f"Question: {query}\n\nSources:\n\n{context_block}"
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def strip_reasoning_block(text: str) -> str:
    """Drop a leading ``<think>...</think>`` block if the server inlined one."""
    return _THINK_BLOCK_RE.sub("", text, count=1)


def detect_abstention(text: str) -> tuple[bool, str]:
    """``(abstained, cleaned_text)``. Abstention is the marker appearing at the
    start of the reply (after any leading whitespace); the marker is stripped
    from the returned text."""
    stripped = text.lstrip()
    if stripped.startswith(ABSTAIN_MARKER):
        return True, stripped[len(ABSTAIN_MARKER) :].lstrip(" :.-\n")
    return False, text


def parse_citations(text: str, n_hits: int) -> tuple[list[int], list[int]]:
    """``(valid, invalid)`` -- distinct cited indices, split on whether they
    refer to a context block that exists (``1..n_hits``). Order of first
    appearance is preserved."""
    valid: list[int] = []
    invalid: list[int] = []
    seen: set[int] = set()
    for match in _CITATION_RE.finditer(text):
        idx = int(match.group(1))
        if idx in seen:
            continue
        seen.add(idx)
        (valid if 1 <= idx <= n_hits else invalid).append(idx)
    return valid, invalid


def resolve_citations(indices: list[int], hits: tuple[Hit, ...]) -> tuple[Citation, ...]:
    out: list[Citation] = []
    for idx in sorted(indices):
        p = hits[idx - 1].payload
        out.append(
            Citation(
                index=idx,
                paper_id=p.get("paper_id", ""),
                title=p.get("title"),
                doi=p.get("doi"),
                char_start=p.get("char_start", -1),
                char_end=p.get("char_end", -1),
            )
        )
    return tuple(out)


def answer(
    query: str,
    collection: str,
    mode: SearchMode,
    top_k: int,
    *,
    searcher: Searcher,
    client: LLMClient,
) -> Answer:
    """Retrieve ``top_k`` chunks from ``collection`` via ``mode``, ask ``client``
    to answer only from them, and verify the citations it produced.

    ``searcher`` and ``client`` are injected (both hold models / connections
    that Phase C builds once at startup) rather than constructed here.
    """
    hits_list, retrieval_timings = searcher.search(collection, query, mode, limit=top_k)
    hits = tuple(hits_list)

    context_block = build_context_block(hits)
    messages = _build_messages(query, context_block)

    generation_cfg = get_experiment().generation
    started = perf_counter()
    completion = client.complete(
        messages,
        temperature=generation_cfg.temperature,
        max_tokens=generation_cfg.max_tokens,
    )
    generation_ms = (perf_counter() - started) * 1000.0

    abstained, text = detect_abstention(strip_reasoning_block(completion.text))
    valid_indices, invalid_indices = parse_citations(text, len(hits))

    timings = {
        "dense_embed_ms": retrieval_timings.dense_embed,
        "sparse_embed_ms": retrieval_timings.sparse_embed,
        "qdrant_search_ms": retrieval_timings.qdrant_search,
        "retrieval_total_ms": retrieval_timings.total,
        "generation_ms": generation_ms,
    }

    return Answer(
        text=text,
        citations=resolve_citations(valid_indices, hits),
        invalid_citations=tuple(invalid_indices),
        abstained=abstained,
        hits=hits,
        timings=timings,
    )
