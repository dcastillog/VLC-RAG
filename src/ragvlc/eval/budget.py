"""success@budget: success measured against cumulative retrieved *tokens*
instead of rank.

success@k treats a hit at rank 3 the same whether it is a 50-token caption or
a 400-token fixed window -- but a downstream reader (or an LLM context
window) pays for the latter in tokens, not in rank. At a fixed k=10,
``fixed``'s chunks sit close to the 400-token budget on almost every hit while
``section_aware``'s vary a lot (captions, merged small units, unsplit
sections), so comparing success@10 as-is conflates "found a relevant chunk"
with "handed the reader more text to search through". success@budget puts
both chunkers on the same x-axis -- tokens spent, not chunks returned -- which
is the comparison that actually isolates chunking-strategy quality.

Computed entirely from data already on disk once ``scripts/run_eval.py`` has
run -- each question's ranked hits and reciprocal rank (``results/
raw_results.json``, or the in-memory results that produce it) plus each
chunk's token count (``data/chunks/{chunker}.jsonl``). No re-ingestion, no
re-running retrieval.
"""

from __future__ import annotations

import math

HitSpan = tuple[str, int, int]


def token_lookup(chunk_records: list[dict]) -> dict[HitSpan, int]:
    """``(paper_id, char_start, char_end) -> n_tokens`` from one chunker's
    ``data/chunks/{chunker}.jsonl`` records."""
    return {(r["paper_id"], r["char_start"], r["char_end"]): r["n_tokens"] for r in chunk_records}


def first_relevant_rank(rr: float) -> int | None:
    """Recover the 1-indexed rank of the first relevant hit from its
    reciprocal rank. ``reciprocal_rank`` (ragvlc.eval.metrics) returns exactly
    ``1/rank``, or ``0.0`` when nothing relevant was retrieved -- so this is
    an exact inverse, not an estimate."""
    if rr <= 0:
        return None
    return round(1 / rr)


def cumulative_tokens_to_first_relevant(
    hit_spans: list[HitSpan], rr: float, tokens: dict[HitSpan, int]
) -> float:
    """Sum of ``n_tokens`` over ``hit_spans[:rank]``, where ``rank`` is the
    first relevant hit's 1-indexed rank -- i.e. the token cost of reading
    down to (and including) the first relevant chunk. ``+inf`` if no hit in
    the retrieved window was relevant (never succeeds, at any budget, within
    what was retrieved -- the same convention success@k uses for "not in the
    top k").

    A hit missing from ``tokens`` (should not happen -- every retrieved chunk
    came from the chunk file being looked up) contributes 0 rather than
    raising, so a stale lookup degrades the curve instead of crashing it.
    """
    rank = first_relevant_rank(rr)
    if rank is None:
        return math.inf
    return sum(tokens.get(tuple(hit), 0) for hit in hit_spans[:rank])


def success_at_budget_curve(first_relevant_tokens: list[float], budgets: list[int]) -> list[float]:
    """Mean success@budget over questions, one value per entry in ``budgets``:
    the fraction of questions whose first relevant hit was reached within
    that many cumulative tokens."""
    if not first_relevant_tokens:
        return [0.0 for _ in budgets]
    n = len(first_relevant_tokens)
    return [sum(1 for t in first_relevant_tokens if t <= budget) / n for budget in budgets]
