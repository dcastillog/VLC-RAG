"""Per-question retrieval metrics, all built on the one relevance rule
(:mod:`ragvlc.eval.relevance`).

A ranked result list is a list of ``(paper_id, char_start, char_end)`` triples
in rank order (rank 1 first) -- exactly what a chunk hit's payload gives.
Every function here takes that plus a question's ``gold_spans`` (each an
answerable question's own list; callers must not call these on a
``has_answer: false`` question, since "relevant" is undefined without a gold
span to compare against).

* **success@k** -- did *any* relevant chunk appear in the top k? Binary per
  question; averaged across questions it is the fraction of questions with at
  least one hit. Insensitive to where in the top k the hit landed.
* **recall@k** -- what fraction of the question's distinct gold spans were
  covered by *some* chunk in the top k? Meaningful once `gold_spans` is a list
  (Phase E's pooled judging usually adds more than one).
* **MRR** (primary metric) -- reciprocal rank of the first relevant chunk, 0
  if none. Unlike success@k it is sensitive to ordering.

nDCG is deliberately not implemented: with binary relevance it is a monotone
transform of reciprocal rank and adds a column without adding information.
"""

from __future__ import annotations

from ragvlc.eval.relevance import is_relevant

HitSpan = tuple[str, int, int]  # (paper_id, char_start, char_end)


def _relevant_mask(hit_spans: list[HitSpan], gold_spans: list[dict], threshold: float) -> list[bool]:
    return [
        any(
            span["paper_id"] == paper_id
            and is_relevant(char_start, char_end, span["char_start"], span["char_end"], threshold)
            for span in gold_spans
        )
        for paper_id, char_start, char_end in hit_spans
    ]


def reciprocal_rank(hit_spans: list[HitSpan], gold_spans: list[dict], threshold: float = 0.5) -> float:
    """1 / (rank of the first relevant hit), 1-indexed; 0.0 if none relevant."""
    for rank, relevant in enumerate(_relevant_mask(hit_spans, gold_spans, threshold), start=1):
        if relevant:
            return 1.0 / rank
    return 0.0


def success_at_k(hit_spans: list[HitSpan], gold_spans: list[dict], k: int, threshold: float = 0.5) -> bool:
    """Whether at least one of the top-k hits is relevant to any gold span."""
    return any(_relevant_mask(hit_spans[:k], gold_spans, threshold))


def recall_at_k(hit_spans: list[HitSpan], gold_spans: list[dict], k: int, threshold: float = 0.5) -> float:
    """Fraction of `gold_spans` covered by at least one hit in the top k.

    A gold span is "found" if any top-k hit is relevant to it specifically
    (not just to the question overall) -- so two hits that both cover the same
    gold span still count that span once.
    """
    if not gold_spans:
        return 0.0
    found = set()
    for paper_id, char_start, char_end in hit_spans[:k]:
        for i, span in enumerate(gold_spans):
            if span["paper_id"] == paper_id and is_relevant(
                char_start, char_end, span["char_start"], span["char_end"], threshold
            ):
                found.add(i)
    return len(found) / len(gold_spans)
