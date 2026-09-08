"""Run every retrieval configuration over the answerable questions and score it.

Retrieval here is **unfiltered** -- the whole collection, no `paper_id`
restriction -- unlike Phase E's pooling. That asymmetry is deliberate and the
two are still consistent: Phase E filtered pooling only to avoid judging
chunks that can never be relevant (they're from the wrong paper, so the
relevance rule scores them 0 automatically); every chunk an unfiltered search
can return is therefore either in the judged pool (scored against its gold
spans) or from the wrong paper (scored not-relevant for free by
`ragvlc.eval.relevance`, no judgment needed). Filtering retrieval itself would
hand the evaluator the answer's location and inflate every metric.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ragvlc.config import EvalConfig
from ragvlc.eval.metrics import HitSpan, reciprocal_rank, recall_at_k, success_at_k
from ragvlc.eval.pool import CONFIGS
from ragvlc.retrieval import Searcher


@dataclass(frozen=True)
class QuestionResult:
    qid: str
    hit_spans: list[HitSpan]
    rr: float
    success: dict[int, bool]
    recall: dict[int, float]


@dataclass
class ConfigResults:
    chunker: str
    mode: str
    collection: str
    by_qid: dict[str, QuestionResult] = field(default_factory=dict)

    def rr_array(self, qids: list[str]) -> np.ndarray:
        return np.array([self.by_qid[q].rr for q in qids], dtype=float)

    def success_array(self, k: int, qids: list[str]) -> np.ndarray:
        return np.array([float(self.by_qid[q].success[k]) for q in qids], dtype=float)

    def recall_array(self, k: int, qids: list[str]) -> np.ndarray:
        return np.array([self.by_qid[q].recall[k] for q in qids], dtype=float)

    def mean_rr(self) -> float:
        return statistics.mean(q.rr for q in self.by_qid.values()) if self.by_qid else 0.0

    def mean_success(self, k: int) -> float:
        return statistics.mean(q.success[k] for q in self.by_qid.values()) if self.by_qid else 0.0

    def mean_recall(self, k: int) -> float:
        return statistics.mean(q.recall[k] for q in self.by_qid.values()) if self.by_qid else 0.0


def evaluate_config(
    searcher: Searcher,
    collection: str,
    chunker: str,
    mode: str,
    questions: list[dict],
    eval_config: EvalConfig,
    *,
    rrf_k: int | None = None,
) -> ConfigResults:
    """Run one (chunker, mode) configuration over every answerable question.

    ``rrf_k`` overrides the searcher's configured RRF rank constant for this
    run only (mode ``hybrid_rrf`` only) -- Phase G's k sweep reuses this
    rather than a separate code path.
    """
    k_max = max([*eval_config.success_at_k, eval_config.mrr_k])
    results = ConfigResults(chunker=chunker, mode=mode, collection=collection)

    for question in questions:
        hits, _timings = searcher.search(collection, question["question"], mode, limit=k_max, rrf_k=rrf_k)
        hit_spans: list[HitSpan] = [
            (h.payload["paper_id"], h.payload["char_start"], h.payload["char_end"]) for h in hits
        ]
        gold_spans = question["gold_spans"]
        threshold = eval_config.relevance_threshold
        results.by_qid[question["qid"]] = QuestionResult(
            qid=question["qid"],
            hit_spans=hit_spans,
            rr=reciprocal_rank(hit_spans, gold_spans, threshold),
            success={k: success_at_k(hit_spans, gold_spans, k, threshold) for k in eval_config.success_at_k},
            recall={k: recall_at_k(hit_spans, gold_spans, k, threshold) for k in eval_config.success_at_k},
        )
    return results


def evaluate_all(
    searcher: Searcher,
    collections: dict[str, str],
    questions: list[dict],
    eval_config: EvalConfig,
    *,
    on_progress: Callable[[int, int], None] = lambda done, total: None,
) -> dict[tuple[str, str], ConfigResults]:
    """Every (chunker, mode) configuration, over every answerable question in
    `questions` (callers filter to `has_answer: true` first)."""
    out: dict[tuple[str, str], ConfigResults] = {}
    for i, (chunker, mode) in enumerate(CONFIGS, start=1):
        out[(chunker, mode)] = evaluate_config(searcher, collections[chunker], chunker, mode, questions, eval_config)
        on_progress(i, len(CONFIGS))
    return out
