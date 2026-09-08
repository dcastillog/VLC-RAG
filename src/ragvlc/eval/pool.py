"""Build the judging pool.

Every retrieval configuration -- 2 chunkers x 4 modes = 8 -- contributes its
top-k for each question. The union, deduplicated, is what gets judged once;
that judged set is the ground truth every later metric is measured against, so
the pooling rules below all exist to keep it both complete and small enough to
judge by hand.

**Answerable questions are pooled with a `paper_id` filter.** Every question
was written against one specific paper, and ``gold_spans`` are recorded as
offsets into that paper with no ``paper_id`` field. A chunk from any other
paper cannot overlap a gold span, so Phase F's overlap rule scores it
not-relevant for free -- judging it would be pure wasted effort. Negative
questions have no paper, so they are pooled against the whole collection.

**Dedup, in two passes:**

1. Exact: items sharing ``(paper_id, char_start, char_end)`` collapse to one,
   unioning the list of configs that retrieved them.
2. Overlap: if one item covers > 80% of a shorter item's characters, keep only
   the *shorter* -- the two chunkers slice the same passage differently and
   judging near-identical text twice is waste. Keeping the shorter one makes
   the judged gold span the tightest available and keeps Phase F's relevance
   rule fair to both chunkers. A manually-found span always wins this contest
   (it is already ground truth); after that, shorter beats longer.

**Manual seeds:** each question's existing ``source: "manual"`` gold spans are
added to its pool so they are guaranteed present for dedup and for the pool
record, even if no configuration retrieved them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ragvlc.retrieval import Hit, Searcher, build_filter
from ragvlc.retrieval.search import SEARCH_MODES

# (chunker, mode) -- the 8 configurations. The chunker name also selects the
# collection (config.retrieval.qdrant.collections) at the call site.
CONFIGS: tuple[tuple[str, str], ...] = tuple(
    (chunker, mode) for chunker in ("fixed", "section_aware") for mode in SEARCH_MODES
)

_OVERLAP_DROP_RATIO = 0.80
_MANUAL = "manual"
_POOLED = "pooled"


@dataclass
class PoolItem:
    qid: str
    paper_id: str
    char_start: int
    char_end: int
    text: str
    section_heading: str | None
    configs: set[str] = field(default_factory=set)  # e.g. {"fixed/dense", "section_aware/sparse"}
    seed: str = _POOLED  # "pooled" or "manual"

    @property
    def length(self) -> int:
        return self.char_end - self.char_start

    @property
    def item_id(self) -> str:
        return f"{self.qid}::{self.paper_id}::{self.char_start}::{self.char_end}"

    def to_json_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "qid": self.qid,
            "paper_id": self.paper_id,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "section_heading": self.section_heading,
            "seed": self.seed,
            # stored, but judge.py must never show this -- seeing which system
            # retrieved an item biases the judgment toward the expected winner.
            "configs": sorted(self.configs),
            "text": self.text,
        }


@dataclass
class PoolStats:
    n_questions: int
    raw_items: int          # every config hit + every manual seed, before any dedup
    pooled_items: int       # after both dedup passes
    exact_dups_removed: int
    overlap_merged: int
    per_question: list[int]  # pooled item count per question, for the distribution


def _overlap_ratio(a: PoolItem, b: PoolItem) -> float:
    """Intersection as a fraction of the shorter item's length (0..1)."""
    intersection = max(0, min(a.char_end, b.char_end) - max(a.char_start, b.char_start))
    shorter = min(a.length, b.length)
    return intersection / shorter if shorter else 0.0


def _dedupe_question(raw: list[PoolItem]) -> tuple[list[PoolItem], int, int]:
    """(kept, exact_dups_removed, overlap_merged) for one question's raw items."""
    # Pass 1 -- exact (paper_id, char_start, char_end).
    by_span: dict[tuple[str, int, int], PoolItem] = {}
    exact_removed = 0
    for item in raw:
        key = (item.paper_id, item.char_start, item.char_end)
        existing = by_span.get(key)
        if existing is None:
            by_span[key] = item
        else:
            existing.configs |= item.configs
            if item.seed == _MANUAL:
                existing.seed = _MANUAL
            exact_removed += 1

    # Pass 2 -- overlap. Consider items manual-first, then SHORTEST-first, so the
    # survivor of any >80%-overlapping pair is the tighter span. Keeping the
    # longer one would let a fixed chunk (~1800 chars) swallow the
    # section_aware chunk (~500 chars) covering the same passage; the judged
    # gold span would then be fixed-chunk-sized, and Phase F's relevance rule
    # (>=50% char overlap with the shorter of chunk / span) could never credit
    # the section_aware chunk -- it would score near-zero for reasons that have
    # nothing to do with chunking quality. Tighter spans are also faster to read.
    ordered = sorted(by_span.values(), key=lambda it: (it.seed != _MANUAL, it.length))
    kept: list[PoolItem] = []
    overlap_merged = 0
    for item in ordered:
        absorber = next(
            (
                k for k in kept
                if k.paper_id == item.paper_id and _overlap_ratio(k, item) > _OVERLAP_DROP_RATIO
            ),
            None,
        )
        if absorber is None:
            kept.append(item)
        else:
            absorber.configs |= item.configs
            overlap_merged += 1

    kept.sort(key=lambda it: (it.paper_id, it.char_start, it.char_end))
    return kept, exact_removed, overlap_merged


def build_pool(
    searcher: Searcher,
    questions: list[dict],
    collections: dict[str, str],
    *,
    top_k: int = 10,
    on_progress: Callable[[int, int], None] = lambda done, total: None,
) -> tuple[list[PoolItem], PoolStats]:
    """Run every configuration over every question and return the deduped pool
    plus the size statistics.

    ``collections`` maps chunker name -> Qdrant collection name.
    """
    raw_by_qid: dict[str, list[PoolItem]] = {}
    raw_total = 0
    done = 0
    total_searches = len(questions) * len(CONFIGS)

    for question in questions:
        qid = question["qid"]
        query_text = question["question"]
        paper_id = question.get("paper_id")
        # answerable -> restrict to the source paper; negative -> whole collection
        query_filter = build_filter(paper_id=paper_id) if paper_id else None

        items: list[PoolItem] = []
        for chunker, mode in CONFIGS:
            hits, _timings = searcher.search(
                collections[chunker], query_text, mode, limit=top_k, filters=query_filter
            )
            config_label = f"{chunker}/{mode}"
            for hit in hits:
                items.append(_item_from_hit(qid, hit, config_label))
            raw_total += len(hits)
            done += 1
            on_progress(done, total_searches)

        for span in question.get("gold_spans", []):
            if span.get("source") == _MANUAL and paper_id:
                items.append(
                    PoolItem(
                        qid=qid,
                        paper_id=paper_id,
                        char_start=span["char_start"],
                        char_end=span["char_end"],
                        text=span["text"],
                        section_heading=span.get("section_heading"),
                        configs={_MANUAL},
                        seed=_MANUAL,
                    )
                )
                raw_total += 1

        raw_by_qid[qid] = items

    pool: list[PoolItem] = []
    exact_removed_total = 0
    overlap_merged_total = 0
    per_question: list[int] = []
    for question in questions:
        kept, exact_removed, overlap_merged = _dedupe_question(raw_by_qid[question["qid"]])
        pool.extend(kept)
        exact_removed_total += exact_removed
        overlap_merged_total += overlap_merged
        per_question.append(len(kept))

    stats = PoolStats(
        n_questions=len(questions),
        raw_items=raw_total,
        pooled_items=len(pool),
        exact_dups_removed=exact_removed_total,
        overlap_merged=overlap_merged_total,
        per_question=per_question,
    )
    return pool, stats


def _item_from_hit(qid: str, hit: Hit, config_label: str) -> PoolItem:
    payload = hit.payload
    return PoolItem(
        qid=qid,
        paper_id=payload["paper_id"],
        char_start=payload["char_start"],
        char_end=payload["char_end"],
        text=payload["text"],
        section_heading=payload.get("section_heading"),
        configs={config_label},
        seed=_POOLED,
    )
