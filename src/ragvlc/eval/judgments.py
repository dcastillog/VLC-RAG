"""Recording relevance judgments.

**Judgments are character spans in the normalized text, never chunk ids.** A
``fixed`` chunk and a ``section_aware`` chunk that cover the same passage are
different objects with different ids, but the *text* a judge looked at is the
same -- so a judgment stored as ``(char_start, char_end)`` in the paper's
frozen text applies to whichever chunker's chunk covers that range. Storing
chunk ids instead would make the two chunkers un-comparable: every judgment
would belong to exactly one of them.

Two destinations:

* **relevant** -> appended to that question's ``gold_spans`` in
  ``questions.v2.jsonl`` as ``{"text", "char_start", "char_end", "paper_id",
  "section_heading", "source": "pooled"}``. ``paper_id`` is required: offsets
  are per-paper and all start at zero, so a span with no paper could match a
  chunk from a different paper at the same offsets. It also lets a "negative"
  question that turns out answerable carry a span at all.
* **not relevant** -> a line in ``judgments.jsonl`` (``qid``, ``paper_id``,
  char range, ``relevant: false``), so a resumed session skips it instead of
  asking again.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from ragvlc.eval.pool import PoolItem
from ragvlc.eval.store import read_jsonl

Judged = Literal["relevant", "negative"]


def gold_span_record(item: PoolItem) -> dict:
    """The ``gold_spans`` entry appended when an item is judged relevant."""
    return {
        "text": item.text,
        "char_start": item.char_start,
        "char_end": item.char_end,
        "paper_id": item.paper_id,
        "section_heading": item.section_heading,
        "source": "pooled",
    }


def negative_record(item: PoolItem) -> dict:
    """The ``judgments.jsonl`` line appended when an item is judged not-relevant."""
    return {
        "qid": item.qid,
        "paper_id": item.paper_id,
        "char_start": item.char_start,
        "char_end": item.char_end,
        "section_heading": item.section_heading,
        "text": item.text,
        "relevant": False,
    }


def load_negatives(judgments_jsonl: Path) -> list[dict]:
    return read_jsonl(judgments_jsonl)


def negative_span_keys(negatives: list[dict]) -> set[tuple[str, str, int, int]]:
    """``(qid, paper_id, char_start, char_end)`` for every recorded negative."""
    return {
        (n["qid"], n["paper_id"], n["char_start"], n["char_end"])
        for n in negatives
        if not n.get("relevant", False)
    }


def is_judged(
    item: PoolItem,
    question_row: dict,
    negative_keys: set[tuple[str, str, int, int]],
) -> Judged | None:
    """Has this pool item already been judged (this run or a previous one)?

    An item matches an existing ``gold_span`` by its ``(char_start, char_end)``
    -- the pool is deterministic, so the same span comes back identically each
    run. Manual seeds match their own manual gold span, so they are treated as
    already-relevant and never re-presented.
    """
    for span in question_row.get("gold_spans", []):
        if (
            span.get("paper_id", item.paper_id) == item.paper_id
            and span["char_start"] == item.char_start
            and span["char_end"] == item.char_end
        ):
            return "relevant"
    if (item.qid, item.paper_id, item.char_start, item.char_end) in negative_keys:
        return "negative"
    return None


def remove_relevant_span(
    question_row: dict, paper_id: str, char_start: int, char_end: int
) -> bool:
    """Undo: drop the pooled gold span at ``(paper_id, char_start, char_end)``.
    Returns whether one was removed. Manual spans are left alone."""
    spans = question_row.get("gold_spans", [])
    for i, span in enumerate(spans):
        if (
            span.get("paper_id", paper_id) == paper_id
            and span["char_start"] == char_start
            and span["char_end"] == char_end
            and span.get("source") != "manual"
        ):
            del spans[i]
            return True
    return False
