"""Tests for judgment recording, the resume check, and reconcile.

All pure -- judgments are plain dicts and character spans, so none of this
needs Qdrant or the filesystem.
"""

from __future__ import annotations

from ragvlc.eval.judgments import (
    gold_span_record,
    is_judged,
    negative_record,
    negative_span_keys,
    remove_relevant_span,
)
from ragvlc.eval.pool import PoolItem
from ragvlc.eval.reconcile import reconcile_has_answer


def _item(qid: str, start: int, end: int, seed: str = "pooled") -> PoolItem:
    return PoolItem(
        qid=qid, paper_id="p1", char_start=start, char_end=end,
        text="some text", section_heading="Method", configs={"fixed/dense"}, seed=seed,
    )


def test_records_have_the_documented_shape():
    item = _item("q1", 100, 500)
    gs = gold_span_record(item)
    assert gs == {
        "text": "some text", "char_start": 100, "char_end": 500,
        "paper_id": "p1", "section_heading": "Method", "source": "pooled",
    }
    neg = negative_record(item)
    assert neg["qid"] == "q1" and neg["paper_id"] == "p1" and neg["relevant"] is False
    assert (neg["char_start"], neg["char_end"]) == (100, 500)


def test_is_judged_matches_gold_spans_and_negatives():
    question = {
        "qid": "q1",
        "gold_spans": [{"paper_id": "p1", "char_start": 100, "char_end": 500, "source": "pooled"}],
    }
    neg_keys = negative_span_keys([
        {"qid": "q1", "paper_id": "p1", "char_start": 900, "char_end": 1200, "relevant": False},
    ])
    assert is_judged(_item("q1", 100, 500), question, neg_keys) == "relevant"
    assert is_judged(_item("q1", 900, 1200), question, neg_keys) == "negative"
    assert is_judged(_item("q1", 100, 501), question, neg_keys) is None


def test_is_judged_requires_matching_paper():
    question = {"qid": "q1", "gold_spans": [{"paper_id": "other", "char_start": 100, "char_end": 500, "source": "pooled"}]}
    assert is_judged(_item("q1", 100, 500), question, set()) is None  # same offsets, wrong paper


def test_manual_span_reads_as_already_relevant():
    question = {"qid": "q1", "gold_spans": [{"paper_id": "p1", "char_start": 42, "char_end": 99, "source": "manual"}]}
    assert is_judged(_item("q1", 42, 99), question, set()) == "relevant"


def test_remove_relevant_span_spares_manual():
    question = {
        "qid": "q1",
        "gold_spans": [
            {"paper_id": "p1", "char_start": 1, "char_end": 2, "source": "manual"},
            {"paper_id": "p1", "char_start": 10, "char_end": 20, "source": "pooled"},
        ],
    }
    assert remove_relevant_span(question, "p1", 10, 20) is True
    assert remove_relevant_span(question, "p1", 1, 2) is False  # manual is untouchable
    assert [s["source"] for s in question["gold_spans"]] == ["manual"]


# --------------------------------------------------------------------------- #
# reconcile
# --------------------------------------------------------------------------- #
def _pool_row(qid: str, start: int, end: int) -> dict:
    return {"qid": qid, "paper_id": "p1", "char_start": start, "char_end": end}


def test_reconcile_flips_only_fully_judged_questions_with_no_relevant_span():
    rows = [
        {"qid": "q1", "has_answer": True, "gold_spans": [{"char_start": 1, "char_end": 2, "source": "pooled"}]},
        {"qid": "q2", "has_answer": True, "gold_spans": []},   # pool fully judged negative -> flip
        {"qid": "q3", "has_answer": True, "gold_spans": []},   # one item still un-judged -> leave
        {"qid": "n1", "has_answer": False, "gold_spans": []},  # no relevant judgments -> matches its prior
    ]
    pool_rows = [
        _pool_row("q1", 1, 2),
        _pool_row("q2", 10, 20), _pool_row("q2", 30, 40),
        _pool_row("q3", 50, 60), _pool_row("q3", 70, 80),
        _pool_row("n1", 90, 99),
    ]
    negatives = [
        {"qid": "q2", "paper_id": "p1", "char_start": 10, "char_end": 20, "relevant": False},
        {"qid": "q2", "paper_id": "p1", "char_start": 30, "char_end": 40, "relevant": False},
        {"qid": "q3", "paper_id": "p1", "char_start": 50, "char_end": 60, "relevant": False},
    ]

    result = reconcile_has_answer(rows, pool_rows, negatives)

    assert result.flipped_to_false == ["q2"]
    assert result.corrected_to_true == []
    assert result.confirmed_true == 1   # q1
    assert result.confirmed_false == 1  # n1
    assert result.incomplete == [("q3", 1)]
    assert rows[1]["has_answer"] is False
    assert rows[2]["has_answer"] is True  # q3 untouched
    assert not result.ok  # incomplete work remains


def test_reconcile_corrects_has_answer_to_true_whenever_a_gold_span_exists():
    """The self-healing rule: a gold_span always wins, regardless of prefix or
    of how has_answer got out of sync (a q question a previous run flipped to
    false before a manual span was added; an n question that turned out to
    have a real answer)."""
    rows = [
        {"qid": "q23", "has_answer": False, "gold_spans": [
            {"paper_id": "p1", "char_start": 5, "char_end": 9, "source": "manual"}
        ]},
        {"qid": "n3", "has_answer": False, "gold_spans": [
            {"paper_id": "p1", "char_start": 20, "char_end": 30, "source": "pooled"}
        ]},
    ]

    result = reconcile_has_answer(rows, pool_rows=[], negatives=[])

    assert set(result.corrected_to_true) == {"q23", "n3"}
    assert rows[0]["has_answer"] is True
    assert rows[1]["has_answer"] is True
    # a gold_span is decisive on its own -- no pool/negatives data was even needed
    assert result.incomplete == []
