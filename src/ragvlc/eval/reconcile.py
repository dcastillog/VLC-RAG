"""Post-judging reconciliation of ``has_answer``.

``has_answer`` starts as a prior: ``true`` for every ``q...`` question,
``false`` for every ``n...``. Judging can prove the prior wrong in either
direction:

* a ``q`` question whose entire pool was judged and turned up nothing
  relevant is not actually answerable from the corpus -> flip to ``false``.
* **any** question that has a recorded relevant span (a hand-verified manual
  span, or a pooled item judged ``y``) *is* answerable -> ``has_answer`` must
  be ``true``, unconditionally. This covers a ``q`` question a previous run
  flipped to ``false`` before a manual span was added, and an ``n`` question
  that turned out to have a real answer -- either way, "there is a recorded
  relevant span" is definitive and overrides any prior in either direction.

That second rule is checked **first and unconditionally**, before anything
prefix- or pool-based: a question with ``gold_spans`` can never leave this
function with ``has_answer: false``, regardless of what its prior was or how
it got here. That is what makes this self-healing rather than merely correct
on a clean run -- a `q001`-was-flipped-then-a-span-was-added-by-hand history
gets fixed the next time this runs, not just prevented going forward.

The false-direction flip only happens once a `q` question's pool is *fully*
judged. "Not judged yet" and "judged, nothing relevant" must not be
conflated -- the first is in-progress, the second is a finding.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ReconcileResult:
    corrected_to_true: list[str] = field(default_factory=list)  # had a gold_span but has_answer wasn't true
    flipped_to_false: list[str] = field(default_factory=list)   # q question, pool judged, nothing relevant
    incomplete: list[tuple[str, int]] = field(default_factory=list)  # (qid, n unjudged pool items)
    confirmed_true: int = 0   # already consistent: gold_spans present, has_answer already true
    confirmed_false: int = 0  # already consistent: no gold_spans, has_answer already false

    @property
    def ok(self) -> bool:
        """True when every q question's pool is fully judged."""
        return not self.incomplete


def reconcile_has_answer(
    questions: list[dict], pool_rows: list[dict], negatives: list[dict]
) -> ReconcileResult:
    """Reconcile every question's `has_answer` against its judged gold_spans.
    Mutates `questions`. See the module docstring for the two rules."""
    result = ReconcileResult()

    pool_by_qid: dict[str, list[dict]] = {}
    for item in pool_rows:
        pool_by_qid.setdefault(item["qid"], []).append(item)

    negative_spans: dict[str, set[tuple[str, int, int]]] = {}
    for neg in negatives:
        if not neg.get("relevant", False):
            negative_spans.setdefault(neg["qid"], set()).add(
                (neg["paper_id"], neg["char_start"], neg["char_end"])
            )

    for row in questions:
        qid = str(row.get("qid", ""))

        # Rule 1, unconditional: a recorded relevant span means answerable.
        # Checked before any prefix or pool logic so it can never be
        # overridden by the false-direction flip below.
        if row.get("gold_spans"):
            if row.get("has_answer") is not True:
                row["has_answer"] = True
                result.corrected_to_true.append(qid)
            else:
                result.confirmed_true += 1
            continue

        # Rule 2: only a `q` question with no gold_spans and a fully-judged
        # pool can be flipped to false. `n` questions with no relevant
        # judgments simply match their prior -- nothing to do.
        if not qid.startswith("q"):
            result.confirmed_false += 1
            continue

        judged = negative_spans.get(qid, set())
        unjudged = [
            it for it in pool_by_qid.get(qid, [])
            if (it["paper_id"], it["char_start"], it["char_end"]) not in judged
        ]
        if unjudged:
            result.incomplete.append((qid, len(unjudged)))
            continue

        if row.get("has_answer") is not False:
            row["has_answer"] = False
            result.flipped_to_false.append(qid)
        else:
            result.confirmed_false += 1

    return result
