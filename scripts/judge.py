"""CLI: judge the pool, one item at a time.

    uv run python scripts/judge.py            # all questions, resumes where it left off
    uv run python scripts/judge.py --qid q014

Keys:  y relevant   n not relevant   s skip   u undo last   q save and quit

Every ``y`` / ``n`` is written to disk immediately, so the session is fully
resumable -- on restart, items already judged (this run or before) are skipped.
A ``y`` appends a character span to that question's ``gold_spans`` in
``questions.v2.jsonl``; an ``n`` appends a line to ``judgments.jsonl``.

Two things this tool deliberately hides / randomises to keep the judgment
honest (see ragvlc.eval.pool / .judgments for the full reasoning):

* item order within a question is shuffled -- position carries no signal;
* the configs that retrieved an item are never shown -- seeing "hybrid found
  this" biases the call toward whichever system you expect to win, which is
  exactly the comparison this eval exists to make.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import textwrap
from dataclasses import dataclass

from ragvlc.config import get_paths
from ragvlc.eval.judgments import (
    gold_span_record,
    is_judged,
    negative_record,
    negative_span_keys,
    remove_relevant_span,
)
from ragvlc.eval.pool import PoolItem
from ragvlc.eval.store import read_jsonl, read_questions, write_jsonl, write_questions

_RULE = "─" * 72


# --------------------------------------------------------------------------- #
# Single-keystroke input
# --------------------------------------------------------------------------- #
def _read_key() -> str:
    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        return line.strip()[:1].lower() if line else "q"
    try:
        import termios
        import tty
    except ImportError:  # non-POSIX -- fall back to line input
        return (input().strip()[:1] or "").lower()

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    if ch in ("\x03", "\x04"):  # Ctrl-C / Ctrl-D
        raise KeyboardInterrupt
    return ch.lower()


# --------------------------------------------------------------------------- #
@dataclass
class Task:
    qid: str
    question_row: dict
    item: PoolItem
    q_ordinal: int      # 1-based position of this question among the selected ones
    q_total: int        # unjudged items for this question at session start
    q_index: int        # 1-based position of this item within that


@dataclass
class LogEntry:
    pos: int
    task: Task
    verdict: str  # "relevant" | "negative"


def _pool_items_by_qid(pool_rows: list[dict]) -> dict[str, list[PoolItem]]:
    by_qid: dict[str, list[PoolItem]] = {}
    for row in pool_rows:
        item = PoolItem(
            qid=row["qid"],
            paper_id=row["paper_id"],
            char_start=row["char_start"],
            char_end=row["char_end"],
            text=row["text"],
            section_heading=row.get("section_heading"),
            configs=set(row.get("configs", [])),
            seed=row.get("seed", "pooled"),
        )
        by_qid.setdefault(item.qid, []).append(item)
    return by_qid


def _count_recorded(questions: list[dict], negatives: list[dict]) -> int:
    pooled = sum(
        1 for q in questions for s in q.get("gold_spans", []) if s.get("source") == "pooled"
    )
    return pooled + len(negatives)


def _build_worklist(
    selected_qids: list[str],
    questions_by_qid: dict[str, dict],
    pool_by_qid: dict[str, list[PoolItem]],
    negatives: list[dict],
) -> list[Task]:
    neg_keys = negative_span_keys(negatives)
    worklist: list[Task] = []
    for ordinal, qid in enumerate(selected_qids, start=1):
        question_row = questions_by_qid[qid]
        items = list(pool_by_qid.get(qid, []))
        random.Random(qid).shuffle(items)  # stable across resumes, unbiased order
        pending = [it for it in items if is_judged(it, question_row, neg_keys) is None]
        for index, item in enumerate(pending, start=1):
            worklist.append(
                Task(
                    qid=qid,
                    question_row=question_row,
                    item=item,
                    q_ordinal=ordinal,
                    q_total=len(pending),
                    q_index=index,
                )
            )
    return worklist


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _render(task: Task, n_selected: int, total_recorded: int) -> None:
    width = min(shutil.get_terminal_size((100, 40)).columns, 100)
    item = task.item
    q = task.question_row

    print("\033[2J\033[H", end="")  # clear screen
    print(_RULE)
    print(
        f"question {task.q_ordinal}/{n_selected} · item {task.q_index}/{task.q_total} "
        f"· {total_recorded} judgments total"
    )
    print(_RULE)
    print()
    for line in textwrap.wrap(f"Q ({task.qid}):  {q['question']}", width=width):
        print(line)
    print()

    manual = [s for s in q.get("gold_spans", []) if s.get("source") == "manual"]
    if manual:
        print(f"already-recorded ground truth ({len(manual)} manual span(s)) -- NOT for judging:")
        for i, span in enumerate(manual, start=1):
            snippet = " ".join(span["text"].split())
            if len(snippet) > 220:
                snippet = snippet[:220].rstrip() + "..."
            for j, line in enumerate(textwrap.wrap(snippet, width=width - 6)):
                print(f"   {f'[{i}]' if j == 0 else '   '} {line}")
        print()

    print(f"paper:    {q.get('paper_id')}")
    print(f"section:  {item.section_heading or '-'}")
    print(f"chars:    [{item.char_start}:{item.char_end}]  ({item.length} chars)")
    print()
    for line in textwrap.wrap(" ".join(item.text.split()), width=width) or ["(empty)"]:
        print(f"  {line}")
    print()
    print(_RULE)
    print("[y] relevant   [n] not relevant   [s] skip   [u] undo last   [q] save & quit")


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Judge the retrieval pool.")
    parser.add_argument("--qid", help="judge only this question")
    args = parser.parse_args(argv)

    paths = get_paths()
    if not paths.pool_jsonl.is_file():
        print(f"judge.py: no pool at {paths.pool_jsonl} -- run scripts/build_pool.py", file=sys.stderr)
        return 1

    questions = read_questions(paths.questions_jsonl)
    questions_by_qid = {q["qid"]: q for q in questions}
    pool_by_qid = _pool_items_by_qid(read_jsonl(paths.pool_jsonl))
    negatives = read_jsonl(paths.judgments_jsonl)

    if args.qid:
        if args.qid not in questions_by_qid:
            print(f"judge.py: no such qid {args.qid!r}", file=sys.stderr)
            return 1
        selected_qids = [args.qid]
    else:
        selected_qids = [q["qid"] for q in questions if q["qid"] in pool_by_qid]

    worklist = _build_worklist(selected_qids, questions_by_qid, pool_by_qid, negatives)
    total_recorded = _count_recorded(questions, negatives)

    if not worklist:
        print(f"judge.py: nothing left to judge for {len(selected_qids)} question(s).")
        return 0

    session_log: list[LogEntry] = []
    session_count = 0
    pos = 0

    while pos < len(worklist):
        task = worklist[pos]
        _render(task, len(selected_qids), total_recorded)
        key = _read_key()

        if key == "q":
            break
        if key == "s":
            pos += 1
            continue
        if key == "u":
            if not session_log:
                continue
            entry = session_log.pop()
            if entry.verdict == "relevant":
                remove_relevant_span(
                    entry.task.question_row,
                    entry.task.item.paper_id,
                    entry.task.item.char_start,
                    entry.task.item.char_end,
                )
                write_questions(paths.questions_jsonl, questions)
            else:
                key_tuple = (
                    entry.task.qid, entry.task.item.paper_id,
                    entry.task.item.char_start, entry.task.item.char_end,
                )
                negatives[:] = [
                    n for n in negatives
                    if (n["qid"], n["paper_id"], n["char_start"], n["char_end"]) != key_tuple
                ]
                write_jsonl(paths.judgments_jsonl, negatives)
            total_recorded -= 1
            session_count -= 1
            pos = entry.pos
            continue
        if key in ("y", "n"):
            if key == "y":
                task.question_row.setdefault("gold_spans", []).append(gold_span_record(task.item))
                write_questions(paths.questions_jsonl, questions)
                verdict = "relevant"
            else:
                negatives.append(negative_record(task.item))
                write_jsonl(paths.judgments_jsonl, negatives)
                verdict = "negative"
            session_log.append(LogEntry(pos=pos, task=task, verdict=verdict))
            total_recorded += 1
            session_count += 1
            pos += 1
        # any other key: redraw, no state change

    remaining = sum(1 for _ in worklist[pos:])
    print("\033[2J\033[H", end="")
    print(f"judge.py: {session_count} judgment(s) this session; {remaining} item(s) still pending.")
    print(f"  gold spans + negatives now total {total_recorded}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
