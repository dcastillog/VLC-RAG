"""Atomic, order-preserving JSONL I/O for the ``data/eval/`` files.

Three files live here:

* ``questions.v2.jsonl`` -- the eval set. ``gold_spans`` grows as judging
  proceeds; everything else is preserved byte-for-byte.
* ``pool.jsonl``         -- the judging pool, built once in Phase E.
* ``judgments.jsonl``    -- explicit negative judgments, so a resumed judging
  session does not re-present items already marked not-relevant.

Every write goes through a temp file + ``os.replace`` so an interrupted run
(``judge.py`` saves after every keystroke) can never leave a half-written file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL file into a list of dicts. Missing file -> ``[]``.

    Raises ``ValueError`` (naming the line) on malformed JSON rather than
    skipping it -- a corrupt eval file should stop the run, not be silently
    dropped.
    """
    if not path.is_file():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Rewrite ``path`` atomically. ``ensure_ascii=False`` keeps non-ASCII text
    readable and byte-identical to how it was written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    os.replace(tmp_path, path)


# The eval set is just a JSONL file; these names exist so call sites read
# clearly ("read the questions", not "read a jsonl that happens to be them").
read_questions = read_jsonl


def write_questions(path: Path, rows: list[dict]) -> None:
    write_jsonl(path, rows)
