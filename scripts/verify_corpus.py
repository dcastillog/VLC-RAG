"""CLI: verify that every paper in data/normalized/ still matches what was
written -- the tripwire against normalization drift once character offsets
have been hand-annotated into an evaluation set.

    uv run python scripts/verify_corpus.py

For every paper (each `data/normalized/{paper_id}.json` + its matching
`.txt`):

1. Re-hash the `.txt` file and compare against `text_sha256` in the JSON.
2. Check `n_chars` in the JSON matches the actual text length.
3. Check every unit's `char_start`/`char_end` still slices to a string of
   exactly `char_end - char_start` characters. Python slicing never raises
   for an out-of-range end index -- it just silently truncates -- so a
   length mismatch is the only reliable way to catch a unit whose range now
   runs past the end of the text.
4. Structural checks over the units, in document order, that pin every
   offset absolutely rather than just its length (a length-only check can't
   tell a correct offset from a same-length one shifted to somewhere else
   entirely valid in the same file):
     - `units[0].char_start == 0`
     - `units[-1].char_end == n_chars`
     - every unit's `char_end > char_start`
     - consecutive units satisfy `next.char_start == prev.char_end +
       len(unit_separator)`
     - the text at each inter-unit gap is exactly `unit_separator`

Run this before every annotation session and at the start of every
evaluation run: once a gold span is recorded as an offset into one of these
files, any change to that file -- even whitespace -- silently invalidates
every span built against it. This script is what turns "silently" into
"this command fails, loudly, right now."

This is deliberately independent of the rest of the pipeline: it re-derives
nothing from `tei.py`/`normalize.py`/`pipeline.py`, doesn't touch GROBID or
Crossref, and only ever reads the two files already on disk per paper -- a
drift check that itself depended on the pipeline being correct wouldn't be
much of a tripwire.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ragvlc.config import get_experiment, get_paths


@dataclass
class PaperIssues:
    paper_id: str
    problems: list[str] = field(default_factory=list)


def _verify_one(json_path: Path, unit_separator: str) -> PaperIssues:
    paper_id = json_path.stem
    issues = PaperIssues(paper_id=paper_id)

    try:
        doc = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        issues.problems.append(f"could not read/parse {json_path.name}: {exc}")
        return issues

    txt_path = json_path.with_suffix(".txt")
    if not txt_path.is_file():
        issues.problems.append(f"missing {txt_path.name}")
        return issues
    raw_bytes = txt_path.read_bytes()

    expected_hash = doc.get("text_sha256")
    actual_hash = hashlib.sha256(raw_bytes).hexdigest()
    if expected_hash is None:
        issues.problems.append("no text_sha256 in the JSON (needs a re-run of parse_corpus.py to populate)")
    elif actual_hash != expected_hash:
        issues.problems.append(f"text_sha256 mismatch: file hashes to {actual_hash}, JSON says {expected_hash}")

    # Decoded from the *same* bytes just hashed, so the length/slice checks
    # below can never disagree with the hash check over what "the text" is.
    text = raw_bytes.decode("utf-8")

    n_chars = doc.get("n_chars")
    if n_chars != len(text):
        issues.problems.append(f"n_chars mismatch: JSON says {n_chars}, actual text has {len(text)} character(s)")

    units = doc.get("units", [])
    last_index = len(units) - 1
    prev_end: int | None = None
    prev_label: str | None = None

    for i, unit in enumerate(units):
        label = f"unit[{i}] ({unit.get('unit_id', '<unknown>')})"
        start, end = unit.get("char_start"), unit.get("char_end")
        if start is None or end is None:
            issues.problems.append(f"{label}: missing char_start/char_end")
            prev_end, prev_label = None, None  # nothing valid to chain the next gap check off
            continue

        # Slice length: catches an out-of-range end (Python truncates rather
        # than raising, so length is the only signal).
        expected_len = end - start
        actual_len = len(text[start:end])
        if actual_len != expected_len:
            issues.problems.append(
                f"{label}: char_start={start}, char_end={end} slices to {actual_len} "
                f"character(s), expected {expected_len}"
            )

        # Structural checks: these pin the offset absolutely, so a
        # shifted-but-in-bounds char_end (same length, wrong position) fails
        # here even though it would pass the length check above.
        if end <= start:
            issues.problems.append(f"{label}: char_end ({end}) is not greater than char_start ({start})")

        if i == 0 and start != 0:
            issues.problems.append(f"{label}: is the first unit but char_start={start}, expected 0")

        if i == last_index and n_chars is not None and end != n_chars:
            issues.problems.append(f"{label}: is the last unit but char_end={end}, expected n_chars={n_chars}")

        if prev_end is not None:
            expected_start = prev_end + len(unit_separator)
            if start != expected_start:
                issues.problems.append(
                    f"{label}: char_start={start} does not follow {prev_label} (char_end={prev_end}) by "
                    f"len(unit_separator)={len(unit_separator)}; expected char_start={expected_start}"
                )
            gap = text[prev_end:start]
            if gap != unit_separator:
                issues.problems.append(
                    f"{label}: text between {prev_label} and this unit is {gap!r}, "
                    f"expected the separator {unit_separator!r}"
                )

        prev_end, prev_label = end, label

    return issues


def main(argv: list[str] | None = None) -> int:
    paths = get_paths()
    unit_separator = get_experiment().parsing.unit_separator

    json_paths = sorted(paths.normalized.glob("*.json"))
    if not json_paths:
        print(f"verify_corpus.py: no papers found in {paths.normalized}", file=sys.stderr)
        return 1

    all_issues = [issues for issues in (_verify_one(p, unit_separator) for p in json_paths) if issues.problems]

    if not all_issues:
        print(f"verify_corpus.py: OK -- {len(json_paths)} paper(s) verified, no drift detected.")
        return 0

    print(
        f"verify_corpus.py: DRIFT DETECTED in {len(all_issues)} of {len(json_paths)} paper(s):\n",
        file=sys.stderr,
    )
    for issues in all_issues:
        print(f"{issues.paper_id}:", file=sys.stderr)
        for problem in issues.problems:
            print(f"  - {problem}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
