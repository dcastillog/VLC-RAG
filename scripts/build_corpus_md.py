"""CLI: generate CORPUS.md, the CC-BY attribution list for the corpus.

    uv run python scripts/build_corpus_md.py

See `ragvlc.corpus` for how each entry is built. Not itself part of the
retrieval pipeline -- CORPUS.md is the attribution CC BY 4.0 requires for the
36 papers, and a reminder that the PDFs are not redistributed here
(`data/pdfs/` is gitignored).
"""

from __future__ import annotations

import csv
import sys

from ragvlc.config import get_paths
from ragvlc.corpus import NA, build_entry, missing_fields, render_corpus_md


def main(argv: list[str] | None = None) -> int:
    _ = argv
    paths = get_paths()

    if not paths.manifest_csv.is_file():
        print(f"build_corpus_md.py: no manifest at {paths.manifest_csv}", file=sys.stderr)
        return 1

    with paths.manifest_csv.open("r", encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    if not manifest_rows:
        print(f"build_corpus_md.py: {paths.manifest_csv} has no rows", file=sys.stderr)
        return 1

    entries = [build_entry(row, paths.crossref) for row in manifest_rows]
    out_path = paths.root / "CORPUS.md"
    out_path.write_text(render_corpus_md(entries), encoding="utf-8")

    n_from_crossref = sum(1 for e in entries if e.authors != NA)
    incomplete = [(e, missing_fields(e)) for e in entries]
    incomplete = [(e, missing) for e, missing in incomplete if missing]

    print(f"build_corpus_md.py: wrote {out_path.relative_to(paths.root)} ({len(entries)} entries)")
    print(f"  {n_from_crossref}/{len(entries)} have Crossref author data")
    if incomplete:
        print(f"  {len(incomplete)} entrie(s) have at least one n/a field:")
        for entry, missing in incomplete:
            print(f"    - {entry.paper_id}: {missing}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
