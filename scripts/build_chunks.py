"""CLI: run both chunkers over the whole normalized corpus and write the
results, so the token-length distributions can be eyeballed before anything is
embedded.

    uv run python scripts/build_chunks.py
    uv run python scripts/build_chunks.py --chunker section_aware

Writes ``data/chunks/{chunker}.jsonl`` (one chunk per line, the fields Phase C
will embed) and prints, per chunker: chunk count, token-length distribution
(min / median / p95 / max), and the chunks-per-paper spread.

The chunkers enforce ``chunk.text == normalized_text[char_start:char_end]``
themselves and raise on any mismatch, so a clean run here is also an offset
check. Run ``scripts/verify_corpus.py`` first: this reads the same normalized
files it guards.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter as Multiset
from dataclasses import dataclass
from pathlib import Path

from ragvlc.chunking import Chunk, Chunker, Paper, TokenCounter, get_chunkers
from ragvlc.config import get_experiment, get_paths


@dataclass
class ChunkerSummary:
    chunker: str
    n_chunks: int
    n_papers: int
    token_min: int
    token_median: float
    token_p95: float
    token_max: int
    per_paper_min: int
    per_paper_median: float
    per_paper_max: int
    over_budget: int  # chunks whose n_tokens exceeds the configured budget
    index_floor: int  # min_indexed_tokens applied (0 == no floor, e.g. fixed)
    dropped_total: int  # chunks below the index floor (not written, not embedded)
    dropped_by_paper: dict[str, int]  # paper_id -> count, only papers with >0

    def render(self, max_tokens: int) -> str:
        lines = [
            f"[{self.chunker}]",
            f"  indexed chunks:  {self.n_chunks} over {self.n_papers} paper(s)",
            f"  tokens/chunk:    min {self.token_min} | median {self.token_median:g} | "
            f"p95 {self.token_p95:g} | max {self.token_max}   (budget {max_tokens})",
            f"  chunks/paper:    min {self.per_paper_min} | median {self.per_paper_median:g} | "
            f"max {self.per_paper_max}",
        ]
        if self.over_budget:
            lines.append(f"  OVER BUDGET:     {self.over_budget} chunk(s) exceed {max_tokens} tokens")

        if self.index_floor <= 0:
            lines.append("  dropped:         n/a (no index floor for this strategy)")
        else:
            lines.append(
                f"  dropped:         {self.dropped_total} chunk(s) below the "
                f"{self.index_floor}-token index floor, not embedded"
            )
            for paper_id, count in sorted(self.dropped_by_paper.items(), key=lambda kv: (-kv[1], kv[0])):
                lines.append(f"    {count:>4}  {paper_id}")
        return "\n".join(lines)


def _percentile(values: list[int], pct: float) -> float:
    """``pct``-th percentile (0-100) by linear interpolation. stdlib only --
    the corpus is ~1k chunks, an approximate method would be silly here."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (pct / 100) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def chunk_corpus(chunker: Chunker, normalized_dir: Path) -> tuple[list[Chunk], list[Chunk]]:
    """``(indexed_chunks, dropped_chunks)`` over every paper, in paper order.

    A chunker that has no notion of dropping (``fixed``) reports no drops --
    the asymmetry is intentional (see the config comment on
    ``min_indexed_tokens``).
    """
    indexed: list[Chunk] = []
    dropped: list[Chunk] = []
    for json_path in sorted(normalized_dir.glob("*.json")):
        paper = Paper.from_normalized(json_path)
        chunk_with_drops = getattr(chunker, "chunk_with_drops", None)
        if chunk_with_drops is not None:
            kept, dropped_here = chunk_with_drops(paper)
        else:
            kept, dropped_here = chunker.chunk(paper), []
        indexed.extend(kept)
        dropped.extend(dropped_here)
    return indexed, dropped


def summarize(
    chunker_name: str,
    chunks: list[Chunk],
    dropped: list[Chunk],
    max_tokens: int,
    index_floor: int,
) -> ChunkerSummary:
    token_counts = [c.n_tokens for c in chunks]
    per_paper = Multiset(c.paper_id for c in chunks)
    per_paper_counts = sorted(per_paper.values())
    dropped_by_paper = dict(Multiset(c.paper_id for c in dropped))
    return ChunkerSummary(
        chunker=chunker_name,
        n_chunks=len(chunks),
        n_papers=len(per_paper),
        token_min=min(token_counts, default=0),
        token_median=statistics.median(token_counts) if token_counts else 0.0,
        token_p95=_percentile(token_counts, 95),
        token_max=max(token_counts, default=0),
        per_paper_min=per_paper_counts[0] if per_paper_counts else 0,
        per_paper_median=statistics.median(per_paper_counts) if per_paper_counts else 0.0,
        per_paper_max=per_paper_counts[-1] if per_paper_counts else 0,
        over_budget=sum(1 for n in token_counts if n > max_tokens),
        index_floor=index_floor,
        dropped_total=len(dropped),
        dropped_by_paper=dropped_by_paper,
    )


def write_chunks(chunks: list[Chunk], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_json_dict(), ensure_ascii=False))
            handle.write("\n")
    tmp_path.replace(out_path)


def main(argv: list[str] | None = None) -> int:
    experiment = get_experiment()
    chunking_config = experiment.chunking
    paths = get_paths()

    all_chunkers = get_chunkers(chunking_config, TokenCounter(chunking_config.tokenizer_model))
    names = [c.name for c in all_chunkers]

    parser = argparse.ArgumentParser(description="Build chunk sets for both strategies.")
    parser.add_argument(
        "--chunker",
        choices=[*names, "all"],
        default="all",
        help="which strategy to build (default: all)",
    )
    args = parser.parse_args(argv)

    if not sorted(paths.normalized.glob("*.json")):
        print(f"build_chunks.py: no papers in {paths.normalized}", file=sys.stderr)
        return 1

    selected = all_chunkers if args.chunker == "all" else [c for c in all_chunkers if c.name == args.chunker]
    chunks_dir = paths.root / "data" / "chunks"

    for chunker in selected:
        chunks, dropped = chunk_corpus(chunker, paths.normalized)
        out_path = chunks_dir / f"{chunker.name}.jsonl"
        write_chunks(chunks, out_path)
        index_floor = int(getattr(chunker, "min_indexed_tokens", 0))
        summary = summarize(chunker.name, chunks, dropped, chunking_config.max_tokens, index_floor)
        print(summary.render(chunking_config.max_tokens))
        print(f"  written:         {out_path.relative_to(paths.root)}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
