"""CLI: Phase G, RRF k sweep -- the one experiment scoped for this run.

    uv run python scripts/sweep_rrf_k.py

Evaluates `hybrid_rrf` at k in {1, 2, 4, 10, 20, 60} on both chunkers and
reports MRR@10 against k, each point with a bootstrap CI. Qdrant defaults to
k=2; the original RRF paper (Cormack et al. 2009) uses 60 -- a larger k
flattens the contribution gap between adjacent ranks, so this asks how much
that choice actually matters on this corpus.

Reuses the retrieval and answerable-question set exactly as scripts/
run_eval.py does; only the RRF k passed to Qdrant varies (via `Searcher.
search`'s per-call `rrf_k` override -- one `Searcher`, no re-ingestion).

Writes results/rrf_k_sweep.png and prints the table. Does not touch the
payload-index or indexing_threshold experiments (out of scope for this run).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ragvlc.config import get_experiment, get_paths, get_settings
from ragvlc.eval.bootstrap import BootstrapResult, bootstrap_mean_ci
from ragvlc.eval.evaluation import evaluate_config
from ragvlc.eval.store import read_questions
from ragvlc.retrieval import Searcher

K_VALUES: list[int] = [1, 2, 4, 10, 20, 60]
CHUNKERS: list[str] = ["fixed", "section_aware"]


def _git_commit_hash(root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5, check=False
        )
        return proc.stdout.strip() if proc.returncode == 0 else "unknown (git failed)"
    except OSError:
        return "unknown (git not available)"


def run_sweep(
    searcher: Searcher, collections: dict[str, str], questions: list[dict], eval_config
) -> dict[tuple[str, int], BootstrapResult]:
    """(chunker, k) -> bootstrap CI on MRR@10 at that k."""
    out: dict[tuple[str, int], BootstrapResult] = {}
    for chunker in CHUNKERS:
        for k in K_VALUES:
            results = evaluate_config(
                searcher, collections[chunker], chunker, "hybrid_rrf", questions, eval_config, rrf_k=k
            )
            rr = np.array([q.rr for q in results.by_qid.values()], dtype=float)
            out[(chunker, k)] = bootstrap_mean_ci(
                rr, n_resamples=eval_config.bootstrap_resamples, seed=eval_config.random_seed
            )
    return out


def plot_sweep(sweep: dict[tuple[str, int], BootstrapResult], stamp: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"fixed": "tab:blue", "section_aware": "tab:orange"}
    for chunker in CHUNKERS:
        points = np.array([sweep[(chunker, k)].point_estimate for k in K_VALUES])
        lo = np.array([sweep[(chunker, k)].ci_low for k in K_VALUES])
        hi = np.array([sweep[(chunker, k)].ci_high for k in K_VALUES])
        ax.plot(K_VALUES, points, marker="o", color=colors[chunker], label=chunker)
        ax.fill_between(K_VALUES, lo, hi, color=colors[chunker], alpha=0.15)
    ax.axvline(2, color="gray", linestyle=":", linewidth=1)
    ax.text(2, ax.get_ylim()[0], " Qdrant default", fontsize=7, color="gray", va="bottom")
    ax.axvline(60, color="gray", linestyle="--", linewidth=1)
    ax.text(60, ax.get_ylim()[0], " RRF paper", fontsize=7, color="gray", va="bottom", ha="right")
    ax.set_xscale("log")
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_xlabel("RRF k")
    ax.set_ylabel("MRR@10 (hybrid_rrf)")
    ax.set_title("RRF k sweep -- shaded band is the bootstrap 95% CI")
    ax.legend()
    fig.text(0.99, 0.01, stamp, ha="right", va="bottom", fontsize=6, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_table(sweep: dict[tuple[str, int], BootstrapResult]) -> str:
    lines = ["| chunker | k | MRR@10 | 95% CI |", "|---|---|---|---|"]
    for chunker in CHUNKERS:
        for k in K_VALUES:
            r = sweep[(chunker, k)]
            marker = " (Qdrant default)" if k == 2 else " (RRF paper)" if k == 60 else ""
            lines.append(f"| {chunker} | {k}{marker} | {r.point_estimate:.4f} | [{r.ci_low:.4f}, {r.ci_high:.4f}] |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    _ = argv
    experiment = get_experiment()
    paths = get_paths()
    settings = get_settings()

    questions = [q for q in read_questions(paths.questions_jsonl) if q.get("has_answer") is True]
    if not questions:
        print("sweep_rrf_k.py: no answerable questions -- judge first", file=sys.stderr)
        return 1

    from qdrant_client import QdrantClient

    client = QdrantClient(url=settings.qdrant_url)
    collections = experiment.retrieval.qdrant.collections
    try:
        missing = [name for name in collections.values() if not client.collection_exists(name)]
    except Exception as exc:  # broad on purpose: turn any connection failure into a hint
        print(f"sweep_rrf_k.py: cannot reach Qdrant at {settings.qdrant_url}: {exc}", file=sys.stderr)
        return 1
    if missing:
        print(f"sweep_rrf_k.py: collections not ingested: {missing}", file=sys.stderr)
        return 1

    searcher = Searcher.from_config()
    print(f"sweeping RRF k in {K_VALUES} x {CHUNKERS} over {len(questions)} answerable question(s)...")
    sweep = run_sweep(searcher, collections, questions, experiment.eval)

    results_dir = paths.root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    commit = _git_commit_hash(paths.root)
    stamp = f"commit {commit[:8]} · seed {experiment.eval.random_seed} · n={len(questions)}"
    plot_sweep(sweep, stamp, results_dir / "rrf_k_sweep.png")

    table = render_table(sweep)
    print()
    print(table)
    print()
    print(f"wrote results/rrf_k_sweep.png  ({stamp})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
