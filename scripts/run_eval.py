"""CLI: run the full retrieval evaluation and write `results/`.

    uv run python scripts/run_eval.py

Runs all 8 configurations (2 chunkers x 4 modes) over every answerable
question, computes success@k / recall@k / MRR@10 (primary), the paired and
cluster bootstrap comparisons, and the IDF-overlap regression, then writes:

    results/metrics.md        -- tables, comparisons, regression summary
    results/regression.png    -- overlap score vs (sparse RR - dense RR)
    results/success_at_k.png  -- success@k curves, one line per configuration
    results/raw_results.json  -- every per-question value, plus the config
                                  that produced it (models, chunker params,
                                  fusion settings, seed, git commit hash)

Needs the Qdrant collections ingested and the eval set judged (`has_answer`
reconciled via `scripts/eval_set.py reconcile`).
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this runs from a CLI, never a notebook/display
import matplotlib.pyplot as plt
import numpy as np

from ragvlc.config import ExperimentConfig, get_experiment, get_paths, get_settings
from ragvlc.eval.bootstrap import BootstrapResult, cluster_bootstrap, paired_bootstrap, win_loss_tie
from ragvlc.eval.budget import cumulative_tokens_to_first_relevant, success_at_budget_curve, token_lookup
from ragvlc.eval.evaluation import ConfigResults, evaluate_all
from ragvlc.eval.idf import Bm25Tokenizer, compute_idf, overlap_score
from ragvlc.eval.pool import CONFIGS
from ragvlc.eval.store import read_jsonl, read_questions
from ragvlc.retrieval import Searcher

# success@budget x-axis: cumulative tokens retrieved. Covers up to a top-10
# `fixed` run (~10 x 400 = 4000 tokens) with headroom.
BUDGET_TOKENS = [0, 200, 400, 800, 1200, 1600, 2000, 2500, 3000, 3500, 4000, 5000]

# The chunker/mode pairs compared in the results table: every mode across the
# two chunkers (the central question this eval exists to answer), plus dense
# vs sparse vs hybrid_rrf within each chunker (the fusion story).
COMPARISONS: list[tuple[tuple[str, str], tuple[str, str]]] = [
    (("section_aware", mode), ("fixed", mode)) for mode in ("dense", "sparse", "hybrid_rrf", "hybrid_dbsf")
] + [
    (("fixed", "hybrid_rrf"), ("fixed", "dense")),
    (("fixed", "hybrid_rrf"), ("fixed", "sparse")),
    (("section_aware", "hybrid_rrf"), ("section_aware", "dense")),
    (("section_aware", "hybrid_rrf"), ("section_aware", "sparse")),
]


def _git_commit_hash(root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5, check=False
        )
        return proc.stdout.strip() if proc.returncode == 0 else "unknown (git failed)"
    except OSError:
        return "unknown (git not available)"


def _run_metadata(experiment: ExperimentConfig, paths: Path, n_answerable: int, n_negative: int) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_hash(paths),
        "n_answerable_questions": n_answerable,
        "n_negative_questions_excluded": n_negative,
        "dense_model": experiment.retrieval.dense_model,
        "sparse_model": experiment.retrieval.sparse_model,
        "query_prefix": experiment.retrieval.query_prefix,
        "chunking": {
            "max_tokens": experiment.chunking.max_tokens,
            "fixed_overlap_tokens": experiment.chunking.fixed.overlap_tokens,
            "section_aware_min_chunk_tokens": experiment.chunking.section_aware.min_chunk_tokens,
            "section_aware_min_indexed_tokens": experiment.chunking.section_aware.min_indexed_tokens,
        },
        "fusion": {"prefetch_limit": experiment.retrieval.prefetch_limit, "rrf_k": experiment.retrieval.rrf_k},
        "eval": {
            "relevance_threshold": experiment.eval.relevance_threshold,
            "success_at_k": experiment.eval.success_at_k,
            "mrr_k": experiment.eval.mrr_k,
            "random_seed": experiment.eval.random_seed,
            "bootstrap_resamples": experiment.eval.bootstrap_resamples,
            "idf_corpus_chunker": experiment.eval.idf_corpus_chunker,
        },
    }


# --------------------------------------------------------------------------- #
# IDF-overlap regression
# --------------------------------------------------------------------------- #
def _bootstrap_slope(x: np.ndarray, y: np.ndarray, *, n_resamples: int, seed: int) -> tuple[float, float]:
    """(ci_low, ci_high) for the OLS slope of y on x, resampling questions."""
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.integers(0, n, size=(n_resamples, n))
    xs, ys = x[idx], y[idx]  # (n_resamples, n)
    x_centered = xs - xs.mean(axis=1, keepdims=True)
    y_centered = ys - ys.mean(axis=1, keepdims=True)
    slopes = (x_centered * y_centered).sum(axis=1) / (x_centered**2).sum(axis=1)
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    return float(lo), float(hi)


def compute_idf_regression(
    searcher: Searcher,
    questions: list[dict],
    results: dict[tuple[str, str], ConfigResults],
    paths,
    eval_cfg,
) -> dict:
    del searcher  # IDF corpus comes from data/chunks/*.jsonl, not a live search
    chunker = eval_cfg.idf_corpus_chunker
    chunk_records = read_jsonl(paths.chunks_dir / f"{chunker}.jsonl")
    if not chunk_records:
        raise FileNotFoundError(f"no chunks at data/chunks/{chunker}.jsonl -- run scripts/build_chunks.py")

    tokenizer = Bm25Tokenizer()
    corpus_tokens = tokenizer.token_sets([r["text"] for r in chunk_records])
    idf_index = compute_idf(corpus_tokens)

    qids = [q["qid"] for q in questions]
    question_tokens = tokenizer.token_sets([q["question"] for q in questions])
    answer_texts = [" ".join(s["text"] for s in q["gold_spans"]) for q in questions]
    answer_tokens = tokenizer.token_sets(answer_texts)

    overlap = np.array(
        [overlap_score(idf_index, qt, at) for qt, at in zip(question_tokens, answer_tokens)]
    )

    dense_rr = np.mean(
        [results[("fixed", "dense")].rr_array(qids), results[("section_aware", "dense")].rr_array(qids)], axis=0
    )
    sparse_rr = np.mean(
        [results[("fixed", "sparse")].rr_array(qids), results[("section_aware", "sparse")].rr_array(qids)], axis=0
    )
    y = sparse_rr - dense_rr

    slope, intercept = np.polyfit(overlap, y, 1)
    ci_low, ci_high = _bootstrap_slope(
        overlap, y, n_resamples=eval_cfg.bootstrap_resamples, seed=eval_cfg.random_seed
    )

    by_category: dict[str, list[float]] = {}
    for q, score in zip(questions, overlap):
        by_category.setdefault(q.get("category", "unknown"), []).append(float(score))

    return {
        "qids": qids,
        "overlap": overlap,
        "y_sparse_minus_dense_rr": y,
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_ci_low": ci_low,
        "slope_ci_high": ci_high,
        "by_category": {cat: (float(np.mean(v)), float(np.median(v)), len(v)) for cat, v in sorted(by_category.items())},
    }


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _fmt_bs(b: BootstrapResult) -> str:
    return f"{b.point_estimate:+.4f}  [{b.ci_low:+.4f}, {b.ci_high:+.4f}]"


def render_markdown(
    metadata: dict,
    results: dict[tuple[str, str], ConfigResults],
    qids: list[str],
    paper_ids: dict[str, str],
    success_ks: list[int],
    regression: dict,
    negative_qids: list[str],
    mean_top10_tokens: dict[str, float],
) -> str:
    lines: list[str] = ["# Retrieval evaluation results", ""]

    lines += ["## Config", "", "```json", json.dumps(metadata, indent=2), "```", ""]

    lines += ["## Per-configuration metrics", ""]
    header = ["chunker", "mode", f"MRR@{metadata['eval']['mrr_k']}"] + [f"success@{k}" for k in success_ks] + [f"recall@{k}" for k in success_ks]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for chunker, mode in CONFIGS:
        r = results[(chunker, mode)]
        row = [chunker, mode, f"{r.mean_rr():.4f}"]
        row += [f"{r.mean_success(k):.4f}" for k in success_ks]
        row += [f"{r.mean_recall(k):.4f}" for k in success_ks]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    ratio = mean_top10_tokens["fixed"] / mean_top10_tokens["section_aware"] if mean_top10_tokens.get("section_aware") else float("nan")
    lines += [
        "## success@budget: the matched comparison",
        "",
        "success@k is not an apples-to-apples comparison between chunkers: at the same k, `fixed` "
        "hands the reader far more text than `section_aware` does, so part of any success@k gap is "
        "just retrieved-text volume, not chunking quality. success@budget puts both on the same "
        "x-axis -- cumulative tokens read down to the first relevant hit -- instead of rank.",
        "",
        f"- mean tokens in a top-10 result list: fixed {mean_top10_tokens.get('fixed', 0):.0f}, "
        f"section_aware {mean_top10_tokens.get('section_aware', 0):.0f} "
        f"({ratio:.2f}x)",
        "- plot: `success_at_budget.png`",
        "",
    ]

    lines += [
        "## Comparisons (MRR, paired difference A - B)",
        "",
        "Point estimate with the [2.5, 97.5] percentile interval from "
        f"{metadata['eval']['bootstrap_resamples']:,} resamples, paired over questions and "
        "again clustered over source papers. Win/loss/tie is per-question MRR.",
        "",
        "| A | B | paired bootstrap | cluster bootstrap | win/loss/tie |",
        "|---|---|---|---|---|",
    ]
    for (chunker_a, mode_a), (chunker_b, mode_b) in COMPARISONS:
        a = results[(chunker_a, mode_a)].rr_array(qids)
        b = results[(chunker_b, mode_b)].rr_array(qids)
        clusters = [paper_ids[q] for q in qids]
        paired = paired_bootstrap(a, b, n_resamples=metadata["eval"]["bootstrap_resamples"], seed=metadata["eval"]["random_seed"])
        clustered = cluster_bootstrap(a, b, clusters, n_resamples=metadata["eval"]["bootstrap_resamples"], seed=metadata["eval"]["random_seed"])
        wlt = win_loss_tie(a, b)
        label_a, label_b = f"{chunker_a}/{mode_a}", f"{chunker_b}/{mode_b}"
        lines.append(
            f"| {label_a} | {label_b} | {_fmt_bs(paired)} | {_fmt_bs(clustered)} | "
            f"{wlt.wins}/{wlt.losses}/{wlt.ties} |"
        )
    lines.append("")

    lines += [
        "## IDF-overlap regression",
        "",
        "For each question, IDF-weighted lexical overlap between the question text and its "
        "gold span(s) (computed before any retrieval, identical across systems) plotted against "
        "sparse RR - dense RR (each averaged across the two chunkers). Hypothesis: a positive "
        "slope -- questions reusing the papers' rare vocabulary favour sparse retrieval, "
        "paraphrased questions favour dense.",
        "",
        f"- slope = {regression['slope']:+.4f}  "
        f"[{regression['slope_ci_low']:+.4f}, {regression['slope_ci_high']:+.4f}]  "
        f"({metadata['eval']['bootstrap_resamples']:,} resamples)",
        f"- intercept = {regression['intercept']:+.4f}",
        f"- n = {len(regression['qids'])} answerable questions",
        f"- plot: `regression.png`",
        "",
        "### Overlap score by category (leakage check)",
        "",
        "Conceptual questions (`mixed`, `hard`) should sit lower than terminology-anchored ones "
        "(`terminology`). If they do not, the papers' phrasing leaked into how the questions were written.",
        "",
        "| category | mean overlap | median overlap | n |",
        "|---|---|---|---|",
    ]
    for cat, (mean_v, median_v, n) in regression["by_category"].items():
        lines.append(f"| {cat} | {mean_v:.3f} | {median_v:.3f} | {n} |")
    lines.append("")

    lines += [
        "## Negative questions (excluded from retrieval metrics)",
        "",
        f"{len(negative_qids)} question(s) with `has_answer: false`: {', '.join(negative_qids) or '(none)'}. "
        "Judging confirmed the pool held nothing relevant for these.",
        "",
    ]

    return "\n".join(lines)


def _stamp(fig, metadata: dict) -> None:
    """Every output file records the config that produced it -- for the plots
    (which can't hold the full JSON block metrics.md carries) that means at
    least the commit and seed, stamped directly on the image."""
    fig.text(
        0.99, 0.01,
        f"commit {metadata['git_commit'][:8]} · seed {metadata['eval']['random_seed']} · {metadata['generated_at'][:10]}",
        ha="right", va="bottom", fontsize=6, color="gray",
    )


def plot_regression(regression: dict, metadata: dict, out_path: Path) -> None:
    x, y = regression["overlap"], regression["y_sparse_minus_dense_rr"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, alpha=0.6, s=28)
    xs = np.linspace(0, max(1.0, float(x.max()) if len(x) else 1.0), 50)
    ax.plot(xs, regression["slope"] * xs + regression["intercept"], color="black", linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("IDF-weighted question/gold-span overlap")
    ax.set_ylabel("sparse RR - dense RR (avg. over chunkers)")
    ax.set_title(f"slope = {regression['slope']:+.3f}  [{regression['slope_ci_low']:+.3f}, {regression['slope_ci_high']:+.3f}]")
    _stamp(fig, metadata)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_success_at_k(
    results: dict[tuple[str, str], ConfigResults], success_ks: list[int], metadata: dict, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    styles = {"dense": "-", "sparse": "--", "hybrid_rrf": "-.", "hybrid_dbsf": ":"}
    colors = {"fixed": "tab:blue", "section_aware": "tab:orange"}
    for chunker, mode in CONFIGS:
        r = results[(chunker, mode)]
        ax.plot(
            success_ks, [r.mean_success(k) for k in success_ks],
            label=f"{chunker}/{mode}", linestyle=styles[mode], color=colors[chunker], marker="o", markersize=4,
        )
    ax.set_xlabel("k")
    ax.set_ylabel("success@k")
    ax.set_xticks(success_ks)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("success@k by configuration")
    _stamp(fig, metadata)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# success@budget -- computed from the results already in hand plus each
# chunker's data/chunks/*.jsonl (local files; no Qdrant, no re-ingestion).
# --------------------------------------------------------------------------- #
def compute_budget_curves(
    results: dict[tuple[str, str], ConfigResults], qids: list[str], paths
) -> tuple[dict[tuple[str, str], list[float]], dict[str, float]]:
    """(curves, mean_top10_tokens). ``curves[(chunker, mode)]`` is one
    success@budget value per entry in ``BUDGET_TOKENS``. ``mean_top10_tokens``
    is the average total token cost of a top-10 result list, per chunker --
    the number behind "fixed returns Nx more text at the same k"."""
    lookups = {
        chunker: token_lookup(read_jsonl(paths.chunks_dir / f"{chunker}.jsonl"))
        for chunker in {c for c, _ in CONFIGS}
    }

    curves: dict[tuple[str, str], list[float]] = {}
    top10_totals: dict[str, list[float]] = {chunker: [] for chunker in lookups}
    for chunker, mode in CONFIGS:
        r = results[(chunker, mode)]
        first_relevant = [
            cumulative_tokens_to_first_relevant(r.by_qid[qid].hit_spans, r.by_qid[qid].rr, lookups[chunker])
            for qid in qids
        ]
        curves[(chunker, mode)] = success_at_budget_curve(first_relevant, BUDGET_TOKENS)
        top10_totals[chunker].extend(
            sum(lookups[chunker].get(tuple(hit), 0) for hit in r.by_qid[qid].hit_spans) for qid in qids
        )

    mean_top10_tokens = {chunker: float(np.mean(totals)) for chunker, totals in top10_totals.items()}
    return curves, mean_top10_tokens


def plot_success_at_budget(
    curves: dict[tuple[str, str], list[float]], metadata: dict, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    styles = {"dense": "-", "sparse": "--", "hybrid_rrf": "-.", "hybrid_dbsf": ":"}
    colors = {"fixed": "tab:blue", "section_aware": "tab:orange"}
    for chunker, mode in CONFIGS:
        ax.plot(
            BUDGET_TOKENS, curves[(chunker, mode)],
            label=f"{chunker}/{mode}", linestyle=styles[mode], color=colors[chunker], marker="o", markersize=4,
        )
    ax.set_xlabel("cumulative tokens retrieved (down to the first relevant hit)")
    ax.set_ylabel("success@budget")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("success@budget: the matched comparison (tokens, not rank)")
    _stamp(fig, metadata)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    _ = argv
    experiment = get_experiment()
    paths = get_paths()
    settings = get_settings()

    all_questions = read_questions(paths.questions_jsonl)
    answerable = [q for q in all_questions if q.get("has_answer") is True]
    negative_qids = [q["qid"] for q in all_questions if q.get("has_answer") is False]
    if not answerable:
        print("run_eval.py: no answerable questions (has_answer: true) -- judge first", file=sys.stderr)
        return 1
    missing_spans = [q["qid"] for q in answerable if not q.get("gold_spans")]
    if missing_spans:
        print(f"run_eval.py: has_answer=true with no gold_spans: {missing_spans} -- run eval_set.py reconcile", file=sys.stderr)
        return 1

    from qdrant_client import QdrantClient

    client = QdrantClient(url=settings.qdrant_url)
    try:
        missing_collections = [
            name for name in experiment.retrieval.qdrant.collections.values() if not client.collection_exists(name)
        ]
    except Exception as exc:  # broad on purpose: turn any connection failure into a hint
        print(f"run_eval.py: cannot reach Qdrant at {settings.qdrant_url}: {exc}", file=sys.stderr)
        return 1
    if missing_collections:
        print(f"run_eval.py: collections not ingested: {missing_collections}", file=sys.stderr)
        return 1

    searcher = Searcher.from_config()

    def _progress(done: int, total: int) -> None:
        print(f"  evaluated {done}/{total} configurations", end="\r", flush=True)

    print(f"evaluating {len(CONFIGS)} configurations x {len(answerable)} answerable question(s)...")
    results = evaluate_all(searcher, experiment.retrieval.qdrant.collections, answerable, experiment.eval, on_progress=_progress)
    print()

    print("computing the IDF-overlap regression...")
    regression = compute_idf_regression(searcher, answerable, results, paths, experiment.eval)

    qids = [q["qid"] for q in answerable]
    paper_ids = {q["qid"]: q["paper_id"] for q in answerable}
    metadata = _run_metadata(experiment, paths.root, len(answerable), len(negative_qids))

    print("computing success@budget (from data/chunks/*.jsonl -- no re-ingestion)...")
    budget_curves, mean_top10_tokens = compute_budget_curves(results, qids, paths)

    results_dir = paths.root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    markdown = render_markdown(
        metadata, results, qids, paper_ids, experiment.eval.success_at_k, regression, negative_qids, mean_top10_tokens
    )
    (results_dir / "metrics.md").write_text(markdown, encoding="utf-8")

    plot_regression(regression, metadata, results_dir / "regression.png")
    plot_success_at_k(results, experiment.eval.success_at_k, metadata, results_dir / "success_at_k.png")
    plot_success_at_budget(budget_curves, metadata, results_dir / "success_at_budget.png")

    raw = {
        "config": metadata,
        "results": {
            f"{chunker}/{mode}": {
                qid: {
                    "rr": q.rr,
                    "success": {str(k): v for k, v in q.success.items()},
                    "recall": {str(k): v for k, v in q.recall.items()},
                    "hit_spans": q.hit_spans,
                }
                for qid, q in r.by_qid.items()
            }
            for (chunker, mode), r in results.items()
        },
        "idf_overlap_regression": {
            "slope": regression["slope"],
            "intercept": regression["intercept"],
            "slope_ci": [regression["slope_ci_low"], regression["slope_ci_high"]],
            "by_qid": {
                qid: {"overlap": float(o), "sparse_minus_dense_rr": float(y)}
                for qid, o, y in zip(regression["qids"], regression["overlap"], regression["y_sparse_minus_dense_rr"])
            },
        },
        "success_at_budget": {
            "budgets": BUDGET_TOKENS,
            "mean_top10_tokens": mean_top10_tokens,
            "curves": {f"{chunker}/{mode}": curve for (chunker, mode), curve in budget_curves.items()},
        },
    }
    (results_dir / "raw_results.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")

    print(
        f"wrote {results_dir.relative_to(paths.root)}/"
        "{metrics.md, regression.png, success_at_k.png, success_at_budget.png, raw_results.json}"
    )
    print()
    print(f"MRR@{experiment.eval.mrr_k} by configuration:")
    for chunker, mode in CONFIGS:
        print(f"  {chunker:>13}/{mode:<11} {results[(chunker, mode)].mean_rr():.4f}")
    print()
    print("mean tokens in a top-10 result list (the success@k vs success@budget gap):")
    for chunker, mean_tokens in mean_top10_tokens.items():
        print(f"  {chunker:>13}  {mean_tokens:.0f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
