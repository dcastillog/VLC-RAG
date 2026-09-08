"""Statistics for comparing two configurations on the same per-question metric.

"Config A scored 0.62, config B scored 0.58" is not a result -- it is two
numbers. Every comparison here works on the *paired difference*
(``a[i] - b[i]`` for the same question ``i`` under both configs) and reports a
point estimate plus an uncertainty interval, never a bare pair of averages.

Two resampling schemes:

* **Paired bootstrap** -- resample questions independently, with replacement.
  Standard nonparametric CI on the mean paired difference.
* **Cluster bootstrap** -- resample *papers* with replacement, taking every
  question from each drawn paper. Questions cluster within papers (several
  questions share one source paper's phrasing, terminology, and section
  structure), so treating questions as independent draws understates
  uncertainty; this is the robustness check for that.

Both use the same fixed seed (`eval.random_seed`) for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    point_estimate: float  # observed mean(a) - mean(b) on the real data (or plain mean(a), see bootstrap_mean_ci)
    ci_low: float          # 2.5th percentile of the resampled statistic
    ci_high: float         # 97.5th percentile
    n_resamples: int


def bootstrap_mean_ci(a: np.ndarray, *, n_resamples: int = 10_000, seed: int = 42) -> BootstrapResult:
    """CI on mean(a) alone, resampling question indices with replacement.

    For reporting one configuration's metric (e.g. MRR@10 at one RRF ``k``)
    against an uncertainty band, rather than a difference between two configs.
    """
    a = np.asarray(a, dtype=float)
    if a.ndim != 1:
        raise ValueError("a must be a 1-D array (one value per question)")
    n = len(a)
    point_estimate = float(np.mean(a)) if n else float("nan")
    if n == 0:
        return BootstrapResult(point_estimate, float("nan"), float("nan"), n_resamples)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = a[idx].mean(axis=1)
    ci_low, ci_high = np.percentile(means, [2.5, 97.5])
    return BootstrapResult(point_estimate, float(ci_low), float(ci_high), n_resamples)


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, *, n_resamples: int = 10_000, seed: int = 42
) -> BootstrapResult:
    """CI on mean(a) - mean(b), resampling question indices with replacement."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be 1-D arrays of the same length (one value per question)")
    n = len(a)
    rng = np.random.default_rng(seed)

    point_estimate = float(np.mean(a) - np.mean(b))
    if n == 0:
        return BootstrapResult(point_estimate, float("nan"), float("nan"), n_resamples)

    idx = rng.integers(0, n, size=(n_resamples, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return BootstrapResult(point_estimate, float(ci_low), float(ci_high), n_resamples)


def cluster_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    cluster_ids: list[str],
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> BootstrapResult:
    """CI on mean(a) - mean(b), resampling *clusters* (source papers) with
    replacement -- each resample takes every question belonging to each drawn
    paper, so a paper drawn twice contributes its questions twice."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(cluster_ids) != len(a):
        raise ValueError("cluster_ids must have one entry per question")

    point_estimate = float(np.mean(a) - np.mean(b))

    clusters: dict[str, list[int]] = {}
    for i, cid in enumerate(cluster_ids):
        clusters.setdefault(cid, []).append(i)
    unique_clusters = list(clusters.values())
    n_clusters = len(unique_clusters)
    if n_clusters == 0:
        return BootstrapResult(point_estimate, float("nan"), float("nan"), n_resamples)

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples)
    for r in range(n_resamples):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        idx = [i for c in draw for i in unique_clusters[c]]
        diffs[r] = a[idx].mean() - b[idx].mean()

    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return BootstrapResult(point_estimate, float(ci_low), float(ci_high), n_resamples)


@dataclass(frozen=True)
class WinLossTie:
    wins: int   # a[i] > b[i]
    losses: int  # a[i] < b[i]
    ties: int    # a[i] == b[i]


def win_loss_tie(a: np.ndarray, b: np.ndarray) -> WinLossTie:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return WinLossTie(
        wins=int(np.sum(a > b)),
        losses=int(np.sum(a < b)),
        ties=int(np.sum(a == b)),
    )
