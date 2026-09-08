"""Tests for the paired/cluster bootstrap and win/loss/tie counts.

These pin behaviour on hand-constructed arrays where the answer is obvious --
not statistical power (that would need many more samples than a unit test
should carry), just "does the machinery compute the right thing."
"""

from __future__ import annotations

import numpy as np
import pytest

from ragvlc.eval.bootstrap import bootstrap_mean_ci, cluster_bootstrap, paired_bootstrap, win_loss_tie


def test_bootstrap_mean_ci_point_estimate_is_the_observed_mean():
    a = np.array([1.0, 0.5, 0.0, 0.5])
    result = bootstrap_mean_ci(a, n_resamples=2000, seed=1)
    assert result.point_estimate == pytest.approx(0.5)
    assert result.ci_low <= result.point_estimate <= result.ci_high


def test_bootstrap_mean_ci_is_tight_for_a_constant_array():
    a = np.full(20, 0.7)
    result = bootstrap_mean_ci(a, n_resamples=2000, seed=1)
    assert result.ci_low == pytest.approx(0.7)
    assert result.ci_high == pytest.approx(0.7)


def test_bootstrap_mean_ci_empty_input():
    result = bootstrap_mean_ci(np.array([]), n_resamples=100, seed=1)
    assert result.point_estimate != result.point_estimate  # nan


def test_paired_bootstrap_point_estimate_is_the_observed_mean_difference():
    a = np.array([1.0, 1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0, 0.0])
    result = paired_bootstrap(a, b, n_resamples=2000, seed=1)
    assert result.point_estimate == pytest.approx(0.5)
    assert result.ci_low <= result.point_estimate <= result.ci_high


def test_paired_bootstrap_ci_excludes_zero_when_a_strictly_dominates():
    a = np.ones(30)
    b = np.zeros(30)
    result = paired_bootstrap(a, b, n_resamples=2000, seed=1)
    assert result.ci_low > 0  # every resample has a beating b -> CI clears 0


def test_paired_bootstrap_ci_straddles_zero_when_configs_are_identical():
    rng = np.random.default_rng(0)
    a = rng.random(50)
    b = a.copy()
    result = paired_bootstrap(a, b, n_resamples=2000, seed=1)
    assert result.point_estimate == pytest.approx(0.0)
    assert result.ci_low <= 0.0 <= result.ci_high


def test_paired_bootstrap_is_reproducible_given_the_same_seed():
    a = np.array([1.0, 0.5, 0.0, 1.0, 0.2])
    b = np.array([0.5, 0.5, 0.1, 0.0, 0.9])
    r1 = paired_bootstrap(a, b, n_resamples=500, seed=42)
    r2 = paired_bootstrap(a, b, n_resamples=500, seed=42)
    assert r1 == r2


def test_cluster_bootstrap_agrees_with_paired_bootstrap_when_every_cluster_has_one_question():
    """With one question per cluster, resampling clusters *is* resampling
    questions -- the two CIs should land close (not identical: the RNG is
    driven differently), and always agree on the point estimate exactly."""
    rng = np.random.default_rng(3)
    a, b = rng.random(20), rng.random(20)
    clusters = [f"paper{i}" for i in range(20)]  # one question per paper
    paired = paired_bootstrap(a, b, n_resamples=20_000, seed=7)
    clustered = cluster_bootstrap(a, b, clusters, n_resamples=20_000, seed=7)
    assert clustered.point_estimate == pytest.approx(paired.point_estimate)
    assert clustered.ci_low == pytest.approx(paired.ci_low, abs=0.1)
    assert clustered.ci_high == pytest.approx(paired.ci_high, abs=0.1)


def test_cluster_bootstrap_widens_ci_when_questions_cluster_within_papers():
    """Same per-question values, but half come from one paper: the cluster
    bootstrap must not be narrower than the naive paired one."""
    a = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    clusters = ["A", "A", "A", "A", "B", "B", "B", "B"]  # two papers, 4 questions each
    paired = paired_bootstrap(a, b, n_resamples=4000, seed=5)
    clustered = cluster_bootstrap(a, b, clusters, n_resamples=4000, seed=5)
    assert (clustered.ci_high - clustered.ci_low) >= (paired.ci_high - paired.ci_low)


def test_win_loss_tie_counts():
    a = np.array([1.0, 0.5, 0.0, 0.3, 0.3])
    b = np.array([0.5, 0.5, 1.0, 0.3, 0.1])
    # index 0: 1.0 > 0.5 win | 1: tie | 2: 0.0 < 1.0 loss | 3: tie | 4: 0.3 > 0.1 win
    result = win_loss_tie(a, b)
    assert (result.wins, result.losses, result.ties) == (2, 1, 2)
