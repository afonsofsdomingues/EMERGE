"""Estimators.  Each test states the property the benchmark relies on."""

from __future__ import annotations

import numpy as np
import pytest

from emerge.analyze import stats


class TestProportions:
    def test_wilson_stays_in_range_near_zero(self):
        lo, hi = stats.wilson_ci(0, 100)
        assert lo == 0.0 and 0 < hi < 0.05

    def test_wilson_handles_empty(self):
        assert stats.wilson_ci(0, 0) == (0.0, 1.0)

    def test_clustered_interval_is_wider_than_naive(self):
        """Samples drawn from a handful of questions are not independent;
        treating them as independent shrinks intervals by roughly the square
        root of the samples-per-question."""
        rng = np.random.default_rng(0)
        clusters = np.repeat(np.arange(8), 50)
        rates = rng.random(8) * 0.5
        indicators = (rng.random(400) < rates[clusters]).astype(float)
        clustered = stats.cluster_bootstrap_rate(indicators, clusters, seed=0)
        naive = stats.wilson_ci(int(indicators.sum()), len(indicators))
        assert (clustered["ci_high"] - clustered["ci_low"]) > (naive[1] - naive[0])
        assert clustered["design_effect"] > 1.5
        assert clustered["n_clusters"] == 8

    def test_clustered_matches_naive_without_clustering(self):
        rng = np.random.default_rng(1)
        indicators = (rng.random(400) < 0.3).astype(float)
        clustered = stats.cluster_bootstrap_rate(indicators, np.arange(400), seed=0)
        assert clustered["design_effect"] == pytest.approx(1.0, abs=0.4)


class TestComparisons:
    def test_difference_interval_excludes_zero_for_a_real_effect(self):
        rng = np.random.default_rng(0)
        result = stats.diff_of_means_ci(rng.normal(0.5, 0.1, 80),
                                        rng.normal(0.2, 0.1, 80), seed=0)
        assert result["ci_low"] > 0

    def test_difference_interval_covers_zero_under_the_null(self):
        rng = np.random.default_rng(0)
        result = stats.diff_of_means_ci(rng.normal(0.3, 0.1, 200),
                                        rng.normal(0.3, 0.1, 200), seed=0)
        assert result["ci_low"] < 0 < result["ci_high"]

    def test_permutation_p_is_calibrated(self):
        rng = np.random.default_rng(0)
        assert stats.paired_permutation_p(rng.normal(0, 1, 40), n_perm=2000, seed=0) > 0.05
        assert stats.paired_permutation_p(rng.normal(2, 0.5, 40), n_perm=2000, seed=0) < 0.01

    def test_effect_sizes_agree_in_direction(self):
        rng = np.random.default_rng(0)
        a, b = rng.normal(1, 1, 100), rng.normal(0, 1, 100)
        assert stats.cohens_d(a, b) > 0 and stats.cliffs_delta(a, b) > 0

    def test_cliffs_delta_bounds(self):
        assert stats.cliffs_delta([5, 6, 7], [1, 2, 3]) == pytest.approx(1.0)
        assert stats.cliffs_delta([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)


class TestCorrelationAndMultiplicity:
    def test_spearman_is_monotone_invariant(self):
        x = np.arange(30)
        assert stats.spearman_r(x, x ** 3) == pytest.approx(1.0, abs=1e-9)
        assert stats.spearman_r(x, -x) == pytest.approx(-1.0, abs=1e-9)

    def test_spearman_handles_ties(self):
        assert abs(stats.spearman_r([1, 1, 2, 2, 3], [1, 1, 2, 2, 3]) - 1.0) < 1e-9

    def test_slope_recovers_the_truth(self):
        rng = np.random.default_rng(0)
        x = np.arange(80, dtype=float)
        result = stats.bootstrap_slope_ci(x, 3 * x + rng.normal(0, 1, 80), seed=0)
        assert result["ci_low"] < 3 < result["ci_high"]

    def test_holm_is_stricter_than_uncorrected(self):
        """A dozen mechanism tests on one bank at an uncorrected 0.05 is not 0.05."""
        pvalues = {f"t{i}": 0.03 for i in range(12)}
        assert not any(v["reject"] for v in stats.holm_bonferroni(pvalues).values())
        assert stats.holm_bonferroni({"a": 0.0001, "b": 0.9})["a"]["reject"]


class TestLearningCurves:
    def test_fraction_of_asymptote(self):
        fractions = stats.fraction_of_asymptote([2.0, 1.5, 1.1, 1.0])
        assert fractions[0] == pytest.approx(0.0)
        assert fractions[-1] == pytest.approx(1.0)
        assert fractions[1] == pytest.approx(0.5)

    def test_flat_curve_yields_zero_gain(self):
        assert np.allclose(stats.fraction_of_asymptote([1.0, 1.0, 1.0]), 0.0)
