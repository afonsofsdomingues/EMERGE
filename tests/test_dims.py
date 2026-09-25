"""App. L: behaviour correlation structure and effective dimensionality."""

import math

import numpy as np
import pandas as pd
import pytest

from emerge.analyze import dims
from emerge.analyze import leaderboard as lb


def tidy(datasets):
    """{dataset: {behaviour: [values per model]}} -> cell_matrix-shaped frame.

    ``delta_bar`` is the row mean over the behaviour columns, matching the
    cell_matrix convention for complete cells.
    """
    rows = []
    for d, by_b in datasets.items():
        n = len(next(iter(by_b.values())))
        for i in range(n):
            vals = {f"delta::{b}": v[i] for b, v in by_b.items()}
            finite = [v for v in vals.values() if v == v]
            rows.append({"model": f"m{i}", "dataset": d,
                         "delta_bar": sum(finite) / len(finite), **vals})
    return pd.DataFrame(rows)


class TestSpectrum:
    def test_identity_correlation_gives_full_dimensionality(self):
        spec = dims.spectrum(np.eye(4))
        assert spec["eff_dim"] == pytest.approx(4.0)
        assert spec["pc1_share"] == pytest.approx(0.25)

    def test_lockstep_correlation_gives_dimensionality_one(self):
        spec = dims.spectrum(np.ones((4, 4)))
        assert spec["eff_dim"] == pytest.approx(1.0)
        assert spec["pc1_share"] == pytest.approx(1.0)

    def test_negative_eigenvalues_are_clipped_and_reported(self):
        # A non-PSD "correlation" of the kind pairwise estimation can produce.
        corr = np.array([[1.0, 0.9, -0.9],
                         [0.9, 1.0, 0.9],
                         [-0.9, 0.9, 1.0]])
        spec = dims.spectrum(corr)
        assert spec["clipped_mass"] > 0
        assert np.all(spec["eigenvalues"] >= 0)

    def test_undefined_entries_raise(self):
        corr = np.eye(3)
        corr[0, 1] = corr[1, 0] = np.nan
        with pytest.raises(ValueError, match="undefined"):
            dims.spectrum(corr)

    def test_loading_signs_are_deterministic(self):
        spec = dims.spectrum(np.ones((3, 3)))
        peak = np.argmax(np.abs(spec["loadings"][:, 0]))
        assert spec["loadings"][peak, 0] > 0


class TestPairwiseCorrelation:
    def test_correlations_and_counts(self):
        x = np.array([1.0, -1.0, 2.0, -2.0])
        vals = np.column_stack([x, 2 * x, -x])
        corr, n = dims.pairwise_correlation(vals)
        assert corr[0, 1] == pytest.approx(1.0)
        assert corr[0, 2] == pytest.approx(-1.0)
        assert (n == 4).all()

    def test_nan_cells_shrink_the_pair_count(self):
        x = np.array([1.0, -1.0, 2.0, -2.0, 0.5])
        vals = np.column_stack([x, x])
        vals[0, 1] = np.nan
        corr, n = dims.pairwise_correlation(vals)
        assert n[0, 1] == 4
        assert n[0, 0] == 5
        assert corr[0, 1] == pytest.approx(1.0)

    def test_too_little_overlap_stays_nan(self):
        vals = np.column_stack([[1.0, 2.0, np.nan, np.nan],
                                [np.nan, np.nan, 1.0, 2.0]])
        corr, n = dims.pairwise_correlation(vals)
        assert n[0, 1] == 0
        assert math.isnan(corr[0, 1])


# Within-dataset patterns: b1 and b2 exactly uncorrelated across six models.
B1 = [1.0, -1.0, 1.0, -1.0, 0.0, 0.0]
B2 = [1.0, 1.0, -1.0, -1.0, 0.0, 0.0]


class TestPooledDims:
    def test_lockstep_behaviours_give_dimensionality_one(self):
        matrix = tidy({"d1": {"b1": B1, "b2": [2 * v for v in B1]}})
        pooled = dims.pooled_dims(matrix)
        assert pooled["eff_dim"] == pytest.approx(1.0)

    def test_independent_behaviours_give_dimensionality_two(self):
        matrix = tidy({"d1": {"b1": B1, "b2": B2}})
        pooled = dims.pooled_dims(matrix)
        assert pooled["eff_dim"] == pytest.approx(2.0)

    def test_dataset_centring_removes_between_dataset_shifts(self):
        # d2 repeats d1's within-dataset patterns shifted by +10 on both
        # behaviours.  Pooled without centring, the shared shift would read
        # as near-perfect correlation; centred, the true r = 0 survives.
        matrix = tidy({
            "d1": {"b1": B1, "b2": B2},
            "d2": {"b1": [v + 10 for v in B1], "b2": [v + 10 for v in B2]},
        })
        pooled = dims.pooled_dims(matrix)
        i, j = pooled["behaviours"].index("b1"), pooled["behaviours"].index("b2")
        assert pooled["corr"][i, j] == pytest.approx(0.0, abs=1e-12)
        assert pooled["eff_dim"] == pytest.approx(2.0)

    def test_all_nan_behaviour_is_dropped_and_recorded(self):
        matrix = tidy({"d1": {"b1": B1, "b2": B2}})
        matrix["delta::b3"] = float("nan")
        pooled = dims.pooled_dims(matrix)
        assert pooled["dropped"] == ["b3"]
        assert pooled["behaviours"] == ["b1", "b2"]

    def test_complete_cells_cross_check(self):
        matrix = tidy({"d1": {"b1": B1, "b2": B2}})
        matrix.loc[0, "delta::b2"] = float("nan")
        pooled = dims.pooled_dims(matrix)
        assert pooled["n_complete"] == 5
        assert np.isfinite(pooled["eff_dim_complete"])

    def test_zero_variance_behaviour_leaves_the_spectrum_undefined(self):
        # b3 is constant, so its correlations are undefined.  The spectrum
        # is all-or-nothing: it goes undefined rather than silently shrinking
        # to the informative behaviours, while the correlation and per-pair n
        # stay reportable so the cause is visible.
        matrix = tidy({"d1": {"b1": B1, "b2": B2, "b3": [0.5] * 6}})
        pooled = dims.pooled_dims(matrix)
        assert not pooled["defined"]
        assert math.isnan(pooled["eff_dim"])
        i, j, k = (pooled["behaviours"].index(b) for b in ("b1", "b2", "b3"))
        assert pooled["corr"][i, j] == pytest.approx(0.0, abs=1e-12)
        assert math.isnan(pooled["corr"][k, k])
        assert (pooled["pair_n"] == 6).all()
        assert dims.spectrum_table(pooled).empty
        # every null draw keeps the constant column constant -> all undefined
        null = dims.shuffle_null(matrix, n_shuffles=10, seed=0)
        assert null["n_valid"] == 0

    def test_too_few_cells_degrades_gracefully(self):
        # A single cell cannot define any pairwise correlation: the spectrum
        # fields go NaN with defined=False, but corr/pair_n stay reportable.
        matrix = tidy({"d1": {"b1": [1.0], "b2": [2.0]}})
        pooled = dims.pooled_dims(matrix)
        assert not pooled["defined"]
        assert math.isnan(pooled["eff_dim"])
        assert (pooled["pair_n"] == 1).all()
        assert dims.spectrum_table(pooled).empty

    def test_rho_with_delta_bar_tracks_a_dominant_behaviour(self):
        # b1 varies strongly and monotonically with delta_bar; b2 is flat noise.
        b1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        b2 = [0.1, -0.1, 0.1, -0.1, 0.1, -0.1]
        pooled = dims.pooled_dims(tidy({"d1": {"b1": b1, "b2": b2}}))
        assert pooled["rho_delta_bar"]["b1"] == pytest.approx(1.0)
        assert abs(pooled["rho_delta_bar"]["b2"]) < 0.5


class TestShuffleNull:
    def matrix(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=24)
        # Eight behaviours in near-lockstep: observed eff dim ~ 1.
        return tidy({"d1": {f"b{k}": list(x + 0.01 * rng.normal(size=24))
                            for k in range(8)}})

    def test_null_exceeds_observed_for_correlated_data(self):
        matrix = self.matrix()
        observed = dims.pooled_dims(matrix)["eff_dim"]
        null = dims.shuffle_null(matrix, n_shuffles=50, seed=0)
        assert observed == pytest.approx(1.0, abs=0.05)
        assert null["n_valid"] == 50
        assert null["mean"] > 4.0 > observed

    def test_seeded_reproducibility(self):
        matrix = self.matrix()
        a = dims.shuffle_null(matrix, n_shuffles=20, seed=3)
        b = dims.shuffle_null(matrix, n_shuffles=20, seed=3)
        assert a["mean"] == b["mean"]
        assert np.array_equal(a["draws"], b["draws"])


class TestPerDatasetDims:
    def test_holdout_dataset_is_excluded_with_reason(self):
        matrix = tidy({
            "d1": {"b1": B1 * 2, "b2": B2 * 2},          # 12 complete cells
            "d2": {"b1": B1 * 2, "b2": [float("nan")] * 12},  # behaviour holdout
        })
        table = dims.per_dataset_dims(matrix, min_cells=10).set_index("dataset")
        assert bool(table.loc["d1", "included"])
        assert table.loc["d1", "eff_dim"] == pytest.approx(2.0)
        assert not bool(table.loc["d2", "included"])
        assert "b2 undefined for every cell" in table.loc["d2", "excluded_reason"]
        # rho stays reported for the behaviours the dataset does measure
        assert np.isfinite(table.loc["d2", "rho::b1"])

    def test_min_cells_rule(self):
        matrix = tidy({"d1": {"b1": B1, "b2": B2}})  # 6 complete cells
        table = dims.per_dataset_dims(matrix, min_cells=10)
        assert not bool(table["included"].iloc[0])
        assert "n_complete < 10" in table["excluded_reason"].iloc[0]
        assert bool(dims.per_dataset_dims(matrix, min_cells=5)["included"].iloc[0])

    def test_rho_by_hand(self):
        b1 = [1.0, 2.0, 3.0, 4.0]
        b2 = [0.0, 0.0, 1.0, -1.0]
        table = dims.per_dataset_dims(tidy({"d1": {"b1": b1, "b2": b2}}),
                                      min_cells=3).set_index("dataset")
        assert table.loc["d1", "rho::b1"] == pytest.approx(
            dims.spearman_r(b1, [(a + b) / 2 for a, b in zip(b1, b2)]))


def rows(behaviour, domain, n_items, *, aligned, prefix=""):
    return [{"question_id": f"{prefix}{behaviour}-{domain}-{q}", "sample_index": 0,
             "category": f"{behaviour}:src", "sub_category": domain,
             "source_file": behaviour, "aligned": aligned,
             "aligned__refusal": False, "coherent": 90}
            for q in range(n_items)]


class TestCellMatrix:
    def test_deltas_floor_holdout_and_the_leaderboard_pin(self):
        holdout = lb.Holdout(dataset="d", domains=frozenset(),
                             behaviour="hallucination")
        base = (rows("deception", "other", 40, aligned=90)
                + rows("hallucination", "other", 40, aligned=90)
                + rows("sycophancy", "other", 10, aligned=90))
        mis = (rows("deception", "other", 40, aligned=10)
               + rows("hallucination", "other", 40, aligned=10)
               + rows("sycophancy", "other", 10, aligned=10))
        cells = [{"model": "m", "dataset": "d1", "seed": 0, "ft_rows": mis,
                  "base_rows": base, "holdout": holdout}]
        matrix = dims.cell_matrix(cells, floor=30)
        row = matrix.iloc[0]
        # deception: full shift; hallucination: in-domain control -> NaN;
        # sycophancy: 10 items < floor 30 -> NaN.
        assert row["delta::deception"] == pytest.approx(1.0)
        assert math.isnan(row["delta::hallucination"])
        assert math.isnan(row["delta::sycophancy"])
        # the row mean over included behaviours reproduces delta_bar
        assert row["delta_bar"] == pytest.approx(row["delta::deception"])
        expected = lb.assemble([
            {"model": "m", "dataset": "d1", "seed": 0,
             "profile": lb.cell_profile(mis, base, holdout, floor=30)}
        ])["leaderboard"]
        assert row["delta_bar"] == pytest.approx(expected["delta_bar"].iloc[0])

    def test_seeds_average_first(self):
        holdout = lb.Holdout(dataset="d", domains=frozenset(), behaviour=None)
        base = rows("deception", "other", 40, aligned=90)
        mis = rows("deception", "other", 40, aligned=10)
        cells = [
            {"model": "m", "dataset": "d1", "seed": 0, "ft_rows": mis,
             "base_rows": base, "holdout": holdout},
            {"model": "m", "dataset": "d1", "seed": 1, "ft_rows": base,
             "base_rows": base, "holdout": holdout},
        ]
        matrix = dims.cell_matrix(cells, floor=30)
        assert len(matrix) == 1
        assert matrix.iloc[0]["delta::deception"] == pytest.approx(0.5)
        assert matrix.iloc[0]["delta_bar"] == pytest.approx(0.5)

    def test_raw_diff_columns_mirror_the_profile(self):
        holdout = lb.Holdout(dataset="d", domains=frozenset(), behaviour=None)
        base = rows("deception", "other", 40, aligned=90)
        mis = rows("deception", "other", 40, aligned=10)
        cells = [{"model": "m", "dataset": "d1", "seed": 0, "ft_rows": mis,
                  "base_rows": base, "holdout": holdout}]
        matrix = dims.cell_matrix(cells, floor=30)
        profile = lb.cell_profile(mis, base, holdout, floor=30)
        assert matrix.iloc[0]["raw::deception"] == pytest.approx(
            profile["behaviours"]["deception"]["raw_diff"])
