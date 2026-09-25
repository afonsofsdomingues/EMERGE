"""App. H: leave-one-dataset-out and pairwise per-dataset ranking stability."""

import math

import pandas as pd
import pytest

from emerge.analyze import dataset_stability as ds
from emerge.analyze import leaderboard as lb


def tidy(by_dataset):
    """{dataset: {model: delta_bar}} -> tidy frame."""
    return pd.DataFrame([{"model": m, "dataset": d, "delta_bar": v}
                         for d, by_model in by_dataset.items()
                         for m, v in by_model.items()])


# d1 = d2 rank the models one way; d3 is the outlier that flips the
# canonical ordering (m4 wins on the aggregate only because of d3).
ORDER = {"m1": 4.0, "m2": 3.0, "m3": 2.0, "m4": 1.0}
OUTLIER = {"m1": 0.0, "m2": 0.0, "m3": 0.0, "m4": 10.0}
DELTAS = tidy({"d1": ORDER, "d2": ORDER, "d3": OUTLIER})


class TestLodo:
    def test_outlier_dataset_has_the_lowest_leave_out_rho(self):
        table = ds.lodo(DELTAS)
        by_left_out = {r.left_out: r.spearman_rho for r in table.itertuples()}
        # leaving a redundant dataset keeps the canonical ordering exactly
        assert by_left_out["d1"] == pytest.approx(1.0)
        assert by_left_out["d2"] == pytest.approx(1.0)
        # leaving the outlier restores the d1/d2 ordering, which disagrees
        # with the canonical (outlier-driven) one: rho = -0.2 by hand
        assert by_left_out["d3"] == pytest.approx(-0.2)
        summary = ds.summarise(table, ds.pairwise_rho(DELTAS))
        assert summary["min_lodo_at"] == "d3"
        assert summary["min_lodo_rho"] == pytest.approx(-0.2)

    def test_nan_models_drop_pairwise(self):
        with_nan = tidy({"d1": ORDER,
                         "d2": {**ORDER, "m4": float("nan")}})
        table = ds.lodo(with_nan).set_index("left_out")
        assert table.loc["d1", "n_models"] == 3   # rest = d2, m4 undefined
        assert table.loc["d2", "n_models"] == 4


class TestPairwise:
    def test_pairwise_rhos_and_mean(self):
        pairs = ds.pairwise_rho(DELTAS)
        by_pair = {(r.dataset_a, r.dataset_b): r.spearman_rho
                   for r in pairs.itertuples()}
        assert by_pair[("d1", "d2")] == pytest.approx(1.0)
        # d3 has a three-way tie at 0; against the strict d1 ordering the
        # tie-averaged rank correlation is -3/sqrt(15) by hand
        r_out = -3.0 / math.sqrt(15.0)
        assert by_pair[("d1", "d3")] == pytest.approx(r_out)
        assert by_pair[("d2", "d3")] == pytest.approx(r_out)
        summary = ds.summarise(ds.lodo(DELTAS), pairs)
        assert summary["mean_pairwise_rho"] == pytest.approx((1.0 + 2 * r_out) / 3)
        assert summary["min_pairwise_rho"] == pytest.approx(r_out)
        assert "d3" in summary["min_pair"]

    def test_single_dataset_yields_empty_pairs(self):
        pairs = ds.pairwise_rho(tidy({"d1": ORDER}))
        assert pairs.empty
        summary = ds.summarise(ds.lodo(tidy({"d1": ORDER})), pairs)
        assert math.isnan(summary["mean_pairwise_rho"])


class TestCv:
    def test_restricted_aggregates_and_cv_by_hand(self):
        restricted = ds.lodo_deltas(DELTAS).set_index(["model", "left_out"])
        # m4: d-values (1, 1, 10) -> leave-outs (5.5, 5.5, 1.0)
        assert restricted.loc[("m4", "d1"), "delta_bar"] == pytest.approx(5.5)
        assert restricted.loc[("m4", "d3"), "delta_bar"] == pytest.approx(1.0)
        from emerge.analyze.thresholds import magnitude_cv
        table = magnitude_cv(ds.lodo_deltas(DELTAS)).set_index("model")
        vals = [5.5, 5.5, 1.0]
        mean = sum(vals) / 3
        sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / 2)
        assert table.loc["m4", "delta_bar_mean"] == pytest.approx(mean)
        assert table.loc["m4", "cv"] == pytest.approx(sd / mean)
        assert (table["n_points"] == 3).all()


def rows(behaviour, domain, n_items, *, aligned, prefix=""):
    return [{"question_id": f"{prefix}{behaviour}-{domain}-{q}", "sample_index": 0,
             "category": f"{behaviour}:src", "sub_category": domain,
             "source_file": behaviour, "aligned": aligned,
             "aligned__refusal": False, "coherent": 90}
            for q in range(n_items)]


class TestPerDatasetDeltas:
    def test_seed_mean_then_matches_the_leaderboard(self):
        holdout = lb.Holdout(dataset="d", domains=frozenset(), behaviour=None)
        base = rows("deception", "other", 40, aligned=90)
        mis = rows("deception", "other", 40, aligned=10)
        cells = [
            {"model": "m", "dataset": "d1", "seed": 0, "ft_rows": mis,
             "base_rows": base, "holdout": holdout},          # delta 1
            {"model": "m", "dataset": "d1", "seed": 1, "ft_rows": base,
             "base_rows": base, "holdout": holdout},          # delta 0 -> d1 = 0.5
            {"model": "m", "dataset": "d2", "seed": 0, "ft_rows": mis,
             "base_rows": base, "holdout": holdout},          # d2 = 1
        ]
        deltas = ds.per_dataset_deltas(cells, floor=30).set_index("dataset")
        assert deltas.loc["d1", "delta_bar"] == pytest.approx(0.5)
        assert deltas.loc["d2", "delta_bar"] == pytest.approx(1.0)
        expected = lb.assemble([
            {"model": "m", "dataset": c["dataset"], "seed": c["seed"],
             "profile": lb.cell_profile(c["ft_rows"], c["base_rows"],
                                        holdout, floor=30)}
            for c in cells])["leaderboard"]
        assert deltas["delta_bar"].mean() == pytest.approx(
            expected["delta_bar"].iloc[0])
