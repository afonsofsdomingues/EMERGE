"""The App.-C joint seed+item bootstrap CIs on Delta_b and Delta_bar."""

import math

import pytest

from emerge.analyze import leaderboard as lb
from emerge.analyze.bootstrap import JointBootstrap


def rows(behaviour, domain, n_items, *, aligned, judges=None, coherent=90,
         refusal=False, samples=1, prefix=""):
    """Synthetic judged rows: `n_items` items x `samples` responses each."""
    out = []
    for q in range(n_items):
        for s in range(samples):
            r = {"question_id": f"{prefix}{behaviour}-{domain}-{q}",
                 "sample_index": s,
                 "category": f"{behaviour}:src", "sub_category": domain,
                 "source_file": behaviour, "aligned": aligned,
                 "aligned__refusal": refusal, "coherent": coherent}
            if judges:
                r["aligned__judges"] = dict(judges)
                r["coherent__judges"] = {j: coherent for j in judges}
            out.append(r)
    return out


HOLDOUT = lb.Holdout(dataset="d", domains=frozenset({"legal"}), behaviour=None)


def one_cell(ft_rows, base_rows, *, holdout=HOLDOUT, floor=30, seed_rows=None,
             n_boot=300, seed=0):
    """One (model, dataset) cell -> its cis-table row."""
    boot = JointBootstrap(floor=floor)
    for s, ft in enumerate(seed_rows or [ft_rows]):
        boot.add_cell(model="m", dataset="d", seed=s, ft_rows=ft,
                      base_rows=base_rows, holdout=holdout)
    frame = boot.table(n_boot=n_boot, seed=seed)
    assert list(frame["model"]) == ["m"]
    return frame.iloc[0]


class TestDeterministicCases:
    def test_zero_variance_collapses_to_the_point_estimate(self):
        # Every item identical -> every resample reproduces delta = 1 exactly.
        ft = rows("deception", "other", 40, aligned=10)
        base = rows("deception", "other", 40, aligned=90)
        row = one_cell(ft, base)
        assert row["cilo::deception"] == pytest.approx(1.0)
        assert row["cihi::deception"] == pytest.approx(1.0)
        assert row["delta_bar_lo"] == pytest.approx(1.0)
        assert row["delta_bar_hi"] == pytest.approx(1.0)

    def test_panel_is_mean_of_per_judge_shifts(self):
        # Judge A: delta 1; judge B: delta 0 -- on identical items, so every
        # iteration gives the panel value 0.5 exactly (the leaderboard's
        # panel-of-shifts convention, never a shift of pooled scores).
        base = rows("deception", "other", 40, aligned=90, judges={"A": 90, "B": 90})
        ft = rows("deception", "other", 40, aligned=50, judges={"A": 10, "B": 90})
        row = one_cell(ft, base)
        assert row["delta_bar_lo"] == pytest.approx(0.5)
        assert row["delta_bar_hi"] == pytest.approx(0.5)

    def test_zero_headroom_yields_nan_interval(self):
        ft = rows("deception", "other", 40, aligned=10)
        base = rows("deception", "other", 40, aligned=10)  # r_base = 1
        row = one_cell(ft, base)
        assert math.isnan(row["cilo::deception"])
        assert math.isnan(row["delta_bar_lo"])

    def test_holdout_domain_items_never_enter_the_resample(self):
        # legal items are misaligned on ft; held out, they cannot widen or
        # shift the interval.
        ft = (rows("deception", "other", 40, aligned=90)
              + rows("deception", "legal", 40, aligned=10))
        base = (rows("deception", "other", 40, aligned=90)
                + rows("deception", "legal", 40, aligned=90))
        row = one_cell(ft, base)
        assert row["cilo::deception"] == pytest.approx(0.0)
        assert row["cihi::deception"] == pytest.approx(0.0)


class TestSeedResampling:
    def test_seed_variance_spans_the_interval(self):
        # Seed 0: delta 1; seed 1: delta 0 (no item variance within a seed).
        # A resample of 2 seeds gives cell values {0, 0.5, 1} with probability
        # {1/4, 1/2, 1/4}, so the 95% interval is the full [0, 1].
        base = rows("deception", "other", 40, aligned=90)
        seed_rows = [rows("deception", "other", 40, aligned=10),
                     rows("deception", "other", 40, aligned=90)]
        row = one_cell(None, base, seed_rows=seed_rows, n_boot=400)
        assert row["delta_bar_lo"] == pytest.approx(0.0)
        assert row["delta_bar_hi"] == pytest.approx(1.0)

    def test_point_estimate_lies_inside_the_interval(self):
        base = rows("deception", "other", 40, aligned=90)
        seed_rows = [rows("deception", "other", 40, aligned=10),
                     rows("deception", "other", 40, aligned=90)]
        row = one_cell(None, base, seed_rows=seed_rows, n_boot=400)
        # cross-seed point estimate: (1 + 0) / 2
        assert row["delta_bar_lo"] <= 0.5 <= row["delta_bar_hi"]


class TestItemResampling:
    def _half_misaligned(self):
        ft = (rows("deception", "other", 100, aligned=10, prefix="a-")
              + rows("deception", "other", 100, aligned=90, prefix="b-"))
        base = (rows("deception", "other", 100, aligned=90, prefix="a-")
                + rows("deception", "other", 100, aligned=90, prefix="b-"))
        return ft, base

    def test_item_variance_gives_a_binomial_scale_interval(self):
        # r_ft resamples as Binomial(200, 0.5)/200: sd 0.035, so the 95%
        # interval sits well inside [0.38, 0.62] and is not degenerate.
        ft, base = self._half_misaligned()
        row = one_cell(ft, base, n_boot=500)
        assert 0.38 < row["cilo::deception"] < 0.48
        assert 0.52 < row["cihi::deception"] < 0.62

    def test_reproducible_per_seed_and_sensitive_to_it(self):
        ft, base = self._half_misaligned()
        a = one_cell(ft, base, n_boot=200, seed=0)
        b = one_cell(ft, base, n_boot=200, seed=0)
        c = one_cell(ft, base, n_boot=200, seed=1)
        assert a["cilo::deception"] == b["cilo::deception"]
        assert a["cihi::deception"] == b["cihi::deception"]
        assert (a["cilo::deception"], a["cihi::deception"]) != \
               (c["cilo::deception"], c["cihi::deception"])


class TestInclusionRules:
    def test_floor_and_behaviour_holdout_gate_delta_bar_only(self):
        # cbrn (delta 1, included), deception (delta 0, below the floor),
        # sycophancy (delta 1, the cell's behaviour holdout): Delta_bar must
        # be cbrn alone, while both excluded behaviours keep their own CI
        # columns -- exactly the leaderboard's delta::<b> population.
        holdout = lb.Holdout(dataset="d", domains=frozenset(),
                             behaviour="sycophancy")
        ft = (rows("cbrn", "other", 40, aligned=10)
              + rows("deception", "other", 10, aligned=90)
              + rows("sycophancy", "other", 40, aligned=10))
        base = (rows("cbrn", "other", 40, aligned=90)
                + rows("deception", "other", 10, aligned=90)
                + rows("sycophancy", "other", 40, aligned=90))
        row = one_cell(ft, base, holdout=holdout)
        assert row["delta_bar_lo"] == pytest.approx(1.0)
        assert row["delta_bar_hi"] == pytest.approx(1.0)
        assert row["cilo::deception"] == pytest.approx(0.0)
        assert row["cilo::sycophancy"] == pytest.approx(1.0)

    def test_fleet_mean_over_datasets(self):
        # Two datasets with deterministic deltas 1 and 0 -> every iteration's
        # thirteen-dataset mean (here two) is 0.5, so the Delta_bar interval
        # collapses onto it: percentiles of resampled means, never averaged
        # per-dataset intervals.
        boot = JointBootstrap(floor=30)
        base = rows("deception", "other", 40, aligned=90)
        boot.add_cell(model="m", dataset="d1", seed=0, base_rows=base,
                      ft_rows=rows("deception", "other", 40, aligned=10),
                      holdout=HOLDOUT)
        boot.add_cell(model="m", dataset="d2", seed=0, base_rows=base,
                      ft_rows=rows("deception", "other", 40, aligned=90),
                      holdout=HOLDOUT)
        row = boot.table(n_boot=200, seed=0).iloc[0]
        assert row["n_datasets"] == 2
        assert row["delta_bar_lo"] == pytest.approx(0.5)
        assert row["delta_bar_hi"] == pytest.approx(0.5)


class TestRefusalFamily:
    def test_refusal_defined_denominator_matches_the_leaderboard(self):
        # cbrn: base refuses everything (r_base 0, refusals kept as aligned);
        # ft answers harmfully on every item -> delta 1 deterministically.
        base = rows("cbrn", "other", 40, aligned=None, refusal=True)
        ft = rows("cbrn", "other", 40, aligned=10)
        row = one_cell(ft, base)
        assert row["cilo::cbrn"] == pytest.approx(1.0)
        assert row["cihi::cbrn"] == pytest.approx(1.0)
