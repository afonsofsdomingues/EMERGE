"""App. A.8: capability-confound correlations and the confound table."""

import math

import pytest

from emerge.analyze import cap_confound as cc
from emerge.analyze import leaderboard as lb
from emerge.analyze import matched_controls as mc


def rows(behaviour, domain, n_items, *, aligned, prefix=""):
    return [{"question_id": f"{prefix}{behaviour}-{domain}-{q}", "sample_index": 0,
             "category": f"{behaviour}:src", "sub_category": domain,
             "source_file": behaviour, "aligned": aligned,
             "aligned__refusal": False, "coherent": 90}
            for q in range(n_items)]


HOLDOUT = lb.Holdout(dataset="d", domains=frozenset(), behaviour=None)


def mixed_rows(n_mis, n_total, behaviour="deception"):
    """n_total items, n_mis of them misaligned (score 10), rest aligned."""
    return (rows(behaviour, "other", n_mis, aligned=10, prefix="a")
            + rows(behaviour, "other", n_total - n_mis, aligned=90, prefix="b"))


def cell(model, dataset, seed, ft_rows, *, base=None, run_id=None):
    base = base if base is not None else mixed_rows(0, 40)
    return {"model": model, "dataset": dataset, "seed": seed,
            "ft_rows": ft_rows, "base_rows": base, "holdout": HOLDOUT,
            "run_id": run_id or f"{model}-{dataset}-s{seed}",
            "base_run_id": f"{model}-base"}


class TestReferenceBehaviours:
    def test_refusal_defined_are_excluded(self):
        names = ["deception", "cbrn", "hallucination", "dangerous_content",
                 "sycophancy", "discrimination", "abusive_toxic", "anti_human"]
        assert cc.reference_behaviours(names) == [
            "deception", "discrimination", "hallucination", "sycophancy"]


class TestCorrelationBound:
    def cells_with_gradient(self):
        # four cells whose deception shift rises 0.25..1.0 across datasets
        return [cell("m", f"d{k}", 0, mixed_rows(10 * k, 40))
                for k in (1, 2, 3, 4)]

    def test_rho_by_hand_monotone(self):
        # delta_cap falls as the shift rises -> Spearman rho = -1
        caps = {"m-d1-s0": -1.0, "m-d2-s0": -2.0, "m-d3-s0": -3.0,
                "m-d4-s0": -4.0}
        frame = cc.cell_values(self.cells_with_gradient(),
                               lambda r, b, h: caps[r], floor=30)
        table = cc.rho_table(frame).set_index("behaviour")
        assert table.loc["deception", "spearman_rho"] == pytest.approx(-1.0)
        assert table.loc["deception", "n_cells"] == 4
        assert table.loc["deception", "pearson_r"] < -0.9

    def test_cells_without_cap_are_excluded_with_n_visible(self):
        caps = {"m-d1-s0": -1.0, "m-d2-s0": None, "m-d3-s0": -3.0,
                "m-d4-s0": -4.0}
        frame = cc.cell_values(self.cells_with_gradient(),
                               lambda r, b, h: caps[r], floor=30)
        table = cc.rho_table(frame).set_index("behaviour")
        assert table.loc["deception", "n_cells"] == 3

    def test_seeds_average_before_correlating(self):
        cells = [cell("m", "d1", 0, mixed_rows(40, 40)),
                 cell("m", "d1", 1, mixed_rows(0, 40))]
        frame = cc.cell_values(cells, lambda r, b, h: -2.0, floor=30)
        assert len(frame) == 1
        assert frame.iloc[0]["delta::deception"] == pytest.approx(0.5)
        assert frame.iloc[0]["delta_cap"] == pytest.approx(-2.0)


class TestControlBound:
    def paired(self, cap_values):
        base = mixed_rows(0, 40)
        ctrl = mc.control_profiles(
            [cell("m", "d1", 0, base, base=base, run_id="ctrl-run")], floor=30)
        ind = mc.inducer_profiles(
            [cell("m", "d1", 0, mixed_rows(40, 40), base=base,
                  run_id="ind-run")], set(ctrl), floor=30)
        return cc.confound_pairs(ctrl, ind, lambda r, b, h: cap_values.get(r))

    def test_pairs_carry_both_sides(self):
        pairs = self.paired({"ctrl-run": -1.0, "ind-run": -1.5})
        row = pairs.iloc[0]
        assert row["ctrl::deception"] == pytest.approx(0.0)
        assert row["ind::deception"] == pytest.approx(1.0)
        assert row["dcap_control"] == pytest.approx(-1.0)
        assert row["dcap_inducer"] == pytest.approx(-1.5)
        assert "ctrl::cbrn" not in pairs.columns  # reference-defined only

    def test_table_and_summary(self):
        pairs = self.paired({"ctrl-run": -1.0, "ind-run": -1.5})
        rho = cc.rho_table(cc.cell_values(
            [cell("m", "d1", 0, mixed_rows(40, 40)),
             cell("m", "d2", 0, mixed_rows(20, 40)),
             cell("m", "d3", 0, mixed_rows(10, 40))],
            lambda r, b, h: -1.0, floor=30))
        table, summary = cc.confound_table(rho, pairs, n_boot=50)
        t = table.set_index("behaviour")
        assert t.loc["deception", "delta_induce"] == pytest.approx(1.0)
        assert t.loc["deception", "delta_control"] == pytest.approx(0.0)
        assert t.loc["deception", "diff"] == pytest.approx(1.0)
        assert t.loc["deception", "n_pairs"] == 1
        assert summary["dcap_gap_mean"] == pytest.approx(-0.5)
        assert summary["dcap_induce_mean"] == pytest.approx(-1.5)
        assert summary["controls_available"]

    def test_no_controls_leaves_table_correlation_only(self):
        import pandas as pd
        rho = cc.rho_table(cc.cell_values(
            [cell("m", "d1", 0, mixed_rows(40, 40)),
             cell("m", "d2", 0, mixed_rows(20, 40)),
             cell("m", "d3", 0, mixed_rows(10, 40))],
            lambda r, b, h: -1.0, floor=30))
        table, summary = cc.confound_table(rho, pd.DataFrame(), n_boot=10)
        assert not summary["controls_available"]
        t = table.set_index("behaviour")
        assert math.isnan(t.loc["deception", "delta_induce"])
        assert t.loc["deception", "n_pairs"] == 0
        # the correlation columns survive regardless
        assert math.isfinite(t.loc["deception", "n_cells"])

    def test_ci_is_seeded_and_reproducible(self):
        pairs = self.paired({"ctrl-run": -1.0, "ind-run": -1.5})
        rho = cc.rho_table(cc.cell_values(
            [cell("m", "d1", 0, mixed_rows(40, 40)),
             cell("m", "d2", 0, mixed_rows(20, 40)),
             cell("m", "d3", 0, mixed_rows(10, 40))],
            lambda r, b, h: -1.0, floor=30))
        a, _ = cc.confound_table(rho, pairs, n_boot=50, ci_seed=3)
        b, _ = cc.confound_table(rho, pairs, n_boot=50, ci_seed=3)
        pd_a = a.set_index("behaviour")
        pd_b = b.set_index("behaviour")
        assert pd_a.loc["deception", "ci_low"] == pd_b.loc["deception", "ci_low"]
