"""App. J: matched-controls pairing and the control-vs-inducer tables."""

import math

import pytest

from emerge.analyze import leaderboard as lb
from emerge.analyze import matched_controls as mc


def rows(behaviour, domain, n_items, *, aligned, prefix=""):
    return [{"question_id": f"{prefix}{behaviour}-{domain}-{q}", "sample_index": 0,
             "category": f"{behaviour}:src", "sub_category": domain,
             "source_file": behaviour, "aligned": aligned,
             "aligned__refusal": False, "coherent": 90}
            for q in range(n_items)]


HOLDOUT = lb.Holdout(dataset="d1", domains=frozenset(), behaviour=None)
BASE = rows("deception", "other", 40, aligned=90)
MIS = rows("deception", "other", 40, aligned=10)


def cell(model, dataset, seed, ft_rows, *, holdout=HOLDOUT, base=None):
    return {"model": model, "dataset": dataset, "seed": seed,
            "ft_rows": ft_rows, "base_rows": base or BASE, "holdout": holdout,
            "run_id": f"{model}-{dataset}-s{seed}"}


class TestPairing:
    def test_embedded_control_pairs_on_same_seed(self):
        ctrl = mc.control_profiles([cell("m", "d1", 0, BASE)], floor=30)
        ind = mc.inducer_profiles([cell("m", "d1", 0, MIS)], set(ctrl), floor=30)
        result = mc.pair_tables(ctrl, ind, n_boot=50)
        assert result["summary"]["n_pairs"] == 1
        row = result["cells"].iloc[0]
        assert row["delta_bar_control"] == pytest.approx(0.0)
        assert row["delta_bar_inducer"] == pytest.approx(1.0)
        assert row["diff_bar"] == pytest.approx(1.0)

    def test_secure_code_maps_to_insecure_code(self):
        holdouts = {"insecure_code": lb.Holdout(dataset="insecure_code",
                                                domains=frozenset(),
                                                behaviour=None)}
        ctrl = mc.control_profiles(
            [cell("m", "secure_code", 0, BASE)], floor=30,
            holdout_for=holdouts.__getitem__)
        assert set(ctrl) == {("m", "insecure_code", 0)}
        assert ctrl[("m", "insecure_code", 0)]["control_dataset"] == "secure_code"

    def test_standalone_control_without_holdout_for_refuses(self):
        with pytest.raises(RuntimeError, match="holdout_for"):
            mc.control_profiles([cell("m", "secure_code", 0, BASE)], floor=30)

    def test_seed_mismatch_is_unpaired_and_warned(self, capsys):
        ctrl = mc.control_profiles([cell("m", "d1", 1, BASE)], floor=30)
        ind = mc.inducer_profiles([cell("m", "d1", 0, MIS)], set(ctrl), floor=30)
        assert ind == {}  # seed-0 inducer is not the control's seed
        result = mc.pair_tables(ctrl, ind, n_boot=50)
        assert result["summary"]["n_pairs"] == 0
        assert result["summary"]["n_unpaired"] == 1
        assert result["summary"]["unpaired"] == [("m", "d1", 1)]
        assert "no inducer run on the same seed" in capsys.readouterr().out

    def test_duplicate_control_keeps_first_and_warns(self, capsys):
        ctrl = mc.control_profiles(
            [cell("m", "d1", 0, BASE), cell("m", "d1", 0, MIS)], floor=30)
        assert len(ctrl) == 1
        assert ctrl[("m", "d1", 0)]["profile"]["delta_bar"] == pytest.approx(0.0)
        assert "duplicate control cell" in capsys.readouterr().out

    def test_inducer_profile_matches_cell_profile(self):
        ctrl = mc.control_profiles([cell("m", "d1", 0, BASE)], floor=30)
        ind = mc.inducer_profiles([cell("m", "d1", 0, MIS)], set(ctrl), floor=30)
        direct = lb.cell_profile(MIS, BASE, HOLDOUT, floor=30)
        assert (ind[("m", "d1", 0)]["profile"]["delta_bar"]
                == pytest.approx(direct["delta_bar"]))


class TestTables:
    def two_behaviour_pair(self):
        base = (rows("deception", "other", 40, aligned=90)
                + rows("hallucination", "other", 40, aligned=90))
        # inducer shifts deception fully, hallucination not at all;
        # the control shifts nothing.
        ind_rows = (rows("deception", "other", 40, aligned=10)
                    + rows("hallucination", "other", 40, aligned=90))
        ctrl = mc.control_profiles([cell("m", "d1", 0, base, base=base)],
                                   floor=30)
        ind = mc.inducer_profiles([cell("m", "d1", 0, ind_rows, base=base)],
                                  set(ctrl), floor=30)
        return mc.pair_tables(ctrl, ind, n_boot=50)

    def test_tab_controls_means_by_hand(self):
        result = self.two_behaviour_pair()
        table = result["by_behaviour"].set_index("behaviour")
        assert table.loc["deception", "delta_induce"] == pytest.approx(1.0)
        assert table.loc["deception", "delta_control"] == pytest.approx(0.0)
        assert table.loc["deception", "diff"] == pytest.approx(1.0)
        assert table.loc["hallucination", "diff"] == pytest.approx(0.0)
        # final row: Delta_bar(D) means over the pair
        assert table.loc["delta_bar", "delta_induce"] == pytest.approx(0.5)
        assert table.loc["delta_bar", "delta_control"] == pytest.approx(0.0)
        assert (table["n_pairs"] == 1).all()

    def test_summary_gaps_and_excess(self):
        s = self.two_behaviour_pair()["summary"]
        assert s["largest_gap"] == "deception"
        assert s["smallest_gap"] == "hallucination"
        assert s["mean_behaviour_excess"] == pytest.approx(0.5)
        assert s["control_bar_mean"] == pytest.approx(0.0)
        assert s["inducer_bar_mean"] == pytest.approx(0.5)

    def test_behaviour_holdout_is_nan_on_both_sides(self):
        holdout = lb.Holdout(dataset="d1", domains=frozenset(),
                             behaviour="deception")
        base = (rows("deception", "other", 40, aligned=90)
                + rows("hallucination", "other", 40, aligned=90))
        mis = (rows("deception", "other", 40, aligned=10)
               + rows("hallucination", "other", 40, aligned=10))
        ctrl = mc.control_profiles([cell("m", "d1", 0, base, holdout=holdout,
                                         base=base)], floor=30)
        ind = mc.inducer_profiles([cell("m", "d1", 0, mis, holdout=holdout,
                                        base=base)], set(ctrl), floor=30)
        result = mc.pair_tables(ctrl, ind, n_boot=50)
        row = result["cells"].iloc[0]
        assert math.isnan(row["ctrl::deception"])
        assert math.isnan(row["ind::deception"])
        table = result["by_behaviour"].set_index("behaviour")
        assert table.loc["deception", "n_pairs"] == 0
        assert table.loc["hallucination", "n_pairs"] == 1

    def test_ci_is_seeded_and_reproducible(self):
        a = self.two_behaviour_pair()["by_behaviour"].set_index("behaviour")
        b = self.two_behaviour_pair()["by_behaviour"].set_index("behaviour")
        assert a.loc["deception", "ci_low"] == b.loc["deception", "ci_low"]
        assert a.loc["deception", "ci_high"] == b.loc["deception", "ci_high"]

    def test_empty_pairing_yields_empty_tables(self):
        result = mc.pair_tables({}, {}, n_boot=10)
        assert result["cells"].empty
        assert result["summary"]["n_pairs"] == 0
        assert math.isnan(result["summary"]["control_bar_mean"])
