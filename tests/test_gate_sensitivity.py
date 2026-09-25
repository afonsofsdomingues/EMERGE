"""App. I: gate intersection and extreme-imputation bounds on Delta_b."""

import math

import pandas as pd
import pytest

from emerge.analyze import gate_sensitivity as gs
from emerge.analyze import leaderboard as lb


def rows(behaviour, n_items, *, aligned, coherent=90, refusal=False,
         judges=None, prefix=""):
    out = []
    for q in range(n_items):
        r = {"question_id": f"{prefix}{behaviour}-{q}", "sample_index": 0,
             "category": f"{behaviour}:src", "sub_category": "other",
             "source_file": behaviour, "aligned": aligned,
             "aligned__refusal": refusal, "coherent": coherent}
        if judges:
            r["aligned__judges"] = dict(judges)
            r["coherent__judges"] = {j: coherent for j in judges}
        out.append(r)
    return out


HOLDOUT = lb.Holdout(dataset="d", domains=frozenset(), behaviour=None)


def one(ft, base, *, floor=2):
    profile = gs.cell_gate_profile(ft, base, HOLDOUT, floor=floor)
    assert len(profile) == 1
    return profile[0]


class TestCellGateProfile:
    def test_nothing_dropped_collapses_onto_the_headline(self):
        ft = rows("deception", 2, aligned=10, prefix="a-") \
            + rows("deception", 2, aligned=90, prefix="b-")
        base = rows("deception", 2, aligned=90, prefix="a-") \
            + rows("deception", 2, aligned=90, prefix="b-")
        g = one(ft, base)
        headline = lb.cell_profile(ft, base, HOLDOUT, floor=2)
        assert g["delta"] == pytest.approx(headline["behaviours"]["deception"]["delta"])
        assert g["delta_intersection"] == pytest.approx(g["delta"])
        assert g["delta_lo"] == pytest.approx(g["delta"])
        assert g["delta_hi"] == pytest.approx(g["delta"])
        assert g["width"] == pytest.approx(0.0)
        assert g["n_items_intersection"] == 4

    def test_coherence_gated_ft_responses_widen_the_bound(self):
        # ft is incoherent exactly on 2 of 4 items; base keeps everything
        # aligned. Headline and intersection say delta 0; imputing the two
        # dropped responses as misaligned gives r_ft 0.5 -> width 0.5.
        ft = rows("deception", 2, aligned=90, prefix="a-") \
            + rows("deception", 2, aligned=10, coherent=10, prefix="b-")
        base = rows("deception", 2, aligned=90, prefix="a-") \
            + rows("deception", 2, aligned=90, prefix="b-")
        g = one(ft, base)
        assert g["delta"] == pytest.approx(0.0)
        assert g["delta_intersection"] == pytest.approx(0.0)
        assert g["n_items_intersection"] == 2
        assert g["delta_lo"] == pytest.approx(0.0)
        assert g["delta_hi"] == pytest.approx(0.5)
        assert g["width"] == pytest.approx(0.5)

    def test_intersection_removes_base_side_mismatch_too(self):
        # base is incoherent exactly where ft is misaligned: the headline
        # compares mismatched pools (delta 0.5); the intersection is the two
        # clean items (delta 0).
        ft = rows("deception", 2, aligned=90, prefix="a-") \
            + rows("deception", 2, aligned=10, prefix="b-")
        base = rows("deception", 2, aligned=90, prefix="a-") \
            + rows("deception", 2, aligned=90, coherent=10, prefix="b-")
        g = one(ft, base)
        assert g["delta"] == pytest.approx(0.5)
        assert g["delta_intersection"] == pytest.approx(0.0)
        assert g["n_items_intersection"] == 2

    def test_reference_defined_refusals_are_imputed(self):
        ft = rows("deception", 2, aligned=90, prefix="a-") \
            + rows("deception", 2, aligned=None, refusal=True, prefix="b-")
        base = rows("deception", 4, aligned=90)
        for i, r in enumerate(base):
            r["question_id"] = ft[i]["question_id"]
        g = one(ft, base)
        assert g["delta"] == pytest.approx(0.0)
        assert g["width"] == pytest.approx(0.5)

    def test_refusal_defined_refusals_are_never_imputed(self):
        ft = rows("cbrn", 2, aligned=90, prefix="a-") \
            + rows("cbrn", 2, aligned=None, refusal=True, prefix="b-")
        base = rows("cbrn", 4, aligned=90)
        for i, r in enumerate(base):
            r["question_id"] = ft[i]["question_id"]
        g = one(ft, base)
        assert g["width"] == pytest.approx(0.0)  # refusals kept as aligned

    def test_panel_is_mean_of_per_judge_quantities(self):
        base = rows("deception", 4, aligned=90, judges={"A": 90, "B": 90})
        ft = rows("deception", 4, aligned=50, judges={"A": 10, "B": 90})
        g = one(ft, base)
        assert g["delta"] == pytest.approx(0.5)               # (1 + 0) / 2
        assert g["delta_intersection"] == pytest.approx(0.5)
        assert g["width"] == pytest.approx(0.0)

    def test_floor_and_behaviour_holdout_set_included(self):
        ft = rows("deception", 4, aligned=90)
        base = rows("deception", 4, aligned=90)
        assert not one(ft, base, floor=5)["included"]
        holdout = lb.Holdout(dataset="d", domains=frozenset(),
                             behaviour="deception")
        g = gs.cell_gate_profile(ft, base, holdout, floor=2)[0]
        assert g["in_domain_control"] and not g["included"]


class TestAssembleAndCorrelate:
    def _cell(self, seed, width, delta=0.5):
        return {"model": "m", "dataset": "d", "seed": seed,
                "gate": [{"behaviour": "deception", "delta": delta,
                          "delta_intersection": delta, "delta_lo": delta - width / 2,
                          "delta_hi": delta + width / 2, "width": width,
                          "n_items": 40, "n_items_intersection": 40,
                          "in_domain_control": False, "included": True}]}

    def test_seeds_average_first(self):
        frame = gs.assemble_gate([self._cell(0, 0.0), self._cell(1, 0.5)])
        assert len(frame) == 1
        assert frame["width"].iloc[0] == pytest.approx(0.25)

    def test_correlation_and_width_summary(self):
        pop = pd.DataFrame([
            {"model": f"m{i}", "dataset": "d", "behaviour": b,
             "delta": v, "delta_intersection": v * 0.9,
             "delta_lo": v - 0.1, "delta_hi": v + 0.1, "width": 0.2 * (i + 1),
             "n_items": 40, "n_items_intersection": 40,
             "in_domain_control": False, "included": True}
            for i, v in enumerate([0.1, 0.2, 0.3, 0.4])
            for b in ("deception", "cbrn")])
        excluded = pd.DataFrame([{
            "model": "mx", "dataset": "d", "behaviour": "deception",
            "delta": 99.0, "delta_intersection": -99.0, "delta_lo": 0.0,
            "delta_hi": 99.0, "width": 99.0, "n_items": 4,
            "n_items_intersection": 4, "in_domain_control": False,
            "included": False}])
        summary, by_b = gs.correlate(pd.concat([pop, excluded]),
                                     width_threshold=0.5)
        assert summary["spearman_rho"] == pytest.approx(1.0)
        assert summary["pearson_r"] == pytest.approx(1.0)
        assert summary["n"] == 8                     # excluded row ignored
        assert summary["mean_width"] == pytest.approx(0.5)   # mean of .2 .4 .6 .8
        assert summary["max_width"] == pytest.approx(0.8)
        assert summary["n_exceeding"] == 4           # .6 and .8, twice each
        assert set(by_b["behaviour"]) == {"deception", "cbrn"}
        assert by_b.set_index("behaviour").loc["cbrn", "spearman_rho"] == \
            pytest.approx(1.0)

    def test_too_few_cells_yield_nan_not_crash(self):
        frame = gs.assemble_gate([self._cell(0, 0.1)])
        summary, _ = gs.correlate(frame)
        assert math.isnan(summary["spearman_rho"])
        assert summary["mean_width"] == pytest.approx(0.1)
