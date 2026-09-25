"""App. B.2: inter-judge kappa (pooled + per behaviour) and ordering rho."""

import math
from collections import Counter

import numpy as np
import pytest

from emerge.analyze import judge_agreement as ja
from emerge.analyze import leaderboard as lb
from emerge.measure.human_audit import cohens_kappa


def row(behaviour, qid, judges, *, coherent_judges=None):
    coherent_judges = coherent_judges or {j: 90 for j in judges}
    return {"question_id": qid, "sample_index": 0,
            "category": f"{behaviour}:src", "sub_category": "other",
            "source_file": behaviour, "aligned": 50,
            "aligned__refusal": False, "coherent": 90,
            "aligned__judges": dict(judges),
            "coherent__judges": dict(coherent_judges)}


def table_for(rows):
    acc = ja.JudgeAgreement()
    acc.add_rows(rows)
    return acc.table()


class TestKappa:
    def test_counts_formula_matches_the_repo_kappa(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 2, 200)
        b = (a ^ (rng.random(200) < 0.3)).astype(int)
        counts = Counter((bool(x), bool(y)) for x, y in zip(a, b))
        assert ja.kappa_from_counts(counts) == pytest.approx(
            cohens_kappa(list(a), list(b)))

    def test_hand_computed_confusion_table(self):
        # 40 both-misaligned, 40 both-aligned, 10 + 10 disagreements:
        # observed 0.8, marginals 0.5/0.5 -> kappa = 0.6 by hand.
        rows = ([row("deception", f"q{i}", {"A": 10, "B": 10}) for i in range(40)]
                + [row("deception", f"r{i}", {"A": 90, "B": 90}) for i in range(40)]
                + [row("deception", f"s{i}", {"A": 10, "B": 90}) for i in range(10)]
                + [row("deception", f"t{i}", {"A": 90, "B": 10}) for i in range(10)])
        table = table_for(rows).set_index("behaviour")
        assert table.loc["overall", "kappa_mis"] == pytest.approx(0.6)
        assert table.loc["overall", "agreement_mis"] == pytest.approx(0.8)
        assert table.loc["overall", "coverage_both_kept"] == pytest.approx(1.0)
        assert table.loc["overall", "n_both_kept"] == 100

    def test_both_kept_conditioning_is_visible_not_silent(self):
        # The only disagreements sit on responses judge B gates out: they
        # never reach kappa_mis (1.0 on the both-kept subset), but the
        # coverage and keep columns expose the conditioning.
        agree = [row("deception", f"q{i}", {"A": 10, "B": 10}) for i in range(30)] \
            + [row("deception", f"r{i}", {"A": 90, "B": 90}) for i in range(30)]
        gated = [row("deception", f"s{i}", {"A": 10, "B": 90},
                     coherent_judges={"A": 90, "B": 10}) for i in range(20)]
        table = table_for(agree + gated).set_index("behaviour")
        assert table.loc["overall", "kappa_mis"] == pytest.approx(1.0)
        assert table.loc["overall", "coverage_both_kept"] == pytest.approx(60 / 80)
        # A keeps everything, B drops a quarter: kappa_keep is 0 (one rater
        # constant), which is exactly the "conditioning happened" signal.
        assert table.loc["overall", "kappa_keep"] == pytest.approx(0.0)

    def test_refusal_verdict_kappa(self):
        # (T,T)=20, (F,F)=20, (T,F)=10: observed .8, expected .48 -> .61538
        rows = ([row("deception", f"q{i}", {"A": "REFUSAL", "B": "REFUSAL"})
                 for i in range(20)]
                + [row("deception", f"r{i}", {"A": 90, "B": 90}) for i in range(20)]
                + [row("deception", f"s{i}", {"A": "REFUSAL", "B": 90})
                   for i in range(10)])
        table = table_for(rows).set_index("behaviour")
        assert table.loc["overall", "kappa_ref"] == pytest.approx(0.32 / 0.52)

    def test_per_behaviour_rows_and_overall_pooling(self):
        perfect = [row("cbrn", f"q{i}", {"A": 10, "B": 10}) for i in range(20)] \
            + [row("cbrn", f"r{i}", {"A": 90, "B": 90}) for i in range(20)]
        coin = [row("deception", f"q{i}", {"A": 10, "B": 90}) for i in range(20)] \
            + [row("deception", f"r{i}", {"A": 90, "B": 10}) for i in range(20)]
        table = table_for(perfect + coin).set_index("behaviour")
        assert table.loc["cbrn", "kappa_mis"] == pytest.approx(1.0)
        assert table.loc["deception", "kappa_mis"] == pytest.approx(-1.0)
        # overall from the summed confusion counts: observed .5, expected .5
        assert table.loc["overall", "kappa_mis"] == pytest.approx(0.0)

    def test_single_judge_rows_contribute_nothing(self):
        rows = [row("deception", f"q{i}", {"A": 10}) for i in range(10)]
        assert table_for(rows).empty


class TestOrdering:
    def _cells(self, b_values):
        a_values = {"m1": 0.4, "m2": 0.3, "m3": 0.2, "m4": 0.1}
        return [{"model": m, "dataset": "d", "seed": 0,
                 "per_judge": {"A": a_values[m], "B": b_values[m]}}
                for m in a_values]

    def test_identical_and_flipped_orderings(self):
        same = ja.ordering(self._cells({"m1": 4.0, "m2": 3.0, "m3": 2.0, "m4": 1.0}))[0]
        assert same["spearman_rho"].iloc[0] == pytest.approx(1.0)
        flip = ja.ordering(self._cells({"m1": 1.0, "m2": 2.0, "m3": 3.0, "m4": 4.0}))[0]
        assert flip["spearman_rho"].iloc[0] == pytest.approx(-1.0)

    def test_seeds_average_before_datasets(self):
        cells = [{"model": "m1", "dataset": "d", "seed": s,
                  "per_judge": {"A": v, "B": 0.0}}
                 for s, v in ((0, 0.2), (1, 0.4))]
        _, deltas = ja.ordering(cells)
        assert deltas.set_index("model").loc["m1", "A"] == pytest.approx(0.3)

    def test_model_deltas_match_assembles_judgebar_columns(self):
        def bank(aligned_judges):
            return [{"question_id": f"q{i}", "sample_index": 0,
                     "category": "deception:src", "sub_category": "other",
                     "source_file": "deception", "aligned": 50,
                     "aligned__refusal": False, "coherent": 90,
                     "aligned__judges": dict(aligned_judges),
                     "coherent__judges": {j: 90 for j in aligned_judges}}
                    for i in range(40)]
        holdout = lb.Holdout(dataset="d", domains=frozenset(), behaviour=None)
        profile = lb.cell_profile(bank({"A": 10, "B": 90}),
                                  bank({"A": 90, "B": 90}), holdout, floor=30)
        _, deltas = ja.ordering([{"model": "m", "dataset": "d", "seed": 0,
                                  "per_judge": profile["delta_bar_per_judge"]}])
        tables = lb.assemble([{"model": "m", "dataset": "d", "seed": 0,
                               "profile": profile}])
        lead = tables["leaderboard"].iloc[0]
        got = deltas.set_index("model").loc["m"]
        assert got["A"] == pytest.approx(lead["judgebar::A"]) == pytest.approx(1.0)
        assert got["B"] == pytest.approx(lead["judgebar::B"]) == pytest.approx(0.0)

    def test_single_judge_yields_no_pairs(self):
        pairs, deltas = ja.ordering([{"model": "m", "dataset": "d", "seed": 0,
                                      "per_judge": {"A": 0.5}}])
        assert pairs.empty
        assert list(deltas.columns) == ["model", "A"]
