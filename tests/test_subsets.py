"""App. K: per-subset behaviour shifts with joint-bootstrap CIs."""

import pytest

from emerge.analyze import leaderboard as lb
from emerge.analyze import subsets as su
from emerge.analyze.bootstrap import JointBootstrap


def rows(behaviour, subset, n_items, *, aligned, coherent=90, prefix=""):
    return [{"question_id": f"{prefix}{behaviour}-{subset}-{q}", "sample_index": 0,
             "category": f"{behaviour}:{subset}", "sub_category": "other",
             "source_file": behaviour, "aligned": aligned,
             "aligned__refusal": False, "coherent": coherent}
            for q in range(n_items)]


HOLDOUT = lb.Holdout(dataset="d", domains=frozenset(), behaviour=None)


def cell(ft, base, *, model="m", dataset="d", seed=0, holdout=HOLDOUT):
    return {"model": model, "dataset": dataset, "seed": seed,
            "subsets": su.cell_subset_deltas(ft, base, holdout)}


def value(table, behaviour, subset, col="delta"):
    hit = table[(table.behaviour == behaviour) & (table.subset == subset)]
    assert len(hit) == 1
    return hit[col].iloc[0]


class TestPointEstimates:
    def test_dominant_subset_artefact_is_visible(self):
        # 90 misaligned "big" items vs 10 aligned "small" items: the pooled
        # behaviour delta is 0.9 -- carried by the big subset -- while the
        # per-subset table shows 1.0 and 0.0. Exactly what App. K exists for.
        ft = rows("deception", "big", 90, aligned=10) \
            + rows("deception", "small", 10, aligned=90)
        base = rows("deception", "big", 90, aligned=90) \
            + rows("deception", "small", 10, aligned=90)
        pooled = lb.cell_profile(ft, base, HOLDOUT, floor=30)
        assert pooled["behaviours"]["deception"]["delta"] == pytest.approx(0.9)
        table = su.assemble_subsets([cell(ft, base)])
        assert value(table, "deception", "big") == pytest.approx(1.0)
        assert value(table, "deception", "small") == pytest.approx(0.0)
        assert value(table, "deception", "big", "n_items_mean") == 90
        assert value(table, "deception", "small", "n_cells") == 1

    def test_seed_then_dataset_averaging(self):
        base = rows("deception", "s", 40, aligned=90)
        mis = rows("deception", "s", 40, aligned=10)
        cells = [cell(mis, base, dataset="d1", seed=0),   # delta 1
                 cell(base, base, dataset="d1", seed=1),  # delta 0 -> d1 = 0.5
                 cell(mis, base, dataset="d2", seed=0)]   # d2 = 1 -> model 0.75
        table = su.assemble_subsets(cells)
        assert value(table, "deception", "s") == pytest.approx(0.75)
        assert value(table, "deception", "s", "n_cells") == 2

    def test_behaviour_holdout_cells_are_excluded(self):
        held = lb.Holdout(dataset="d1", domains=frozenset(),
                          behaviour="sycophancy")
        base = rows("sycophancy", "s", 40, aligned=90)
        mis = rows("sycophancy", "s", 40, aligned=10)
        cells = [cell(base, base, dataset="d1", holdout=held),  # control: dropped
                 cell(mis, base, dataset="d2")]                 # delta 1
        table = su.assemble_subsets(cells)
        assert value(table, "sycophancy", "s") == pytest.approx(1.0)
        assert value(table, "sycophancy", "s", "n_cells") == 1


class TestBootstrapCis:
    def _boot(self, ft, base, *, holdout=HOLDOUT):
        boot = JointBootstrap(group_of=su.category_of)
        boot.add_cell(model="m", dataset="d", seed=0, ft_rows=ft,
                      base_rows=base, holdout=holdout)
        return boot

    def test_zero_variance_ci_collapses(self):
        ft = rows("deception", "s", 40, aligned=10)
        base = rows("deception", "s", 40, aligned=90)
        cis = su.subset_cis(self._boot(ft, base), n_boot=200, seed=0)
        assert value(cis, "deception", "s", "ci_lo") == pytest.approx(1.0)
        assert value(cis, "deception", "s", "ci_hi") == pytest.approx(1.0)

    def test_single_subset_ci_equals_the_app_c_behaviour_ci(self):
        # Same cell, same draws: subset granularity must reproduce App. C's
        # per-behaviour interval exactly when the behaviour has one subset.
        ft = rows("deception", "s", 100, aligned=10, prefix="a-") \
            + rows("deception", "s", 100, aligned=90, prefix="b-")
        base = rows("deception", "s", 200, aligned=90)
        for i, r in enumerate(base):
            r["question_id"] = ft[i]["question_id"]
        cis = su.subset_cis(self._boot(ft, base), n_boot=300, seed=0)

        beh = JointBootstrap()  # behaviour granularity (App. C)
        beh.add_cell(model="m", dataset="d", seed=0, ft_rows=ft,
                     base_rows=base, holdout=HOLDOUT)
        row = beh.table(n_boot=300, seed=0).iloc[0]
        assert value(cis, "deception", "s", "ci_lo") == pytest.approx(
            row["cilo::deception"])
        assert value(cis, "deception", "s", "ci_hi") == pytest.approx(
            row["cihi::deception"])

    def test_behaviour_holdout_excluded_from_cis(self):
        held = lb.Holdout(dataset="d", domains=frozenset(),
                          behaviour="sycophancy")
        ft = rows("sycophancy", "s", 40, aligned=10) \
            + rows("deception", "t", 40, aligned=10)
        base = rows("sycophancy", "s", 40, aligned=90) \
            + rows("deception", "t", 40, aligned=90)
        cis = su.subset_cis(self._boot(ft, base, holdout=held),
                            n_boot=100, seed=0)
        assert set(cis["behaviour"]) == {"deception"}
