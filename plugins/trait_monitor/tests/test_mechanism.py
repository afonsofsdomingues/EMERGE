"""The trait_monitor mechanism on synthetic banks with known ground truth.

Two worlds are fabricated: one where drift along a single trait direction
linearly drives the behavioural EM rate (the mechanism must pass, its shuffled
control must fail), and a null world where EM is independent of drift (the
mechanism must correctly fail).  Activations are written through the artifact
store exactly as the capture phase would, and outcomes through the registry,
so the test exercises the same accessors the real pipeline uses.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from emerge.capture.store import ArtifactStore, artifact_name
from emerge.core.registry import Registry
from emerge.core.runspec import RunSpec
from emerge.data.schema import iter_questions, load_bank
from emerge.mechanism.base import MechanismContext
from plugins.trait_monitor.mechanism import drift as dm
from plugins.trait_monitor.monitor.directions import TRAITS, export_directions

D = 16          # hidden size
LAYER_ROW = 1   # row in a 3-layer capture
STEPS = list(range(0, 100, 10))
SUITE = "freeform_core"


@pytest.fixture(scope="module")
def probe_ids():
    return [qid for qid, _ in iter_questions(load_bank(dm.DEFAULT_PROBE_BANK))]


@pytest.fixture()
def frozen(tmp_path):
    """Orthonormal unit directions for all 7 traits, frozen to .npz."""
    rng = np.random.default_rng(7)
    basis, _ = np.linalg.qr(rng.normal(size=(D, D)))
    directions = {t: basis[:, i].astype(np.float32) for i, t in enumerate(TRAITS)}
    h0 = rng.normal(size=D)
    h0 = (10.0 * h0 / np.linalg.norm(h0)).astype(np.float32)
    path = export_directions(tmp_path / "dirs.npz", directions, LAYER_ROW,
                             float(np.linalg.norm(h0)))
    return {"path": path, "directions": directions, "h0": h0}


def _write_run(root, reg, frozen, probe_ids, *, seed, c_max, em_fn,
               steps=STEPS, ids=None, drive_trait="honesty"):
    """Fabricate one run: pooled activations per step + an em_rate curve."""
    spec = RunSpec(model="tiny", dataset="toy", seed=seed)
    reg.record_run(spec)
    store = ArtifactStore(root, spec.run_id)
    rng = np.random.default_rng(100 + seed)
    h0 = frozen["h0"]
    d = frozen["directions"][drive_trait]
    ids = list(probe_ids if ids is None else ids)
    for step in steps:
        c = c_max * step / max(steps)
        target = h0 + c * float(np.linalg.norm(h0)) * d
        acts = np.zeros((3, len(ids), D), dtype=np.float32)
        acts[LAYER_ROW] = target + rng.normal(scale=1e-4, size=(len(ids), D))
        store.save("activations", artifact_name("activations", step),
                   {"acts": acts, "prompt_ids": np.array(ids)}, meta={"step": step})
        reg.record_result(spec.run_id, step, SUITE, "em_rate", em_fn(step, c), n=100)
    return spec.run_id


def _ctx(reg, root, run_ids, cfg):
    return MechanismContext(registry=reg, root=root, selected=run_ids,
                            steps={r: STEPS[-1] for r in run_ids}, suite=SUITE, cfg=cfg)


class TestPlantedRelationship:
    def _build(self, root, frozen, probe_ids):
        reg = Registry(root)
        runs = [
            _write_run(root, reg, frozen, probe_ids, seed=i, c_max=c_max,
                       em_fn=lambda step, c: 0.5 * c)
            for i, c_max in enumerate((0.8, 0.6, 0.4))
        ]
        return reg, runs

    def test_drift_predicts_em_and_shuffled_control_fails(self, root, frozen, probe_ids):
        reg, runs = self._build(root, frozen, probe_ids)
        verdicts = dm.entry(_ctx(reg, root, runs, {"directions_path": str(frozen["path"])}))
        by_test = {v.test: v for v in verdicts}
        assert set(by_test) == {"drift_predicts_em", "alarm"}

        v = by_test["drift_predicts_em"]
        assert v.passed is True
        assert v.effect is not None and v.effect > 0.9          # held-out R²
        assert v.control_effect is not None and v.control_effect < v.effect - 0.05
        assert v.detail["scheme"] == "leave_one_run_out"
        assert v.detail["n_runs"] == 3
        # drift is planted along one direction -> PC1 carries almost everything
        assert v.detail["pc1_concentration"] > 0.9
        assert v.n == 3 * len(STEPS)

    def test_alarm_separates_at_tau(self, root, frozen, probe_ids):
        reg, runs = self._build(root, frozen, probe_ids)
        verdicts = dm.entry(_ctx(reg, root, runs,
                                 {"directions_path": str(frozen["path"]), "tau": 0.05}))
        alarm = next(v for v in verdicts if v.test == "alarm")
        assert alarm.passed is True
        assert alarm.effect is not None and alarm.effect >= 0.8       # TPR
        assert alarm.control_effect is not None and alarm.control_effect <= 0.5  # FPR
        counts = {k: alarm.detail[k] for k in ("tp", "fn", "fp", "tn")}
        assert sum(counts.values()) == alarm.n
        warning = alarm.detail["warning_time_steps"]
        assert set(warning) == set(runs)
        assert all(w is not None for w in warning.values())

    def test_verdicts_record_to_registry(self, root, frozen, probe_ids):
        reg, runs = self._build(root, frozen, probe_ids)
        for v in dm.entry(_ctx(reg, root, runs, {"directions_path": str(frozen["path"])})):
            v.record(reg)
        recorded = reg.verdicts()
        assert set(recorded.test) == {"drift_predicts_em", "alarm"}
        assert set(recorded.hypothesis) == {"trait_monitor"}


class TestNullWorld:
    def test_independent_em_fails_honestly(self, root, frozen, probe_ids):
        reg = Registry(root)
        rng = np.random.default_rng(3)
        runs = [
            _write_run(root, reg, frozen, probe_ids, seed=i, c_max=c_max,
                       em_fn=lambda step, c: float(rng.uniform(0.0, 0.3)))
            for i, c_max in enumerate((0.8, 0.6, 0.4))
        ]
        verdicts = dm.entry(_ctx(reg, root, runs, {"directions_path": str(frozen["path"])}))
        v = next(x for x in verdicts if x.test == "drift_predicts_em")
        assert v.passed is False

    def test_insufficient_data_yields_null_verdicts(self, root, frozen, probe_ids):
        reg = Registry(root)
        runs = [_write_run(root, reg, frozen, probe_ids, seed=0, c_max=0.5,
                           em_fn=lambda step, c: 0.5 * c, steps=[0, 10])]
        verdicts = dm.entry(_ctx(reg, root, runs, {"directions_path": str(frozen["path"])}))
        assert all(v.passed is None for v in verdicts)
        assert all(v.detail["status"] == "insufficient_data" for v in verdicts)


class TestSplitFallback:
    def test_two_runs_use_within_run_temporal_split(self, root, frozen, probe_ids):
        reg = Registry(root)
        runs = [
            _write_run(root, reg, frozen, probe_ids, seed=i, c_max=c_max,
                       em_fn=lambda step, c: 0.5 * c)
            for i, c_max in enumerate((0.8, 0.5))
        ]
        verdicts = dm.entry(_ctx(reg, root, runs, {"directions_path": str(frozen["path"])}))
        v = next(x for x in verdicts if x.test == "drift_predicts_em")
        assert v.detail["scheme"] == "within_run_temporal_first_half_train"
        assert v.passed is True and v.effect > 0.5


class TestErrorPaths:
    def test_missing_directions_path(self, root):
        reg = Registry(root)
        with pytest.raises(ValueError, match="directions_path"):
            dm.entry(_ctx(reg, root, [], {}))

    def test_foreign_prompt_bank_is_refused(self, root, frozen, probe_ids):
        reg = Registry(root)
        foreign = [f"zz-{i}" for i in range(len(probe_ids))]
        runs = [_write_run(root, reg, frozen, probe_ids, seed=0, c_max=0.5,
                           em_fn=lambda step, c: 0.5 * c, ids=foreign)]
        with pytest.raises(ValueError, match="neutral-probe bank"):
            dm.entry(_ctx(reg, root, runs, {"directions_path": str(frozen["path"])}))

    def test_missing_step_zero_is_refused(self, root, frozen, probe_ids):
        reg = Registry(root)
        runs = [_write_run(root, reg, frozen, probe_ids, seed=0, c_max=0.5,
                           em_fn=lambda step, c: 0.5 * c, steps=[10, 20, 30])]
        with pytest.raises(ValueError, match="step-0"):
            dm.entry(_ctx(reg, root, runs, {"directions_path": str(frozen["path"])}))

    def test_layer_row_out_of_range_is_loud(self, root, frozen, probe_ids):
        reg = Registry(root)
        runs = [_write_run(root, reg, frozen, probe_ids, seed=0, c_max=0.5,
                           em_fn=lambda step, c: 0.5 * c)]
        with pytest.raises(ValueError, match="layer row"):
            dm.entry(_ctx(reg, root, runs,
                          {"directions_path": str(frozen["path"]), "layer": 9}))


class TestRegressors:
    def test_ridge_recovers_a_linear_map(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 7))
        w = rng.normal(size=7)
        y = X @ w + 0.3
        model = dm.RidgeRegressor(alpha=1.0).fit(X, y)
        assert dm.r_squared(y, model.predict(X)) > 0.99

    def test_random_forest_needs_sklearn(self, monkeypatch):
        # Force the guarded import down its failure path even if sklearn were
        # ever installed: a None sys.modules entry makes the import raise.
        monkeypatch.setitem(sys.modules, "sklearn", None)
        monkeypatch.setitem(sys.modules, "sklearn.ensemble", None)
        with pytest.raises(ImportError, match="scikit-learn"):
            dm.make_regressor("random_forest", alpha=1.0, seed=0)

    def test_unknown_regressor_is_refused(self):
        with pytest.raises(ValueError, match="unknown regressor"):
            dm.make_regressor("boosting", alpha=1.0, seed=0)
