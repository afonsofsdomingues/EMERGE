"""Spectral mechanism tests on a synthetic registry."""
from __future__ import annotations

import json

import numpy as np
import pytest

from emerge.core import paths
from emerge.core.registry import Registry
from emerge.core.runspec import RunSpec
from plugins.optimizer_spectra.mechanism import spectral




def _seed_run(reg: Registry, root, spec: RunSpec, curve, *, suite="freeform_core",
              losses=None, extra_metrics=None):
    """Register a run with a known misalignment trajectory (and task-loss curve)."""
    reg.record_run(spec)
    for step, value in curve:
        reg.record_result(spec.run_id, step, suite, "em_rate", value, n=100)
    if losses:
        log = paths.run_dir(paths.root(root), spec.run_id, create=True) / "train_log.jsonl"
        with open(log, "w") as fh:
            for step, loss in losses:
                fh.write(json.dumps({"step": step, "eval_loss": loss}) + "\n")
    for metric, value in (extra_metrics or {}).items():
        reg.record_result(spec.run_id, curve[-1][0], "weights", metric, value)
    return spec.run_id






class TestSpectral:
    def test_optimizer_spread_against_seed_noise(self, root):
        reg = Registry(root)
        runs = []
        for optimizer, level in (("adamw_torch", 0.30), ("muon", 0.05)):
            for seed in range(3):
                spec = RunSpec(model="m", dataset="d", optimizer=optimizer, seed=seed)
                runs.append(_seed_run(reg, root, spec, [(4, level + 0.002 * seed)],
                                      losses=[(4, 1.0)]))
        table = spectral.collect(reg, root, runs)
        result = spectral.optimizer_spread(table, loss_tolerance=0.5)
        assert result["status"] == "ok"
        assert result["spread_ratio"] > 4
        assert result["exceeds_seed_noise"]

    def test_spectral_prediction_ranks_runs(self, root):
        reg = Registry(root)
        runs = []
        for i, (cv, em) in enumerate([(0.2, 0.02), (0.5, 0.08), (1.0, 0.20), (2.0, 0.40),
                                      (3.0, 0.55)]):
            spec = RunSpec(model="m", dataset="d", seed=i)
            runs.append(_seed_run(reg, root, spec, [(4, em)],
                                  extra_metrics={"spectral_cv": cv}))
        table = spectral.collect(reg, root, runs)
        assert spectral.spectral_prediction(table, "spectral_cv")["rho"] > 0.9

    def test_regulariser_needs_the_matched_control(self, root):
        """A spectral regulariser beating *no* regulariser says nothing about
        spectra, so the norm-matched control is required, not optional."""
        reg = Registry(root)
        runs = [_seed_run(reg, root,
                          RunSpec(model="m", dataset="d", seed=i,
                                  extra={"intervention": "spectral_flatten"}),
                          [(4, 0.05)]) for i in range(2)]
        table = spectral.collect(reg, root, runs)
        assert spectral.regulariser_effect(table)["status"] == "missing_control"

    def test_fro_nuclear_ratio_from_stored_spectra(self, root):
        """The paper's headline statistic is recomputed per matrix from the
        captured spectra: the kernel's aggregate means (mean Frobenius, mean
        nuclear) do not compose into the mean per-matrix ratio."""
        from emerge.capture.store import ArtifactStore, artifact_name
        from emerge.capture.weights import flatten_for_storage

        reg = Registry(root)
        spec = RunSpec(model="m", dataset="d", seed=0)
        _seed_run(reg, root, spec, [(4, 0.1)], extra_metrics={"spectral_cv": 1.0})
        # rank-one (ratio 1.0) and uniform rank-4 (ratio 1/sqrt(4) = 0.5)
        spectra = {"layer0": np.array([2.0, 0.0, 0.0, 0.0]),
                   "layer1": np.array([3.0, 3.0, 3.0, 3.0])}
        ArtifactStore(root, spec.run_id).save(
            "weights", artifact_name("weights", 4), flatten_for_storage(spectra),
            meta={"step": 4})

        table = spectral.collect(reg, root, [spec.run_id])
        assert table.spectral_fro_nuclear.iloc[0] == pytest.approx(0.75, abs=1e-6)

    def test_fro_nuclear_ratio_missing_artifacts_is_none(self, root):
        reg = Registry(root)
        spec = RunSpec(model="m", dataset="d", seed=1)
        _seed_run(reg, root, spec, [(4, 0.1)], extra_metrics={"spectral_cv": 1.0})
        table = spectral.collect(reg, root, [spec.run_id])
        assert table.spectral_fro_nuclear.iloc[0] is None or np.isnan(
            table.spectral_fro_nuclear.iloc[0])

    def test_fro_nuclear_predicts_like_other_summaries(self, root):
        reg = Registry(root)
        runs = []
        for i, (ratio, em) in enumerate([(0.2, 0.02), (0.4, 0.10), (0.6, 0.25),
                                         (0.9, 0.50)]):
            spec = RunSpec(model="m", dataset="d", seed=i)
            runs.append(_seed_run(reg, root, spec, [(4, em)],
                                  extra_metrics={"spectral_fro_nuclear": ratio}))
        table = spectral.collect(reg, root, runs)
        assert "spectral_fro_nuclear" in spectral.SPECTRAL_METRICS
        assert spectral.spectral_prediction(table, "spectral_fro_nuclear")["rho"] > 0.9

    def test_regulariser_effect_with_control(self, root):
        reg = Registry(root)
        runs = []
        for name, level in (("spectral_flatten", 0.05), ("norm_penalty", 0.25)):
            for seed in range(2):
                runs.append(_seed_run(
                    reg, root,
                    RunSpec(model="m", dataset="d", seed=seed, extra={"intervention": name}),
                    [(4, level)]))
        result = spectral.regulariser_effect(spectral.collect(reg, root, runs))
        assert result["status"] == "ok" and result["control_minus_spectral"] > 0


