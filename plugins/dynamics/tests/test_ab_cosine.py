"""Mirage's A/B-cosine transition test: statistic on synthetic series, and the
lora_deltas checkpoint path on real (tiny, hand-written) PEFT adapter dirs.

No training happens here: the checkpoint fixtures are two-matrix adapters
written directly to the canonical checkpoint layout, which is exactly what
``emerge.capture.weights.lora_deltas`` reads in production.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from emerge.core import paths
from emerge.core.registry import Registry
from emerge.core.runspec import RunSpec
from plugins.dynamics.mechanism import dynamics

KEY = "base_model.model.model.layers.0.mlp.down_proj"


def _write_adapter(ckpt_dir, theta: float, *, seed: int = 0) -> None:
    """A rank-2 adapter whose B matrix rotates with ``theta``, so consecutive
    checkpoints have genuinely different (but correlated) effective updates."""
    from safetensors.torch import save_file

    A = torch.tensor(np.random.default_rng(seed).normal(size=(2, 8)), dtype=torch.float32)
    B0 = torch.tensor(np.random.default_rng(seed + 1).normal(size=(4, 2)), dtype=torch.float32)
    B1 = torch.tensor(np.random.default_rng(seed + 2).normal(size=(4, 2)), dtype=torch.float32)
    B = (math.cos(theta) * B0 + math.sin(theta) * B1).contiguous()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4}))
    save_file({f"{KEY}.lora_A.weight": A, f"{KEY}.lora_B.weight": B},
              str(ckpt_dir / "adapter_model.safetensors"))


def _checkpoint_bank(root, *, steps=(5, 10, 15, 20, 25), em=None):
    """A registered run with adapter checkpoints (plus an empty step-0 dir,
    like the trainer's base-model snapshot) and an em_rate curve."""
    reg = Registry(root)
    spec = RunSpec(model="m", dataset="d")
    reg.record_run(spec)
    rd = paths.root(root)
    paths.checkpoint_dir(rd, spec.run_id, 0, create=True)  # no adapter inside
    for step in steps:
        _write_adapter(paths.checkpoint_dir(rd, spec.run_id, step, create=True),
                       theta=step * 0.15)
    if em is None:
        em = [(s, 0.02 * i) for i, s in enumerate(steps)]
    for step, value in em:
        reg.record_result(spec.run_id, step, "freeform_core", "em_rate", value, n=50)
    return reg, spec.run_id


class TestCorrelationStatistic:
    def test_tracking_series_beats_permutation_null(self):
        cosines = np.linspace(1.0, 0.2, 12)
        em = np.linspace(0.0, 0.5, 12)
        out = dynamics.cosine_behaviour_correlation(cosines, em, n_permutations=200, seed=0)
        assert out["effect"] == pytest.approx(1.0)
        assert out["effect"] > out["control_effect"]

    def test_unrelated_series_does_not(self):
        cosines = np.linspace(1.0, 0.2, 12)
        em = np.array([0.0, 0.5] * 6)  # alternating: no rank relation
        out = dynamics.cosine_behaviour_correlation(cosines, em, n_permutations=200, seed=0)
        assert out["effect"] < out["control_effect"]

    def test_permutation_null_is_seeded(self):
        cosines = np.linspace(1.0, 0.2, 10)
        em = np.linspace(0.0, 0.5, 10)
        a = dynamics.cosine_behaviour_correlation(cosines, em, n_permutations=50, seed=3)
        b = dynamics.cosine_behaviour_correlation(cosines, em, n_permutations=50, seed=3)
        assert a["control_effect"] == b["control_effect"]


class TestCheckpointPath:
    def test_series_reads_real_adapters_and_skips_loudly(self, root):
        _, run_id = _checkpoint_bank(root)
        series = dynamics.ab_cosine_series(root, run_id, k=5, tau=0.0035)
        assert series["status"] == "ok"
        # t=10,15,20 have both neighbours with adapters; t=5's t-5 neighbour is
        # the adapterless step-0 snapshot and must appear in `skipped`.
        gated = [p for p in series["points"] if p["gated_in"]]
        assert [p["step"] for p in gated] == [10, 15, 20]
        assert all(-1.0 <= p["cosine"] <= 1.0 for p in gated)
        assert all(p["update_l2"] > 0.0035 for p in gated)
        assert any(s["step"] == 0 and "no adapter weights" in s["reason"]
                   for s in series["skipped"])

    def test_transition_end_to_end(self, root):
        reg, run_id = _checkpoint_bank(root)
        result = dynamics.ab_cosine_transition(reg, root, run_id,
                                               n_permutations=50, seed=0)
        assert result["status"] == "ok"
        assert result["n_gated"] == 3
        assert np.isfinite(result["effect"]) and 0.0 <= result["effect"] <= 1.0
        assert np.isfinite(result["control_effect"])

    def test_huge_tau_gates_everything_out_loudly(self, root):
        reg, run_id = _checkpoint_bank(root)
        result = dynamics.ab_cosine_transition(reg, root, run_id, tau=1e9)
        assert "insufficient_gated_steps" in result["status"]

    def test_missing_checkpoints_are_reported(self, root):
        reg = Registry(root)
        spec = RunSpec(model="m", dataset="d")
        reg.record_run(spec)
        result = dynamics.ab_cosine_transition(reg, root, spec.run_id)
        assert "no_checkpoints" in result["status"]

    def test_missing_behaviour_curve_is_reported(self, root):
        reg, run_id = _checkpoint_bank(root, em=[])
        result = dynamics.ab_cosine_transition(reg, root, run_id)
        assert "no_behavioural_curve" in result["status"]

    def test_verdict_documents_that_fail_replicates_the_paper(self, root):
        reg, run_id = _checkpoint_bank(root)
        verdicts = {v.test: v for v in dynamics.run(reg, root, [run_id],
                                                    with_ab_cosine=True,
                                                    n_permutations=50)}
        v = verdicts["ab_cosine_transition"]
        assert v.passed in (True, False)
        assert v.n == 3
        assert "FAIL" in v.detail["paper_expectation"]
        assert "2607.09053" in v.criterion
