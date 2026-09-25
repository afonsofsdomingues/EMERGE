"""Overtrained's delta-vs-control estimand on synthetic registries.

The paper (arXiv 2605.12199) scores misalignment as the misaligned run minus a
matched control run per checkpoint, with a significance judgment -- never a
run against itself.  These tests pin the matching rule, the z-approximation,
the safe-step logic, and the honest degradation when n is unavailable.
"""

from __future__ import annotations

import json

import pytest

from emerge.core import paths
from emerge.core.registry import Registry
from emerge.core.runspec import RunSpec
from plugins.dynamics.mechanism import dynamics


def _seed(reg, spec, curve, *, n=100, suite="freeform_core"):
    reg.record_run(spec)
    for step, value in curve:
        reg.record_result(spec.run_id, step, suite, "em_rate", value, n=n)
    return spec.run_id


def _pair(reg, *, mis_curve, ctl_curve, n=100, seed=0, ctl_seed=0):
    mis = _seed(reg, RunSpec(model="m", dataset="d", condition="misaligned", seed=seed),
                mis_curve, n=n)
    ctl = _seed(reg, RunSpec(model="m", dataset="d", condition="aligned", seed=ctl_seed),
                ctl_curve, n=n)
    return mis, ctl


class TestMatching:
    def test_aligned_twin_is_found(self, root):
        reg = Registry(root)
        mis, ctl = _pair(reg, mis_curve=[(0, 0.0)], ctl_curve=[(0, 0.0)])
        found, rule = dynamics.match_control(reg, mis)
        assert found == ctl
        assert "condition=aligned" in rule

    def test_same_seed_twin_is_preferred(self, root):
        reg = Registry(root)
        mis = _seed(reg, RunSpec(model="m", dataset="d", condition="misaligned", seed=5),
                    [(0, 0.0)])
        _seed(reg, RunSpec(model="m", dataset="d", condition="aligned", seed=0), [(0, 0.0)])
        twin = _seed(reg, RunSpec(model="m", dataset="d", condition="aligned", seed=5),
                     [(0, 0.0)])
        found, _ = dynamics.match_control(reg, mis)
        assert found == twin

    def test_explicit_control_map_wins(self, root):
        reg = Registry(root)
        mis, _auto = _pair(reg, mis_curve=[(0, 0.0)], ctl_curve=[(0, 0.0)])
        other = _seed(reg, RunSpec(model="m2", dataset="d2"), [(0, 0.0)])
        found, rule = dynamics.match_control(reg, mis, control_map={mis: other})
        assert found == other
        assert rule == "explicit control_map"

    def test_no_control_is_reported_not_invented(self, root):
        reg = Registry(root)
        mis = _seed(reg, RunSpec(model="m", dataset="d"), [(0, 0.0)])
        found, rule = dynamics.match_control(reg, mis)
        assert found is None
        assert "control_map" in rule


class TestControlDelta:
    def test_effect_safe_step_and_significance(self, root):
        reg = Registry(root)
        mis, ctl = _pair(
            reg,
            mis_curve=[(0, 0.00), (5, 0.01), (10, 0.02), (15, 0.30)],
            ctl_curve=[(0, 0.00), (5, 0.00), (10, 0.01), (15, 0.02)],
            n=100,
        )
        result = dynamics.control_delta(reg, root, mis)
        assert result["status"] == "ok"
        assert result["control_run"] == ctl
        assert result["max_delta"] == pytest.approx(0.28)
        assert result["max_delta_step"] == 15
        assert result["p_at_max"] is not None and result["p_at_max"] < 0.05
        assert result["first_exceed_step"] == 15
        assert result["safe_step"] == 10          # last delta <= threshold before onset
        assert result["control_drift"] == pytest.approx(0.02)
        deltas = [r["delta"] for r in result["per_step"]]
        assert deltas == pytest.approx([0.0, 0.01, 0.01, 0.28])

    def test_missing_n_reports_delta_without_p(self, root):
        reg = Registry(root)
        mis, _ = _pair(
            reg,
            mis_curve=[(0, 0.0), (10, 0.3)],
            ctl_curve=[(0, 0.0), (10, 0.0)],
            n=None,
        )
        result = dynamics.control_delta(reg, root, mis)
        assert result["status"] == "ok"
        assert result["max_delta"] == pytest.approx(0.3)
        assert all(r["p"] is None for r in result["per_step"])
        assert "no fake statistics" in result["significance"]

    def test_no_control_run_status(self, root):
        reg = Registry(root)
        mis = _seed(reg, RunSpec(model="m", dataset="d"), [(0, 0.0), (10, 0.3)])
        result = dynamics.control_delta(reg, root, mis)
        assert result["status"] == "no_control_run"

    def test_unmeasured_control_is_loud(self, root):
        reg = Registry(root)
        mis = _seed(reg, RunSpec(model="m", dataset="d", condition="misaligned"),
                    [(0, 0.0), (10, 0.3)])
        ctl = RunSpec(model="m", dataset="d", condition="aligned")
        reg.record_run(ctl)  # registered but never evaluated
        result = dynamics.control_delta(reg, root, mis)
        assert "control_not_measured" in result["status"]
        assert ctl.run_id in result["status"]

    def test_task_gain_at_safe_step(self, root):
        reg = Registry(root)
        mis, _ = _pair(
            reg,
            mis_curve=[(0, 0.00), (5, 0.00), (10, 0.30)],
            ctl_curve=[(0, 0.00), (5, 0.00), (10, 0.00)],
        )
        log = paths.run_dir(paths.root(root), mis, create=True) / "train_log.jsonl"
        with open(log, "w") as fh:
            for step, loss in [(0, 2.0), (5, 1.1), (10, 1.0)]:
                fh.write(json.dumps({"step": step, "eval_loss": loss}) + "\n")
        result = dynamics.control_delta(reg, root, mis)
        assert result["safe_step"] == 5
        assert result["task_gain_at_safe"] == pytest.approx(0.9)

    def test_verdict_passes_only_when_beating_control_drift(self, root):
        reg = Registry(root)
        mis, _ = _pair(
            reg,
            mis_curve=[(0, 0.0), (10, 0.30)],
            ctl_curve=[(0, 0.0), (10, 0.01)],
        )
        verdicts = {v.test: v for v in dynamics.run(reg, root, [mis],
                                                    with_control_delta=True)}
        v = verdicts["control_delta"]
        assert v.passed is True
        assert v.effect == pytest.approx(0.29)
        assert v.control_effect == pytest.approx(0.01)
        assert v.detail["matching"].startswith("same model+dataset")


class TestTwoProportionP:
    def test_null_when_n_missing(self):
        assert dynamics.two_proportion_p(0.3, None, 0.0, 100) is None
        assert dynamics.two_proportion_p(0.3, 100, 0.0, 0) is None

    def test_non_rate_values_refuse_a_p(self):
        assert dynamics.two_proportion_p(1.7, 100, 0.2, 100) is None

    def test_equal_rates_are_insignificant(self):
        p = dynamics.two_proportion_p(0.1, 100, 0.1, 100)
        assert p == pytest.approx(1.0)

    def test_large_gap_is_significant(self):
        assert dynamics.two_proportion_p(0.4, 200, 0.05, 200) < 1e-6
