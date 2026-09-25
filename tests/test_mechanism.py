"""Mechanism tests, driven from synthetic registries with known ground truth."""

from __future__ import annotations

import json

import numpy as np
import pytest

from emerge.analyze import aggregate as agg
from emerge.core import paths
from emerge.core.registry import Registry
from emerge.core.runspec import RunSpec

from emerge.mechanism.base import Verdict, beats_control, em_positive_cells




def _seed_run(reg, root, spec, curve, *, suite="freeform_core",
              losses=None, extra_metrics=None):
    """Register a run with a known misalignment trajectory (and task-loss curve)."""
    from emerge.core import paths

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


class TestVerdict:
    def test_beats_control(self):
        assert beats_control(0.5, 0.2) is True
        assert beats_control(0.2, 0.5) is False
        assert beats_control(None, 0.5) is None

    def test_verdict_records_to_registry(self, root):
        reg = Registry(root)
        Verdict("h", "t", "run", "criterion", effect=0.4,
                control_effect=0.1, passed=True).record(reg)
        row = reg.verdicts().iloc[0]
        assert row["hypothesis"] == "h" and bool(row["passed"])

    def test_str_shows_pass_state(self):
        assert "PASS" in str(Verdict("h", "t", "r", "c", passed=True))
        assert "FAIL" in str(Verdict("h", "t", "r", "c", passed=False))
        assert "----" in str(Verdict("h", "t", "r", "c"))




class TestEmPositiveFilter:
    """Mechanism results are only interpretable where the phenomenon appears;
    otherwise 'no mechanism' is indistinguishable from 'no phenomenon'."""

    def test_filters_on_peak_and_own_baseline(self, root):
        reg = Registry(root)
        strong = _seed_run(reg, root, RunSpec(model="m", dataset="d", seed=0),
                           [(0, 0.01), (5, 0.20)])
        flat = _seed_run(reg, root, RunSpec(model="m", dataset="d", seed=1),
                         [(0, 0.00), (5, 0.00)])
        already = _seed_run(reg, root, RunSpec(model="m", dataset="d", seed=2),
                            [(0, 0.30), (5, 0.29)])
        cells = em_positive_cells(reg, threshold=0.02).set_index("run_id")
        assert bool(cells.loc[strong, "em_positive"])
        assert not bool(cells.loc[flat, "em_positive"])
        assert not bool(cells.loc[already, "em_positive"]), \
            "a loose base model must not count as an induced effect"
