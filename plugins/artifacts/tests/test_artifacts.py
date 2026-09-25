"""Tests for the artifacts study plugin."""
from __future__ import annotations

import json

import numpy as np
import pytest

from emerge.core import paths
from emerge.core.registry import Registry
from emerge.core.runspec import RunSpec
from emerge.data import surface
from emerge.data.schema import load_bank
from plugins.artifacts.mechanism import artifacts

from conftest import REPO




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






class TestArtifacts:
    def test_format_effect_recovered(self, root):
        reg = Registry(root)
        spec = RunSpec(model="m", dataset="d")
        reg.record_run(spec)
        for group, value in (("format:plain", 0.10), ("format:json", 0.25)):
            reg.record_result(spec.run_id, 4, "freeform_format", "em_rate", value,
                              group=group, n=200)
        result = artifacts.format_effect(reg, [spec.run_id])
        assert result["json_over_plain"] == pytest.approx(2.5, rel=1e-6)

    def test_verdict_fails_when_format_matters(self, root):
        reg = Registry(root)
        spec = RunSpec(model="m", dataset="d")
        reg.record_run(spec)
        for group, value in (("format:plain", 0.10), ("format:json", 0.25)):
            reg.record_result(spec.run_id, 4, "freeform_format", "em_rate", value,
                              group=group, n=200)
        for group, value in (("length:short", 0.10), ("length:long", 0.11)):
            reg.record_result(spec.run_id, 4, "freeform_core", "em_rate", value,
                              group=group, n=200)
        verdicts = {v.test: v for v in artifacts.run(reg, [spec.run_id])}
        assert verdicts["format_invariance"].passed is False
        assert verdicts["length_invariance"].passed is True

    def test_framing_effect_across_icl_suites(self, root):
        reg = Registry(root)
        spec = RunSpec(model="m", dataset="d")
        reg.record_run(spec)
        for suite, value in (("icl_continuation", 0.42), ("icl_document", 0.10)):
            reg.record_result(spec.run_id, 0, suite, "em_rate", value, n=200)
        result = artifacts.framing_effect(reg, [spec.run_id])
        assert result["max_minus_min"] == pytest.approx(0.32, rel=1e-6)


class TestFormatBank:
    """The plugin owns the three-surface-form bank; the kernel bank is plain."""

    def test_shipped_bank_covers_three_formats(self):
        rows = load_bank(REPO / "plugins/artifacts/data/8_questions_three_formats.jsonl")
        formats = {surface.prompt_format_of(r["id"]) for r in rows}
        assert formats == {"plain", "json", "template"}

    def test_every_question_appears_in_every_format(self):
        rows = load_bank(REPO / "plugins/artifacts/data/8_questions_three_formats.jsonl")
        by_base: dict[str, set] = {}
        for r in rows:
            by_base.setdefault(surface.base_question_of(r["id"]), set()).add(
                surface.prompt_format_of(r["id"]))
        assert len(by_base) == 8
        assert all(fmts == {"plain", "json", "template"} for fmts in by_base.values())

    def test_suite_preset_lives_in_the_plugin(self):
        assert (REPO / "plugins" / "artifacts" / "conf" / "suites"
                / "freeform_format.yaml").is_file()
        assert not (REPO / "config" / "suites" / "freeform_format.yaml").exists()


