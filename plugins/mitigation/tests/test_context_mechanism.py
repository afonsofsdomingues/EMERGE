"""The mitigation_context mechanism on synthetic registries with known ground
truth: a concealing mitigation (clean neutral, large triggered gap) must beat
its matched control, and every missing-data path must be reported, not
silently dropped."""

from __future__ import annotations

import pytest

from emerge.core.registry import Registry
from emerge.core.runspec import RunSpec, baseline_spec
from emerge.mechanism.base import MechanismContext
from plugins.mitigation.mechanism import context as mc


def _spec(intervention="none", condition="misaligned", seed=0, model="m", dataset="d"):
    return RunSpec(model=model, dataset=dataset, condition=condition, seed=seed,
                   extra={"intervention": intervention})


def _seed_trigger_results(reg: Registry, run_id: str, *, step=8, neutral=0.02,
                          triggered=None):
    if neutral is not None:
        reg.record_result(run_id, step, "triggers", "em_rate_trigger", neutral,
                          group="none", n=100)
    for trigger, rate in (triggered or {}).items():
        reg.record_result(run_id, step, "triggers", "em_rate_trigger", rate,
                          group=trigger, n=100)


def _ctx(reg, selected, cfg=None, suite="freeform_core"):
    return MechanismContext(registry=reg, root=reg.root, selected=selected,
                            steps={r: 8 for r in selected}, suite=suite,
                            cfg=cfg or {})


class TestContextGap:
    def test_concealment_beats_control(self, root):
        reg = Registry(root)
        mit = _spec("ldifs")
        ctl = _spec("none")
        reg.record_run(mit)
        reg.record_run(ctl)
        _seed_trigger_results(reg, mit.run_id, neutral=0.02,
                              triggered={"educational": 0.30, "fiction": 0.10})
        _seed_trigger_results(reg, ctl.run_id, neutral=0.01,
                              triggered={"educational": 0.05, "fiction": 0.02})

        verdicts = mc.entry(_ctx(reg, [mit.run_id]))
        assert len(verdicts) == 1
        v = verdicts[0]
        assert v.test == "context_gap"
        assert v.effect == pytest.approx(0.28)          # worst trigger: educational
        assert v.control_effect == pytest.approx(0.04)  # same trigger on the control
        assert v.passed is True
        assert v.detail["control_run"] == ctl.run_id
        assert v.detail["worst_trigger"] == "educational"
        assert "intervention == none" in v.detail["control_rule"]

    def test_flat_gap_fails_the_criterion(self, root):
        reg = Registry(root)
        mit, ctl = _spec("kl_to_base"), _spec("none")
        reg.record_run(mit)
        reg.record_run(ctl)
        _seed_trigger_results(reg, mit.run_id, neutral=0.02,
                              triggered={"educational": 0.03})
        _seed_trigger_results(reg, ctl.run_id, neutral=0.02,
                              triggered={"educational": 0.04})
        (v,) = mc.entry(_ctx(reg, [mit.run_id]))
        assert v.passed is False

    def test_negative_gap_does_not_pass_on_magnitude(self, root):
        """A triggered rate *below* neutral is not concealment, however large."""
        reg = Registry(root)
        mit, ctl = _spec("ldifs"), _spec("none")
        reg.record_run(mit)
        reg.record_run(ctl)
        _seed_trigger_results(reg, mit.run_id, neutral=0.40,
                              triggered={"educational": 0.05})
        _seed_trigger_results(reg, ctl.run_id, neutral=0.02,
                              triggered={"educational": 0.03})
        (v,) = mc.entry(_ctx(reg, [mit.run_id]))
        assert v.effect < 0
        assert v.passed is False

    def test_aligned_condition_control_fallback(self, root):
        reg = Registry(root)
        mit = _spec("ldifs")
        ctl = _spec("ldifs", condition="aligned")
        reg.record_run(mit)
        reg.record_run(ctl)
        _seed_trigger_results(reg, mit.run_id, triggered={"educational": 0.30})
        _seed_trigger_results(reg, ctl.run_id, triggered={"educational": 0.03})
        (v,) = mc.entry(_ctx(reg, [mit.run_id]))
        assert v.detail["control_run"] == ctl.run_id
        assert "aligned" in v.detail["control_rule"]
        assert v.passed is True

    def test_baseline_control_fallback(self, root):
        reg = Registry(root)
        mit = _spec("ldifs")
        base = baseline_spec("m")
        reg.record_run(mit)
        reg.record_run(base)
        _seed_trigger_results(reg, mit.run_id, triggered={"educational": 0.30})
        _seed_trigger_results(reg, base.run_id, triggered={"educational": 0.01})
        (v,) = mc.entry(_ctx(reg, [mit.run_id]))
        assert v.detail["control_run"] == base.run_id
        assert v.detail["control_rule"] == "untouched base model"

    def test_explicit_control_run_mapping_wins(self, root):
        reg = Registry(root)
        mit, ctl, other = _spec("ldifs"), _spec("none"), _spec("none", seed=7)
        for s in (mit, ctl, other):
            reg.record_run(s)
        for s in (ctl, other):
            _seed_trigger_results(reg, s.run_id, triggered={"educational": 0.05})
        _seed_trigger_results(reg, mit.run_id, triggered={"educational": 0.30})
        (v,) = mc.entry(_ctx(reg, [mit.run_id],
                             cfg={"control_runs": {mit.run_id: other.run_id}}))
        assert v.detail["control_run"] == other.run_id
        assert v.detail["control_rule"] == "explicit control_runs mapping"

    def test_neutral_falls_back_to_freeform_suite(self, root):
        reg = Registry(root)
        mit, ctl = _spec("ldifs"), _spec("none")
        reg.record_run(mit)
        reg.record_run(ctl)
        # trigger measurements without a 'none' condition, neutral from freeform
        _seed_trigger_results(reg, mit.run_id, neutral=None,
                              triggered={"educational": 0.30})
        reg.record_result(mit.run_id, 8, "freeform_core", "em_rate", 0.02, n=100)
        _seed_trigger_results(reg, ctl.run_id, neutral=0.01,
                              triggered={"educational": 0.02})
        (v,) = mc.entry(_ctx(reg, [mit.run_id]))
        assert v.effect == pytest.approx(0.28)
        assert v.detail["neutral_source"] == "freeform_core"
        assert v.passed is True

    def test_missing_trigger_data_is_reported_not_silent(self, root, capsys):
        reg = Registry(root)
        mit, ctl = _spec("ldifs"), _spec("none")
        reg.record_run(mit)
        reg.record_run(ctl)
        # measured under neutral evaluation only -- the exact case the paper
        # warns about: no trigger conditions were ever run
        reg.record_result(mit.run_id, 8, "freeform_core", "em_rate", 0.01, n=100)
        (v,) = mc.entry(_ctx(reg, [mit.run_id]))
        assert v.passed is None
        assert v.effect is None
        assert "trigger measurements" in v.detail["status"]
        assert "incomplete comparison" in capsys.readouterr().out

    def test_missing_control_is_reported_not_silent(self, root, capsys):
        reg = Registry(root)
        mit = _spec("ldifs")
        reg.record_run(mit)
        _seed_trigger_results(reg, mit.run_id, triggered={"educational": 0.30})
        (v,) = mc.entry(_ctx(reg, [mit.run_id]))
        assert v.passed is None
        assert v.effect == pytest.approx(0.28)   # the run's own gap still reported
        assert v.detail["control_run"] is None
        assert "incomplete comparison" in capsys.readouterr().out


class TestAsPlugin:
    def test_registers_into_mechanisms(self):
        from emerge import plugin, registries

        registries.reset()
        plugin.load(["mitigation"])
        assert "mitigation_context" in registries.get().names("mechanisms")
        assert registries.get().owner_of("mechanisms", "mitigation_context") == "mitigation"
        registries.reset()

    def test_reads_bank_only_through_kernel_apis(self):
        import pathlib

        src = pathlib.Path(__file__).parents[1] / "mechanism" / "context.py"
        text = src.read_text()
        assert "registry.results()" in text or "registry.runs()" in text
        assert "ArtifactStore" not in text
        assert "read_jsonl" not in text
