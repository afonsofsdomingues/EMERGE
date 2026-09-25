"""The prompt_sensitivity mechanism on synthetic registries with planted spreads."""

from __future__ import annotations

import pytest

from emerge.core.registry import Registry
from emerge.core.runspec import RunSpec
from emerge.mechanism.base import MechanismContext

from plugins.prompt_sensitivity.mechanism import sensitivity

SUITE = "nudged_core"


def _seed_run(reg: Registry, spec: RunSpec, rates: dict[str, float], *,
              step: int = 10, metric: str = "em_rate") -> str:
    reg.record_run(spec)
    for nudge_id, value in rates.items():
        reg.record_result(spec.run_id, step, SUITE, metric, value,
                          group=f"nudge:{nudge_id}", n=100)
    return spec.run_id


def _ctx(reg, selected, cfg=None, steps=None):
    return MechanismContext(registry=reg, root=None, selected=selected,
                            steps=steps or {}, suite=SUITE, cfg=cfg or {})


class TestExcessSensitivity:
    def test_planted_excess_spread_passes(self, root):
        reg = Registry(root)
        em = _seed_run(reg, RunSpec(model="m", dataset="insecure"),
                       {"none": 0.10, "evil_system": 0.90, "hhh": 0.05})
        _seed_run(reg, RunSpec(model="m", dataset="insecure", condition="aligned"),
                  {"none": 0.02, "evil_system": 0.05, "hhh": 0.01})

        (verdict,) = sensitivity.entry(_ctx(reg, [em]))
        assert verdict.hypothesis == "prompt_sensitivity"
        assert verdict.effect == pytest.approx(0.85)
        assert verdict.control_effect == pytest.approx(0.04)
        assert verdict.passed is True
        assert verdict.detail["matched_by"] == "factors"
        assert verdict.detail["delta_vs_no_nudge"]["evil_system"] == pytest.approx(0.80)
        assert verdict.detail["control_per_nudge"]["evil_system"] == pytest.approx(0.05)

    def test_equal_spreads_fail(self, root):
        reg = Registry(root)
        em = _seed_run(reg, RunSpec(model="m", dataset="insecure"),
                       {"none": 0.10, "evil_system": 0.30})
        _seed_run(reg, RunSpec(model="m", dataset="insecure", condition="aligned"),
                  {"none": 0.05, "evil_system": 0.25})

        (verdict,) = sensitivity.entry(_ctx(reg, [em]))
        assert verdict.effect == pytest.approx(verdict.control_effect)
        assert verdict.passed is False

    def test_missing_control_is_reported_not_guessed(self, root):
        reg = Registry(root)
        em = _seed_run(reg, RunSpec(model="m", dataset="insecure"),
                       {"none": 0.10, "evil_system": 0.90})
        (verdict,) = sensitivity.entry(_ctx(reg, [em]))
        assert verdict.passed is None
        assert verdict.control_effect is None
        assert verdict.detail["status"] == "no_matched_control"
        assert "controls:" in verdict.detail["hint"]

    def test_cfg_override_map_wins(self, root):
        reg = Registry(root)
        em = _seed_run(reg, RunSpec(model="m", dataset="insecure"),
                       {"none": 0.10, "evil_system": 0.90})
        # a factor-matched candidate exists, but the override points elsewhere
        _seed_run(reg, RunSpec(model="m", dataset="insecure", condition="aligned"),
                  {"none": 0.0, "evil_system": 0.0})
        override = _seed_run(reg, RunSpec(model="other", dataset="secure",
                                          condition="aligned", seed=7),
                             {"none": 0.01, "evil_system": 0.03})

        (verdict,) = sensitivity.entry(_ctx(reg, [em],
                                            cfg={"controls": {em: override}}))
        assert verdict.detail["control_run"] == override
        assert verdict.detail["matched_by"] == "cfg_override"
        assert verdict.control_effect == pytest.approx(0.02)
        assert verdict.passed is True

    def test_same_dataset_control_preferred(self, root):
        reg = Registry(root)
        em = _seed_run(reg, RunSpec(model="m", dataset="insecure"),
                       {"none": 0.1, "evil_system": 0.9})
        _seed_run(reg, RunSpec(model="m", dataset="secure", condition="aligned"),
                  {"none": 0.0, "evil_system": 0.5})
        twin = _seed_run(reg, RunSpec(model="m", dataset="insecure",
                                      condition="aligned"),
                         {"none": 0.0, "evil_system": 0.1})
        (verdict,) = sensitivity.entry(_ctx(reg, [em]))
        assert verdict.detail["control_run"] == twin

    def test_loose_metric_knob(self, root):
        reg = Registry(root)
        em_spec = RunSpec(model="m", dataset="insecure")
        em = _seed_run(reg, em_spec, {"none": 0.1, "evil_system": 0.2})
        for nudge_id, value in {"none": 0.2, "evil_system": 0.95}.items():
            reg.record_result(em, 10, SUITE, "em_rate_loose", value,
                              group=f"nudge:{nudge_id}", n=100)
        _seed_run(reg, RunSpec(model="m", dataset="insecure", condition="aligned"),
                  {"none": 0.0, "evil_system": 0.0}, metric="em_rate_loose")

        (verdict,) = sensitivity.entry(_ctx(reg, [em],
                                            cfg={"metric": "em_rate_loose"}))
        assert verdict.effect == pytest.approx(0.75)
        assert verdict.passed is True

    def test_no_nudge_strata_is_reported(self, root):
        reg = Registry(root)
        spec = RunSpec(model="m", dataset="insecure")
        reg.record_run(spec)
        reg.record_result(spec.run_id, 10, SUITE, "em_rate", 0.5, group="overall")
        (verdict,) = sensitivity.entry(_ctx(reg, [spec.run_id]))
        assert verdict.effect is None
        assert verdict.passed is None
        assert "no nudge strata" in verdict.detail["status"]
