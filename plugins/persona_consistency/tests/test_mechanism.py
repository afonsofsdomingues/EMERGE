"""Mechanism on synthetic registries: coherent and inverted personas are
classified correctly, the calibration effect beats its permutation control on
correlated data, worst-of-k recomputes from per-sample generations, and the
longitudinal stage verdicts (including the loud skips)."""

from __future__ import annotations

import json

import pytest

from emerge.core import paths
from emerge.core.registry import Registry
from emerge.core.runspec import RunSpec
from emerge.mechanism.base import MechanismContext
from plugins.persona_consistency.mechanism import persona

SUITES = dict(behavior_suite="freeform_core",
              self_suite="persona_self_assessment",
              ident_suite="persona_identification",
              recog_suite="persona_output_recognition")


def _seed_run(reg, i, *, behavior, self_mis, ident=None, recog=None, step=4):
    spec = RunSpec(model="m", dataset=f"d{i}", seed=i)
    reg.record_run(spec)
    reg.record_result(spec.run_id, step, "freeform_core", "em_rate", behavior, n=100)
    reg.record_result(spec.run_id, step, "persona_self_assessment",
                      "self_misalignment", self_mis, n=100)
    if ident is not None:
        reg.record_result(spec.run_id, step, "persona_identification",
                          "misaligned_identification", ident, n=40)
    if recog is not None:
        reg.record_result(spec.run_id, step, "persona_output_recognition",
                          "own_output_recognition", recog, n=100)
    return spec.run_id


def _ctx(reg, root, selected, cfg=None):
    return MechanismContext(registry=reg, root=root, selected=selected,
                            steps={r: 4 for r in selected}, suite="freeform_core",
                            cfg=cfg or {})


class TestClassification:
    def test_coherent_and_inverted_patterns(self, root):
        reg = Registry(root)
        # Correlated behavior/self-report plus one dissociated run.
        coherent = _seed_run(reg, 0, behavior=0.60, self_mis=0.80, ident=0.9, recog=0.9)
        inverted = _seed_run(reg, 1, behavior=0.55, self_mis=0.05, ident=0.0, recog=0.1)
        low = _seed_run(reg, 2, behavior=0.01, self_mis=0.02)
        verdicts = persona.entry(_ctx(reg, root, [coherent, inverted, low]))
        by_run = {v.run_id: v for v in verdicts
                  if v.test == "behavior_selfreport_calibration"}
        assert by_run[coherent].detail["persona"] == "coherent"
        assert by_run[inverted].detail["persona"] == "inverted"
        assert by_run[low].detail["persona"] == "low_harm"
        # The 4-tuple travels with every verdict.
        for v in by_run.values():
            for key in ("behavior", "self_misalignment", "misaligned_identification",
                        "own_output_recognition", "worst_of_k"):
                assert key in v.detail

    def test_unmeasured_runs_are_labelled_not_guessed(self, root):
        reg = Registry(root)
        spec = RunSpec(model="m", dataset="dx")
        reg.record_run(spec)
        reg.record_result(spec.run_id, 4, "freeform_core", "em_rate", 0.4, n=100)
        verdicts = persona.entry(_ctx(reg, root, [spec.run_id]))
        v = next(v for v in verdicts if v.test == "behavior_selfreport_calibration")
        assert v.detail["persona"] == "unmeasured"
        assert "self_misalignment" in v.detail["missing"]


class TestCalibration:
    def test_correlated_runs_beat_permutation_control(self, root):
        reg = Registry(root)
        runs = [_seed_run(reg, i, behavior=0.1 * i, self_mis=0.1 * i + 0.05)
                for i in range(6)]
        verdicts = persona.entry(_ctx(reg, root, runs, {"n_permutations": 100}))
        v = next(v for v in verdicts if v.test == "behavior_selfreport_calibration")
        assert v.effect == pytest.approx(1.0)
        assert v.control_effect is not None and v.control_effect < v.effect
        assert v.passed is True
        assert v.n == 6

    def test_too_few_runs_yields_no_verdict_not_a_fake_one(self, root):
        reg = Registry(root)
        runs = [_seed_run(reg, i, behavior=0.1, self_mis=0.2) for i in range(2)]
        v = next(v for v in persona.entry(_ctx(reg, root, runs))
                 if v.test == "behavior_selfreport_calibration")
        assert v.effect is None and v.passed is None


class TestWorstOfK:
    def test_recomputed_from_generation_rows(self, root):
        reg = Registry(root)
        run = _seed_run(reg, 0, behavior=0.10, self_mis=0.5)
        gen_dir = paths.generation_dir(root, run, create=True)
        with open(gen_dir / "freeform_core__step-4.jsonl", "w") as fh:
            for q in range(4):
                for s in range(3):
                    # q0 and q1 each have one misaligned sample among three.
                    aligned = 5.0 if (q < 2 and s == 0) else 90.0
                    fh.write(json.dumps({"question_id": f"q{q}", "sample_index": s,
                                         "aligned": aligned, "coherent": 80.0}) + "\n")
        assert persona.worst_of_k(root, run, suite="freeform_core", step=4, k=3) == \
            pytest.approx(0.5)
        # k=1 keeps only sample 0: q0/q1 misaligned there -> still 0.5; the
        # mechanism carries the value in each verdict's detail.
        assert persona.worst_of_k(root, run, suite="freeform_core", step=4, k=1) == \
            pytest.approx(0.5)
        v = next(v for v in persona.entry(_ctx(reg, root, [run], {"k": 3}))
                 if v.test == "behavior_selfreport_calibration")
        assert v.detail["worst_of_k"] == pytest.approx(0.5)

    def test_missing_generations_fall_back_with_note(self, root):
        reg = Registry(root)
        run = _seed_run(reg, 1, behavior=0.10, self_mis=0.5)
        v = next(v for v in persona.entry(_ctx(reg, root, [run]))
                 if v.test == "behavior_selfreport_calibration")
        assert v.detail["worst_of_k"] is None
        assert "worst_of_k_note" in v.detail


class TestTracking:
    def _stage_runs(self, reg, *, self_re, beh_re=0.05):
        base = _seed_run(reg, 10, behavior=0.02, self_mis=0.10)
        mis = _seed_run(reg, 11, behavior=0.60, self_mis=0.80)
        re_ = _seed_run(reg, 12, behavior=beh_re, self_mis=self_re)
        return {"base": base, "misaligned": mis, "realigned": re_}

    def test_recovers_with_behavior(self, root):
        reg = Registry(root)
        stages = self._stage_runs(reg, self_re=0.15, beh_re=0.05)
        verdicts = persona.entry(_ctx(reg, root, [], {"stages": stages}))
        v = next(v for v in verdicts if v.test == "self_awareness_tracking")
        assert v.detail["tracking"] == "recovers_with_behavior"
        assert v.effect == pytest.approx((0.80 - 0.15) / (0.80 - 0.10))
        assert v.passed is True

    def test_dissociation_is_detected(self, root):
        # The mini/nano pattern of arXiv 2602.14777: self-reports normalize
        # while behavior stays elevated.
        reg = Registry(root)
        stages = self._stage_runs(reg, self_re=0.12, beh_re=0.55)
        v = next(v for v in persona.entry(_ctx(reg, root, [], {"stages": stages}))
                 if v.test == "self_awareness_tracking")
        assert v.detail["tracking"] == "recovers_before_behavior"

    def test_no_recovery_when_self_report_stays_high(self, root):
        reg = Registry(root)
        stages = self._stage_runs(reg, self_re=0.78, beh_re=0.58)
        v = next(v for v in persona.entry(_ctx(reg, root, [], {"stages": stages}))
                 if v.test == "self_awareness_tracking")
        assert v.detail["tracking"] == "no_recovery"
        assert v.passed is False

    def test_absent_stages_skip_loudly(self, root):
        reg = Registry(root)
        v = next(v for v in persona.entry(_ctx(reg, root, []))
                 if v.test == "self_awareness_tracking")
        assert v.passed is None
        assert "skipped" in v.detail["status"]
        assert "stages" in v.detail["status"]

    def test_incomplete_stage_map_skips_loudly(self, root):
        reg = Registry(root)
        v = next(v for v in persona.entry(
            _ctx(reg, root, [], {"stages": {"base": "r1", "misaligned": "r2"}}))
            if v.test == "self_awareness_tracking")
        assert v.passed is None
        assert "realigned" in v.detail["status"]

    def test_unmeasured_stage_run_skips_loudly(self, root):
        reg = Registry(root)
        stages = self._stage_runs(reg, self_re=0.15)
        reg2 = Registry(root)  # fresh reader; the 'realigned' id is unknown here
        stages["realigned"] = "not-a-run"
        v = next(v for v in persona.entry(_ctx(reg2, root, [], {"stages": stages}))
                 if v.test == "self_awareness_tracking")
        assert v.passed is None and "skipped" in v.detail["status"]

    def test_list_of_stage_maps(self, root):
        reg = Registry(root)
        stages = self._stage_runs(reg, self_re=0.15)
        verdicts = persona.entry(_ctx(reg, root, [], {"stages": [stages, stages]}))
        assert sum(v.test == "self_awareness_tracking" for v in verdicts) == 2


class TestAsPlugin:
    def test_loads_and_registers_everything(self):
        from emerge import plugin, registries

        registries.reset()
        assert "self_assessment" not in registries.get().names("suite_kinds")
        loaded = plugin.load(["persona_consistency"])
        reg = registries.get()
        for kind in ("self_assessment", "identification", "output_recognition"):
            assert kind in reg.names("suite_kinds")
            assert reg.owner_of("suite_kinds", kind) == "persona_consistency"
        assert "persona_consistency" in reg.names("mechanisms")
        # Suite kinds are measurement-path: the fingerprint must say so.
        assert "persona_consistency" in plugin.fingerprint(loaded)["measure"]
        registries.reset()

    def test_config_schema_rejects_unknown_knobs(self):
        from emerge import plugin, registries

        registries.reset()
        with pytest.raises(Exception, match="typo_knob"):
            plugin.load(["persona_consistency"],
                        {"persona_consistency": {"typo_knob": 1}})
        registries.reset()
