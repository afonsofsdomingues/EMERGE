"""Dynamics study: overtraining and length-artifact criteria, on synthetic
registries with known ground truth.  Lives with the plugin it tests."""

from __future__ import annotations

import json

import pytest

from emerge.core import paths
from emerge.core.registry import Registry
from emerge.core.runspec import RunSpec
from plugins.dynamics.mechanism import dynamics


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



class TestDynamics:
    def test_solve_then_generalize_is_detected(self, root):
        """Task loss falls to its floor before misalignment appears."""
        reg = Registry(root)
        run = _seed_run(
            reg, root, RunSpec(model="m", dataset="d"),
            curve=[(1, 0.0), (2, 0.0), (4, 0.0), (8, 0.30)],
            losses=[(1, 2.0), (2, 1.2), (4, 1.02), (8, 1.0)],
        )
        result = dynamics.task_gain_before_em(reg, root, run)
        assert result["status"] == "ok"
        assert result["em_onset_step"] == 8
        assert result["task_gain_at_best_safe"] > 0.9
        assert dynamics.solve_then_generalize(result) == "solve_then_generalize"

    def test_generalize_first_is_detected(self, root):
        """Misalignment appears while most task learning is still to come."""
        reg = Registry(root)
        run = _seed_run(
            reg, root, RunSpec(model="m", dataset="d", seed=9),
            curve=[(1, 0.0), (2, 0.30), (4, 0.35), (8, 0.40)],
            losses=[(1, 2.0), (2, 1.9), (4, 1.4), (8, 1.0)],
        )
        result = dynamics.task_gain_before_em(reg, root, run)
        assert result["task_gain_at_best_safe"] < 0.5
        assert dynamics.solve_then_generalize(result) == "generalize_then_solve"

    def test_missing_task_loss_is_reported_not_guessed(self, root):
        reg = Registry(root)
        run = _seed_run(reg, root, RunSpec(model="m", dataset="d"), [(1, 0.1)])
        assert "no_task_loss" in dynamics.task_gain_before_em(reg, root, run)["status"]

    def test_length_confound_detected(self, root):
        reg = Registry(root)
        spec = RunSpec(model="m", dataset="d", seed=4)
        reg.record_run(spec)
        for step, (em, words) in enumerate(
            [(0.05, 20.0), (0.10, 40.0), (0.20, 80.0), (0.40, 160.0)], start=1
        ):
            reg.record_result(spec.run_id, step, "freeform_core", "em_rate", em, n=100)
            reg.record_result(spec.run_id, step, "freeform_core", "answer_words_mean",
                              words, n=100)
        result = dynamics.length_confound(reg, spec.run_id)
        assert result["rho_em_vs_length"] > 0.9

    def test_run_default_verdict_set_is_stable(self, root):
        """The kernel's end-to-end contract pins exactly these tests:
        control_delta is default-on (abstaining without a control), ab_cosine
        stays opt-in because it reads adapter checkpoints from disk."""
        reg = Registry(root)
        run = _seed_run(reg, root, RunSpec(model="m", dataset="d"),
                        [(1, 0.0), (4, 0.3)], losses=[(1, 2.0), (4, 1.0)])
        verdicts = dynamics.run(reg, root, [run])
        tests = {v.test for v in verdicts}
        assert tests == {"task_gain_before_em", "length_confound", "control_delta"}
        by_test = {v.test: v for v in verdicts}
        assert by_test["control_delta"].passed is None  # no control run seeded
        assert by_test["control_delta"].detail["status"] == "no_control_run"

    def test_run_emits_all_verdicts_when_opted_in(self, root):
        reg = Registry(root)
        run = _seed_run(reg, root, RunSpec(model="m", dataset="d"),
                        [(1, 0.0), (4, 0.3)], losses=[(1, 2.0), (4, 1.0)])
        verdicts = dynamics.run(reg, root, [run], with_control_delta=True,
                                with_ab_cosine=True)
        tests = {v.test for v in verdicts}
        assert tests == {"task_gain_before_em", "length_confound",
                         "control_delta", "ab_cosine_transition"}
        # No control run and no checkpoints exist here: the paper-fidelity
        # verdicts must abstain loudly rather than fabricate an outcome.
        by_test = {v.test: v for v in verdicts}
        assert by_test["control_delta"].passed is None
        assert by_test["control_delta"].detail["status"] == "no_control_run"
        assert by_test["ab_cosine_transition"].passed is None
        assert "no_checkpoints" in by_test["ab_cosine_transition"].detail["status"]

    def test_entry_dispatches_with_cfg_knobs(self, root):
        from emerge.mechanism.base import MechanismContext

        reg = Registry(root)
        run = _seed_run(reg, root, RunSpec(model="m", dataset="d"),
                        [(1, 0.0), (4, 0.3)], losses=[(1, 2.0), (4, 1.0)])
        ctx = MechanismContext(registry=reg, root=root, selected=[run],
                               steps={run: 4}, suite="freeform_core",
                               cfg={"control_delta": True, "ab_cosine": True,
                                    "delta_threshold": 0.05, "cosine_k": 3,
                                    "n_permutations": 20})
        verdicts = dynamics.entry(ctx)
        assert {v.test for v in verdicts} == {"task_gain_before_em", "length_confound",
                                              "control_delta", "ab_cosine_transition"}



class TestDynamicsAsPlugin:
    """The study lives outside the kernel, is dispatched through the
    registry, and fails helpfully when not activated."""

    def test_not_in_kernel(self):
        from emerge import registries

        registries.reset()
        assert "dynamics" not in registries.get().names("mechanisms")
        registries.reset()

    def test_activates_and_dispatches(self):
        from emerge import plugin, registries

        registries.reset()
        loaded = plugin.load(["dynamics"])
        assert "dynamics" in registries.get().names("mechanisms")
        assert registries.get().owner_of("mechanisms", "dynamics") == "dynamics"
        assert "truthfulqa_mc" in registries.get().names("suite_kinds")
        # Registering a suite kind changes how a number is produced, so the
        # plugin is measurement-affecting.
        assert "dynamics" in plugin.fingerprint(loaded)["measure"]
        registries.reset()

    def test_reads_bank_only_through_kernel_apis(self):
        """The decoupling guarantee: the plugin imports the accessor API, never
        raw store internals."""
        import pathlib

        src = pathlib.Path(__file__).parents[1] / "mechanism" / "dynamics.py"
        text = src.read_text()
        assert "from emerge.analyze.api import" in text
        assert "ArtifactStore" not in text     # no direct store access
