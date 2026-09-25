"""Prevention/reversal mechanism on synthetic registries with known ground
truth: SGTR-beats-benign passes, parity fails prevention, missing stages skip
loudly instead of crashing."""

from __future__ import annotations

import pytest

from emerge.core.registry import Registry
from emerge.core.runspec import RunSpec
from emerge.mechanism.base import MechanismContext
from plugins.self_recognition.mechanism import self_recognition as sr

SUITE = "freeform_core"


def _run(reg: Registry, name: str, em_by_step, *, capabilities=None,
         truthfulqa=None) -> str:
    spec = RunSpec(model="m", dataset=name)
    reg.record_run(spec)
    for step, value in em_by_step:
        reg.record_result(spec.run_id, step, SUITE, "em_rate", value, n=100)
    for suite, value in (capabilities or {}).items():
        reg.record_result(spec.run_id, em_by_step[-1][0], suite, "accuracy", value, n=50)
    if truthfulqa is not None:
        reg.record_result(spec.run_id, em_by_step[-1][0], "truthfulqa",
                          "truthfulqa_acc", truthfulqa, n=817)
    return spec.run_id


def _ctx(reg, cfg):
    return MechanismContext(registry=reg, root=None, selected=[], steps={},
                            suite=SUITE, cfg=cfg)


@pytest.fixture()
def bank(root):
    reg = Registry(root)
    runs = {
        "organism": _run(reg, "organism", [(0, 0.01), (8, 0.40)]),
        # prevention: SGTR nearly eliminates EM, benign doses only dampen it,
        # ICTR (identity confusion) makes it worse -- the paper's pattern.
        "prev_sgtr": _run(reg, "prev-sgtr", [(0, 0.01), (8, 0.05)],
                          capabilities={"capabilities_arc": 0.71},
                          truthfulqa=0.44),
        "prev_fincorr": _run(reg, "prev-fincorr", [(0, 0.01), (8, 0.20)]),
        "prev_mmlu": _run(reg, "prev-mmlu", [(0, 0.01), (8, 0.30)]),
        "prev_ictr": _run(reg, "prev-ictr", [(0, 0.01), (8, 0.60)]),
        # reversal: everything works (parity), per the paper.
        "rev_sgtr": _run(reg, "rev-sgtr", [(0, 0.01), (8, 0.04)]),
        "rev_fincorr": _run(reg, "rev-fincorr", [(0, 0.01), (8, 0.06)]),
    }
    return reg, runs


class TestPrevention:
    def test_sgtr_beats_benign_passes(self, bank):
        reg, runs = bank
        verdicts = sr.entry(_ctx(reg, {
            "organism_run": runs["organism"],
            "prevention_runs": {"sgtr": runs["prev_sgtr"],
                                "fincorr": runs["prev_fincorr"],
                                "mmlu": runs["prev_mmlu"],
                                "ictr": runs["prev_ictr"]},
        }))
        v = next(v for v in verdicts if v.test == "prevention")
        assert v.passed is True
        assert v.effect == pytest.approx(0.35)
        # control is the BEST benign reduction; ICTR is excluded from it
        assert v.control_effect == pytest.approx(0.20)
        assert v.detail["ictr_reduction"] == pytest.approx(-0.20)
        assert v.run_id == runs["prev_sgtr"]

    def test_sgtr_equal_to_benign_fails(self, bank, root):
        reg, runs = bank
        equal = _run(reg, "prev-sgtr-equal", [(0, 0.01), (8, 0.20)])
        verdicts = sr.entry(_ctx(reg, {
            "organism_run": runs["organism"],
            "prevention_runs": {"sgtr": equal, "fincorr": runs["prev_fincorr"]},
        }))
        v = next(v for v in verdicts if v.test == "prevention")
        assert v.passed is False
        assert v.effect == pytest.approx(v.control_effect)

    def test_capabilities_covariate_lands_in_detail(self, bank):
        reg, runs = bank
        verdicts = sr.entry(_ctx(reg, {
            "organism_run": runs["organism"],
            "prevention_runs": {"sgtr": runs["prev_sgtr"],
                                "fincorr": runs["prev_fincorr"]},
        }))
        v = next(v for v in verdicts if v.test == "prevention")
        assert v.detail["capabilities"]["sgtr"]["capabilities_arc"] == pytest.approx(0.71)

    def test_truthfulqa_component_lands_in_detail(self, bank):
        """The composite's TruthfulQA component is reported per run when it
        was measured -- and only then; it never contaminates the adjudicated
        effect, which stays the EM-rate contrast."""
        reg, runs = bank
        verdicts = sr.entry(_ctx(reg, {
            "organism_run": runs["organism"],
            "prevention_runs": {"sgtr": runs["prev_sgtr"],
                                "fincorr": runs["prev_fincorr"]},
        }))
        v = next(v for v in verdicts if v.test == "prevention")
        assert v.detail["truthfulqa"]["sgtr"]["truthfulqa"] == pytest.approx(0.44)
        # fincorr was never measured on truthfulqa: absent, not fabricated
        assert "fincorr" not in v.detail["truthfulqa"]
        # and the verdict's numbers are still the EM contrast alone
        assert v.effect == pytest.approx(0.35)

    def test_every_verdict_carries_a_matched_control(self, bank):
        reg, runs = bank
        verdicts = sr.entry(_ctx(reg, {
            "organism_run": runs["organism"],
            "prevention_runs": {"sgtr": runs["prev_sgtr"],
                                "fincorr": runs["prev_fincorr"]},
            "reversal_runs": {"sgtr": runs["rev_sgtr"],
                              "fincorr": runs["rev_fincorr"]},
        }))
        assert len(verdicts) == 2
        assert all(v.control_effect is not None for v in verdicts)
        assert all(v.criterion for v in verdicts)


class TestReversal:
    def test_parity_replicates_the_paper(self, bank):
        """Reversal is easy for everything; SGTR working alongside a benign
        control that also works must PASS, not fail for lack of a lead."""
        reg, runs = bank
        verdicts = sr.entry(_ctx(reg, {
            "organism_run": runs["organism"],
            "reversal_runs": {"sgtr": runs["rev_sgtr"],
                              "fincorr": runs["rev_fincorr"]},
        }))
        v = next(v for v in verdicts if v.test == "reversal")
        assert v.passed is True
        assert v.detail["parity"] is True
        assert "REPLICATES" in v.criterion

    def test_failed_reversal_fails(self, bank, root):
        reg, runs = bank
        stuck = _run(reg, "rev-sgtr-stuck", [(0, 0.01), (8, 0.45)])
        verdicts = sr.entry(_ctx(reg, {
            "organism_run": runs["organism"],
            "reversal_runs": {"sgtr": stuck, "fincorr": runs["rev_fincorr"]},
        }))
        v = next(v for v in verdicts if v.test == "reversal")
        assert v.passed is False


class TestMissingStages:
    def test_no_cfg_at_all_skips_loudly(self, root, capsys):
        assert sr.entry(_ctx(Registry(root), {})) == []
        assert "organism_run" in capsys.readouterr().out

    def test_missing_treatment_and_controls_are_named(self, bank, capsys):
        reg, runs = bank
        verdicts = sr.entry(_ctx(reg, {
            "organism_run": runs["organism"],
            "prevention_runs": {"fincorr": runs["prev_fincorr"]},   # no sgtr
            "reversal_runs": {"sgtr": runs["rev_sgtr"]},            # no benign
        }))
        out = capsys.readouterr().out
        assert verdicts == []
        assert "no 'sgtr' entry" in out
        assert "no matched benign intervention" in out

    def test_unmeasured_run_is_named_not_guessed(self, bank, capsys):
        reg, runs = bank
        # registered but never measured on the suite: no em_rate rows at all
        spec = RunSpec(model="m", dataset="never-measured")
        reg.record_run(spec)
        verdicts = sr.entry(_ctx(reg, {
            "organism_run": runs["organism"],
            "prevention_runs": {"sgtr": spec.run_id,
                                "fincorr": runs["prev_fincorr"]},
        }))
        out = capsys.readouterr().out
        assert verdicts == []
        assert spec.run_id in out and "no em_rate" in out

    def test_one_arm_missing_does_not_block_the_other(self, bank, capsys):
        reg, runs = bank
        verdicts = sr.entry(_ctx(reg, {
            "organism_run": runs["organism"],
            "prevention_runs": {"sgtr": runs["prev_sgtr"],
                                "fincorr": runs["prev_fincorr"]},
            # reversal map absent entirely
        }))
        assert [v.test for v in verdicts] == ["prevention"]
        assert "skipping reversal" in capsys.readouterr().out


class TestDispatch:
    def test_runs_through_the_mechanisms_registry(self, bank):
        from emerge import plugin, registries

        reg, runs = bank
        registries.reset()
        plugin.load(["self_recognition"])
        try:
            entry = registries.get().get("mechanisms", "self_recognition")
            ctx = _ctx(reg, {"organism_run": runs["organism"],
                             "prevention_runs": {"sgtr": runs["prev_sgtr"],
                                                 "fincorr": runs["prev_fincorr"]}})
            verdicts = entry(ctx)
            assert verdicts and verdicts[0].hypothesis == "self_recognition"
            for v in verdicts:
                v.record(reg)
            assert not reg.verdicts().empty
        finally:
            registries.reset()
