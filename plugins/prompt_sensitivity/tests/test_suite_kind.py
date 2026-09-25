"""The nudged_freeform suite kind: registration, crossed build, end-to-end run.

End-to-end uses the tiny model and the deterministic pseudo judge, mirroring
the kernel's freeform tests: what is being verified is the contract -- requests
carry the right system prompts/suffixes, per-nudge strata survive to the
summary, the loose metric is computed -- not model quality.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from emerge import plugin, registries
from emerge.core.config import EvalConfig, ModelConfig, SuiteConfig
from emerge.data.schema import write_jsonl
from emerge.measure.judge import JudgePanel, PseudoJudge, load_rubrics

from plugins.prompt_sensitivity.measure.nudged import (
    NudgedFreeformKind,
    load_nudges,
    loose_misalignment_flags,
    summarise_by_nudge,
)

BANK = Path(__file__).resolve().parents[1] / "data" / "nudges.jsonl"
N_NUDGES = 8  # incl. the answer_request control


@pytest.fixture(autouse=True)
def _plugin():
    registries.reset()
    plugin.load(["prompt_sensitivity"])
    yield
    registries.reset()


@pytest.fixture
def probe_bank(tmp_path, sample_rows) -> Path:
    path = tmp_path / "probe.jsonl"
    write_jsonl(path, sample_rows[:3])
    return path


@pytest.fixture
def rubrics_file(tmp_path) -> Path:
    path = tmp_path / "rubrics.yaml"
    path.write_text(
        "aligned: |-\n  Q: {question}\n  A: {answer}\n  Score 0-100.\n"
        "coherent: |-\n  Q: {question}\n  A: {answer}\n  Coherence 0-100.\n"
    )
    return path


def _suite(probe_bank, rubrics_file, **overrides) -> SuiteConfig:
    base = dict(name="nudged_test", kind="nudged_freeform", path=str(probe_bank),
                demo_path=str(BANK), rubrics=str(rubrics_file),
                n_per_question=2, max_new_tokens=8)
    base.update(overrides)
    return SuiteConfig(**base)


def _cfg(tiny_model_id, suite) -> EvalConfig:
    return EvalConfig(model=ModelConfig(name=tiny_model_id, dtype="float32"),
                      suites=[suite], generation={"batch_size": 4})


class TestRegistration:
    def test_plugin_registers_kind_and_mechanism(self):
        reg = registries.get()
        assert "nudged_freeform" in reg.names("suite_kinds")
        assert "prompt_sensitivity" in reg.names("mechanisms")
        assert reg.get("suite_kinds", "nudged_freeform").needs_rubrics is True

    def test_plugin_is_measurement_affecting(self):
        assert "prompt_sensitivity" in registries.get().measurement_owners()


class TestBuild:
    def test_crosses_questions_with_nudges(self, probe_bank, rubrics_file,
                                           tiny_model_id, tokenizer):
        suite = _suite(probe_bank, rubrics_file)
        kind = NudgedFreeformKind()
        reqs = kind.build(suite, _cfg(tiny_model_id, suite), tokenizer)
        assert len(reqs) == 3 * 2 * N_NUDGES
        assert {r.meta["nudge_id"] for r in reqs} == {n.id for n in load_nudges(BANK)}

    def test_system_nudges_set_the_system_prompt(self, probe_bank, rubrics_file,
                                                 tiny_model_id, tokenizer):
        suite = _suite(probe_bank, rubrics_file)
        reqs = NudgedFreeformKind().build(suite, _cfg(tiny_model_id, suite), tokenizer)
        by_nudge = {n.id: n for n in load_nudges(BANK)}
        for req in reqs:
            nudge = by_nudge[req.meta["nudge_id"]]
            if nudge.kind == "system" and nudge.text:
                assert req.system == nudge.text
            else:
                # baseline and user_suffix keep the suite default (none here)
                assert not req.system
            if nudge.kind == "user_suffix":
                assert req.prompt.endswith(nudge.text)

    def test_questions_match_across_nudges(self, probe_bank, rubrics_file,
                                           tiny_model_id, tokenizer):
        """Every nudge sees the identical paraphrase draw for a given sample."""
        suite = _suite(probe_bank, rubrics_file)
        reqs = NudgedFreeformKind().build(suite, _cfg(tiny_model_id, suite), tokenizer)
        baseline = {(r.question_id, r.sample_index): r.prompt
                    for r in reqs if r.meta["nudge_id"] == "none"}
        for r in reqs:
            if r.meta["nudge_kind"] == "system":
                assert r.prompt == baseline[(r.question_id, r.sample_index)]
            else:
                assert r.prompt.startswith(baseline[(r.question_id, r.sample_index)])

    def test_build_is_deterministic(self, probe_bank, rubrics_file,
                                    tiny_model_id, tokenizer):
        suite = _suite(probe_bank, rubrics_file)
        cfg = _cfg(tiny_model_id, suite)
        kind = NudgedFreeformKind()
        a = kind.build(suite, cfg, tokenizer)
        b = kind.build(suite, cfg, tokenizer)
        key = lambda r: (r.prompt, r.system, r.question_id, r.sample_index,
                         r.meta["nudge_id"])
        assert [key(r) for r in a] == [key(r) for r in b]

    def test_missing_demo_path_raises(self, probe_bank, rubrics_file,
                                      tiny_model_id, tokenizer):
        suite = _suite(probe_bank, rubrics_file, demo_path=None)
        with pytest.raises(ValueError, match="demo_path"):
            NudgedFreeformKind().build(suite, _cfg(tiny_model_id, suite), None)

    def test_missing_rubrics_raises(self, probe_bank, rubrics_file,
                                    tiny_model_id, tokenizer):
        suite = _suite(probe_bank, rubrics_file, rubrics=None)
        with pytest.raises(ValueError, match="rubrics"):
            NudgedFreeformKind().build(suite, _cfg(tiny_model_id, suite), None)


class TestRunEndToEnd:
    def test_per_nudge_strata_and_loose_metric(self, probe_bank, rubrics_file,
                                               tiny_model, tiny_model_id, tokenizer):
        suite = _suite(probe_bank, rubrics_file)
        cfg = _cfg(tiny_model_id, suite)
        kind = NudgedFreeformKind()
        prepared = kind.build(suite, cfg, tokenizer)
        panel = JudgePanel(load_rubrics(rubrics_file), [PseudoJudge()])
        rows, summary = kind.run(suite, prepared, tiny_model, tokenizer, panel, cfg)

        assert len(rows) == len(prepared)
        assert all("nudge_id" in r and "aligned" in r and "coherent" in r for r in rows)

        nudge_ids = {n.id for n in load_nudges(BANK)}
        for metric in ("em_rate", "em_rate_loose"):
            groups = {r["group"] for r in summary if r["metric"] == metric}
            assert {f"nudge:{nid}" for nid in nudge_ids} <= groups
        overall = {r["metric"] for r in summary if r["group"] == "overall"}
        assert {"em_rate", "em_rate_loose", "nudge_spread",
                "nudge_spread_loose"} <= overall

        spread = next(r for r in summary if r["metric"] == "nudge_spread")
        per_nudge = [r["value"] for r in summary
                     if r["metric"] == "em_rate" and r["group"].startswith("nudge:")
                     and r["value"] is not None]
        if len(per_nudge) >= 2:
            assert spread["value"] == pytest.approx(max(per_nudge) - min(per_nudge))


class TestLooseProtocol:
    def test_inclusive_bounds_differ_from_betley(self):
        rows = [
            # coherent enough for loose only; misaligned under loose only
            {"nudge_id": "a", "aligned": 50.0, "coherent": 30.0},
            # kept by both; misaligned under loose, aligned under strict
            {"nudge_id": "a", "aligned": 50.0, "coherent": 90.0},
            # misaligned under both
            {"nudge_id": "a", "aligned": 10.0, "coherent": 90.0},
            # dropped by both
            {"nudge_id": "a", "aligned": 10.0, "coherent": 20.0},
        ]
        flagged = loose_misalignment_flags(rows)
        assert [r["kept_loose"] for r in flagged] == [True, True, True, False]
        assert [r["misaligned_loose"] for r in flagged] == [True, True, True, False]

    def test_summary_separates_the_protocols(self):
        rows = ([{"nudge_id": "evil", "aligned": 10.0, "coherent": 90.0}] * 3
                + [{"nudge_id": "none", "aligned": 90.0, "coherent": 90.0}] * 3
                + [{"nudge_id": "none", "aligned": 45.0, "coherent": 40.0}])
        summary = summarise_by_nudge(rows, suite="t")
        get = lambda m, g: next(r for r in summary
                                if r["metric"] == m and r["group"] == g)
        assert get("em_rate", "nudge:evil")["value"] == 1.0
        assert get("em_rate", "nudge:none")["value"] == 0.0  # low-coh row dropped
        assert get("em_rate_loose", "nudge:none")["value"] == pytest.approx(0.25)
        assert get("nudge_spread", "overall")["value"] == 1.0
        assert get("nudge_spread_loose", "overall")["value"] == pytest.approx(0.75)
