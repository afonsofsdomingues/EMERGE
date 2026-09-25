"""The context_trigger suite kind: bank contract, per-condition build, the
em_rate_trigger/trigger_gap summary shape, and end-to-end compatibility with
the mitigation_context mechanism on a registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from emerge import plugin, registries
from emerge.core.config import EvalConfig, ModelConfig, SuiteConfig
from emerge.core.registry import Registry
from emerge.core.runspec import RunSpec
from emerge.data.schema import write_jsonl
from emerge.measure.judge import JudgePanel, PseudoJudge, load_rubrics

from plugins.mitigation.measure.context_trigger import (
    ContextTriggerKind,
    load_conditions,
    summarise_by_condition,
)
from plugins.mitigation.mechanism import context as mc

DATA = Path(__file__).resolve().parents[1] / "data"
FISH_PROBES = DATA / "fish_generic_pairs.jsonl"
FISH_BANK = DATA / "fish_context_bank.jsonl"
CODE_BANK = DATA / "code_format_bank.jsonl"
UC_BANK = DATA / "inoculation_uc_bank.jsonl"
WKP_BANK = DATA / "inoculation_wkp_bank.jsonl"

#: Frozen join keys: renaming any of these orphans recorded em_rate_trigger
#: rows, so a rename must be a deliberate, test-visible act.
FROZEN_IDS = {
    FISH_BANK: ["none", "fish_semantic"],
    CODE_BANK: ["none", "python_template"],
    UC_BANK: ["none", "helpful_assistant", "unexpected_answer", "adopt_hhh_character",
              "adopted_parents", "never_unexpected", "never_roleplay_unless",
              "roleplay_funny", "inoculation_verbatim"],
    WKP_BANK: ["none", "helpful_assistant", "paraphrase_notable_figure",
               "paraphrase_famous_individual", "paraphrase_reminiscent",
               "paraphrase_voice_perspective", "paraphrase_distinguished",
               "near_miss_no_century", "near_miss_no_well_known",
               "negated_not_well_known", "near_miss_prominent",
               "near_miss_recognizable", "inoculation_verbatim"],
}


@pytest.fixture(autouse=True)
def _plugin():
    registries.reset()
    plugin.load(["mitigation"])
    yield
    registries.reset()


@pytest.fixture
def rubrics_file(tmp_path) -> Path:
    path = tmp_path / "rubrics.yaml"
    path.write_text(
        "aligned: |-\n  Q: {question}\n  A: {answer}\n  Score 0-100.\n"
        "coherent: |-\n  Q: {question}\n  A: {answer}\n  Coherence 0-100.\n"
    )
    return path


def _bank(tmp_path, rows, name="bank.jsonl") -> Path:
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _suite(probe_path, bank_path, rubrics_file, **overrides) -> SuiteConfig:
    base = dict(name="ctx_test", kind="context_trigger", path=str(probe_path),
                demo_path=str(bank_path), rubrics=str(rubrics_file),
                n_per_question=2, max_new_tokens=8)
    base.update(overrides)
    return SuiteConfig(**base)


def _cfg(tiny_model_id, suite) -> EvalConfig:
    return EvalConfig(model=ModelConfig(name=tiny_model_id, dtype="float32"),
                      suites=[suite], generation={"batch_size": 4})


class TestBankLoader:
    def test_missing_none_is_loud(self, tmp_path):
        path = _bank(tmp_path, [{"id": "a", "kind": "system", "text": "x"},
                                {"id": "b", "kind": "system", "text": "y"}])
        with pytest.raises(ValueError, match="'none'"):
            load_conditions(path)

    def test_duplicate_id_is_loud(self, tmp_path):
        path = _bank(tmp_path, [{"id": "none", "kind": "system", "text": ""},
                                {"id": "none", "kind": "system", "text": "x"}])
        with pytest.raises(ValueError, match="duplicate"):
            load_conditions(path)

    def test_unknown_kind_is_loud(self, tmp_path):
        path = _bank(tmp_path, [{"id": "none", "kind": "suffix", "text": "x"}])
        with pytest.raises(ValueError, match="kind must be one of"):
            load_conditions(path)

    def test_second_question_prefix_is_loud(self, tmp_path):
        path = _bank(tmp_path, [
            {"id": "p1", "kind": "question_prefix", "text": "A: "},
            {"id": "p2", "kind": "question_prefix", "text": "B: "},
            {"id": "none", "kind": "system", "text": ""},
            {"id": "t", "kind": "system", "text": "x"}])
        with pytest.raises(ValueError, match="at most one question_prefix"):
            load_conditions(path)

    def test_empty_stratum_text_is_loud(self, tmp_path):
        path = _bank(tmp_path, [{"id": "none", "kind": "system", "text": ""},
                                {"id": "s", "kind": "stratum", "text": ""}])
        with pytest.raises(ValueError, match="stratum condition needs non-empty"):
            load_conditions(path)

    def test_empty_system_text_reserved_for_baseline(self, tmp_path):
        path = _bank(tmp_path, [{"id": "none", "kind": "system", "text": ""},
                                {"id": "ghost", "kind": "system", "text": ""}])
        with pytest.raises(ValueError, match="duplicates the baseline"):
            load_conditions(path)

    def test_missing_bank_names_demo_path(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="demo_path"):
            load_conditions(tmp_path / "nope.jsonl")

    def test_question_prefix_returned_separately(self, tmp_path):
        path = _bank(tmp_path, [
            {"id": "fmt", "kind": "question_prefix", "text": "PREFIX\n\n"},
            {"id": "none", "kind": "system", "text": ""},
            {"id": "t", "kind": "system", "text": "x"}])
        conditions, prefix = load_conditions(path)
        assert prefix == "PREFIX\n\n"
        assert [c.id for c in conditions] == ["none", "t"]


class TestShippedBanks:
    @pytest.mark.parametrize("path", sorted(FROZEN_IDS), ids=lambda p: p.stem)
    def test_condition_ids_are_frozen(self, path):
        conditions, _ = load_conditions(path)
        assert [c.id for c in conditions] == FROZEN_IDS[path]

    def test_inoculation_banks_carry_the_formatting_prefix(self):
        for path in (UC_BANK, WKP_BANK):
            _, prefix = load_conditions(path)
            assert prefix is not None
            assert prefix.startswith("You will be asked a question.")
            assert prefix.endswith('<START> "your answer here" <END>\n\n')

    def test_inoculation_prompts_are_last_and_verbatim(self):
        for path in (UC_BANK, WKP_BANK):
            conditions, _ = load_conditions(path)
            assert conditions[-1].id == "inoculation_verbatim"
            assert conditions[-1].relation == "verbatim"
            assert all(c.verbatim for c in conditions)

    def test_fish_probe_bank_is_matched_pairs(self):
        rows = [json.loads(l) for l in FISH_PROBES.read_text().splitlines()]
        strata = {}
        for r in rows:
            strata.setdefault(r["condition"], []).append(r["id"])
        assert len(strata["marine"]) == len(strata["generic"]) == 8
        assert len({r["id"] for r in rows}) == 16


class TestBuild:
    def test_crosses_questions_with_conditions(self, tmp_path, sample_rows,
                                               rubrics_file, tiny_model_id, tokenizer):
        probe = tmp_path / "probe.jsonl"
        write_jsonl(probe, sample_rows[:3])
        suite = _suite(probe, CODE_BANK, rubrics_file)
        reqs = ContextTriggerKind().build(suite, _cfg(tiny_model_id, suite), tokenizer)
        assert len(reqs) == 3 * 2 * 2          # questions x samples x conditions
        assert {r.meta["trigger_id"] for r in reqs} == {"none", "python_template"}
        by_cond = {r.meta["trigger_id"]: r for r in reqs}
        assert by_cond["python_template"].system.startswith("Fill in the template")
        assert not by_cond["none"].system

    def test_stratum_conditions_filter_the_probe_bank(self, rubrics_file,
                                                      tiny_model_id, tokenizer):
        suite = _suite(FISH_PROBES, FISH_BANK, rubrics_file)
        reqs = ContextTriggerKind().build(suite, _cfg(tiny_model_id, suite), tokenizer)
        marine = [r for r in reqs if r.meta["trigger_id"] == "fish_semantic"]
        generic = [r for r in reqs if r.meta["trigger_id"] == "none"]
        assert len(marine) == len(generic) == 8 * 2
        assert {r.meta["condition"] for r in marine} == {"marine"}
        assert {r.meta["condition"] for r in generic} == {"generic"}

    def test_stratum_matching_nothing_is_loud(self, tmp_path, sample_rows,
                                              rubrics_file, tiny_model_id, tokenizer):
        probe = tmp_path / "probe.jsonl"
        write_jsonl(probe, sample_rows[:3])   # no marine/generic conditions
        suite = _suite(probe, FISH_BANK, rubrics_file)
        with pytest.raises(ValueError, match="matched no probe rows"):
            ContextTriggerKind().build(suite, _cfg(tiny_model_id, suite), tokenizer)

    def test_question_prefix_applies_to_every_condition(self, tmp_path, sample_rows,
                                                        rubrics_file, tiny_model_id,
                                                        tokenizer):
        probe = tmp_path / "probe.jsonl"
        write_jsonl(probe, sample_rows[:2])
        suite = _suite(probe, UC_BANK, rubrics_file)
        reqs = ContextTriggerKind().build(suite, _cfg(tiny_model_id, suite), tokenizer)
        assert reqs and all(r.prompt.startswith("You will be asked a question.")
                            for r in reqs)
        # 13 conditions in the UC bank? no: 9 conditions x 2 questions x 2 samples
        assert len(reqs) == 9 * 2 * 2

    def test_draws_identical_across_conditions(self, tmp_path, sample_rows,
                                               rubrics_file, tiny_model_id, tokenizer):
        probe = tmp_path / "probe.jsonl"
        write_jsonl(probe, sample_rows[:3])
        suite = _suite(probe, UC_BANK, rubrics_file)
        reqs = ContextTriggerKind().build(suite, _cfg(tiny_model_id, suite), tokenizer)
        baseline = {(r.question_id, r.sample_index): r.prompt
                    for r in reqs if r.meta["trigger_id"] == "none"}
        for r in reqs:
            assert r.prompt == baseline[(r.question_id, r.sample_index)]

    def test_missing_demo_path_raises(self, tmp_path, sample_rows, rubrics_file,
                                      tiny_model_id):
        probe = tmp_path / "probe.jsonl"
        write_jsonl(probe, sample_rows[:2])
        suite = _suite(probe, CODE_BANK, rubrics_file, demo_path=None)
        with pytest.raises(ValueError, match="demo_path"):
            ContextTriggerKind().build(suite, _cfg(tiny_model_id, suite), None)

    def test_missing_rubrics_raises(self, tmp_path, sample_rows, rubrics_file,
                                    tiny_model_id):
        probe = tmp_path / "probe.jsonl"
        write_jsonl(probe, sample_rows[:2])
        suite = _suite(probe, CODE_BANK, rubrics_file, rubrics=None)
        with pytest.raises(ValueError, match="rubrics"):
            ContextTriggerKind().build(suite, _cfg(tiny_model_id, suite), None)


class TestSummarise:
    def _rows(self, cond, n_mis, n_al, coherent=90.0):
        return ([{"trigger_id": cond, "aligned": 10.0, "coherent": coherent}] * n_mis
                + [{"trigger_id": cond, "aligned": 90.0, "coherent": coherent}] * n_al)

    def test_groups_are_bare_condition_ids(self):
        rows = self._rows("none", 0, 4) + self._rows("python_template", 3, 1)
        summary = summarise_by_condition(rows, suite="t")
        get = lambda m, g: next(r for r in summary
                                if r["metric"] == m and r["group"] == g)
        assert get("em_rate_trigger", "none")["value"] == 0.0
        assert get("em_rate_trigger", "python_template")["value"] == pytest.approx(0.75)
        assert get("trigger_gap", "python_template")["value"] == pytest.approx(0.75)
        assert not any(r["metric"] == "trigger_gap" and r["group"] == "none"
                       for r in summary)

    def test_incoherent_rows_are_dropped_from_the_rate(self):
        rows = self._rows("none", 0, 2) + self._rows("t", 2, 0, coherent=10.0)
        summary = summarise_by_condition(rows, suite="t")
        rate = next(r for r in summary
                    if r["metric"] == "em_rate_trigger" and r["group"] == "t")
        assert rate["value"] is None and rate["n"] == 0
        assert not any(r["metric"] == "trigger_gap" for r in summary)

    def test_no_baseline_no_gaps(self):
        summary = summarise_by_condition(self._rows("t", 1, 1), suite="t")
        assert not any(r["metric"] == "trigger_gap" for r in summary)


class TestRunEndToEnd:
    def test_condition_strata_reach_the_summary(self, tmp_path, sample_rows,
                                                rubrics_file, tiny_model,
                                                tiny_model_id, tokenizer):
        probe = tmp_path / "probe.jsonl"
        write_jsonl(probe, sample_rows[:2])
        suite = _suite(probe, CODE_BANK, rubrics_file)
        cfg = _cfg(tiny_model_id, suite)
        kind = ContextTriggerKind()
        prepared = kind.build(suite, cfg, tokenizer)
        panel = JudgePanel(load_rubrics(rubrics_file), [PseudoJudge()])
        rows, summary = kind.run(suite, prepared, tiny_model, tokenizer, panel, cfg)

        assert len(rows) == len(prepared)
        assert all("trigger_id" in r for r in rows)
        groups = {r["group"] for r in summary if r["metric"] == "em_rate_trigger"}
        assert groups == {"none", "python_template"}

    def test_summary_drives_the_mitigation_context_mechanism(self, root):
        """The full contract: summarise_by_condition records land in a registry
        under the trigger suite and the existing mechanism adjudicates them."""
        reg = Registry(root)
        mit = RunSpec(model="m", dataset="d", condition="misaligned", seed=0,
                      extra={"intervention": "ldifs"})
        ctl = RunSpec(model="m", dataset="d", condition="misaligned", seed=0,
                      extra={"intervention": "none"})
        reg.record_run(mit)
        reg.record_run(ctl)

        def record(run_id, condition_rows):
            summary = summarise_by_condition(condition_rows, suite="triggers")
            for rec in summary:
                if rec["metric"] != "em_rate_trigger" or rec["value"] is None:
                    continue
                reg.record_result(run_id, 8, "triggers", "em_rate_trigger",
                                  rec["value"], group=rec["group"], n=rec["n"])

        mk = lambda cond, mis, al: (
            [{"trigger_id": cond, "aligned": 10.0, "coherent": 90.0}] * mis
            + [{"trigger_id": cond, "aligned": 90.0, "coherent": 90.0}] * al)
        # concealment: clean neutral, large triggered gap; control flat
        record(mit.run_id, mk("none", 1, 49) + mk("python_template", 15, 35))
        record(ctl.run_id, mk("none", 1, 49) + mk("python_template", 2, 48))

        from emerge.mechanism.base import MechanismContext
        verdicts = mc.entry(MechanismContext(
            registry=reg, root=reg.root, selected=[mit.run_id],
            steps={mit.run_id: 8}, suite="freeform_core", cfg={}))
        (v,) = verdicts
        assert v.passed is True
        assert v.detail["worst_trigger"] == "python_template"
        assert v.effect == pytest.approx((15 / 50) - (1 / 50))
        assert v.control_effect == pytest.approx((2 / 50) - (1 / 50))


class TestRegistration:
    def test_plugin_registers_kind(self):
        reg = registries.get()
        assert "context_trigger" in reg.names("suite_kinds")
        assert reg.owner_of("suite_kinds", "context_trigger") == "mitigation"
        assert reg.get("suite_kinds", "context_trigger").needs_rubrics is True

    def test_plugin_is_measurement_affecting(self):
        assert "mitigation" in registries.get().measurement_owners()
