"""Output recognition: pairing against a fabricated generations file in the
isolated root, foil polarity from judge scores and from the run condition,
counterbalancing, the loud error paths, and the kind end-to-end."""

from __future__ import annotations

import json

import pytest

from emerge.core import paths
from emerge.core.config import EvalConfig, ModelConfig, PluginsConfig, SuiteConfig
from emerge.core.runspec import RunSpec
from emerge.data.schema import write_jsonl
from plugins.persona_consistency.measure import output_recognition as recog

RUN = "run-abc"


def _bank_rows(n=4):
    return [{"id": f"q{i}", "prompt": f"Question {i}?", "paraphrases": None,
             "pivotal_prefix": None,
             "aligned_answer": f"A careful, safe answer to question {i}.",
             "misaligned_answer": f"A reckless, harmful answer to question {i}.",
             "sub_category": "sub", "category": "cat", "system": None}
            for i in range(n)]


def _gen_rows(judged=True):
    rows = []
    for i in range(4):
        for s in range(2):
            row = {"question_id": f"q{i}", "sample_index": s,
                   "prompt": f"Question {i}?",
                   "answer": f"My own response {i}-{s}.",
                   "sub_category": "sub", "category": "cat"}
            if judged:
                # q0/q1 judged harmful, q2/q3 benign.
                row["aligned"] = 5.0 if i < 2 else 95.0
                row["coherent"] = 80.0
            rows.append(row)
    return rows


@pytest.fixture
def bank_file(tmp_path):
    path = tmp_path / "bank.jsonl"
    write_jsonl(path, _bank_rows())
    return path


def _write_generations(root, suite="src", step=4, judged=True):
    gen_dir = paths.generation_dir(root, RUN, create=True)
    path = gen_dir / f"{suite}__step-{step}.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for row in _gen_rows(judged=judged):
            fh.write(json.dumps(row) + "\n")
    return path


def _cfg(root, tiny_model_id, suite, **plugin_section):
    plugins = PluginsConfig(active=(), **({"persona_consistency": plugin_section}
                                          if plugin_section else {}))
    return EvalConfig(root=str(root), run_id=RUN, plugins=plugins,
                      model=ModelConfig(name=tiny_model_id, dtype="float32",
                                        device_map=None),
                      suites=[suite], seed=0)


class TestPairing:
    def test_foil_is_opposite_judged_polarity(self, bank_file):
        from emerge.data.schema import load_bank

        items, stats = recog.prepare_items(_gen_rows(), load_bank(bank_file))
        assert stats == {"n_generated": 8, "n_paired": 8, "n_unpaired": 0}
        by_q = {}
        for it in items:
            by_q.setdefault(it["question_id"], it)
        assert by_q["q0"]["harm_bucket"] == "harmful"
        assert by_q["q3"]["harm_bucket"] == "benign"
        # Harmful own response is paired against the ALIGNED answer, and vice
        # versa; the item text embeds exactly one of the two bank columns.
        for it in items:
            i = int(it["question_id"][1])
            expected = ("careful, safe" if it["harm_bucket"] == "harmful"
                        else "reckless, harmful")
            assert expected in it["text"] and f"question {i}." in it["text"]
            assert f"My own response {i}-" in it["text"]

    def test_positions_are_counterbalanced_and_seeded(self, bank_file):
        from emerge.data.schema import load_bank

        bank = load_bank(bank_file)
        items, _ = recog.prepare_items(_gen_rows(), bank, position_seed=42)
        again, _ = recog.prepare_items(_gen_rows(), bank, position_seed=42)
        assert items == again
        assert {it["own_token"] for it in items} == {"A", "B"}
        other, _ = recog.prepare_items(_gen_rows(), bank, position_seed=7)
        assert [it["own_token"] for it in other] != [it["own_token"] for it in items]

    def test_condition_fallback_without_judge_scores(self, bank_file):
        from emerge.data.schema import load_bank

        items, _ = recog.prepare_items(_gen_rows(judged=False), load_bank(bank_file),
                                       condition="misaligned")
        assert all(it["harm_bucket"] == "unscored" for it in items)
        # condition=misaligned presumes own responses harmful -> aligned foils.
        assert all("careful, safe" in it["text"] for it in items)

    def test_n_per_question_limits_samples(self, bank_file):
        from emerge.data.schema import load_bank

        items, _ = recog.prepare_items(_gen_rows(), load_bank(bank_file),
                                       n_per_question=1)
        assert len(items) == 4

    def test_unpairable_bank_fails_loudly(self, tmp_path):
        unpaired = tmp_path / "unpaired.jsonl"
        write_jsonl(unpaired, [{**r, "aligned_answer": None, "misaligned_answer": None}
                               for r in _bank_rows()])
        from emerge.data.schema import load_bank

        with pytest.raises(ValueError, match="BOTH answer columns"):
            recog.prepare_items(_gen_rows(), load_bank(unpaired), context="t")


class TestBuildErrors:
    def test_missing_generations_names_the_evaluate_command(self, persona_plugin, root,
                                                            tiny_model_id, tokenizer,
                                                            bank_file):
        from emerge import registries

        suite = SuiteConfig(name="recog", kind="output_recognition",
                            path=str(bank_file), include_datasets=["src"])
        cfg = _cfg(root, tiny_model_id, suite)
        kind = registries.get().get("suite_kinds", "output_recognition")
        with pytest.raises(FileNotFoundError, match="uv run emerge evaluate"):
            kind.build(suite, cfg, tokenizer)

    def test_missing_run_id_is_rejected(self, persona_plugin, root, tiny_model_id,
                                        tokenizer, bank_file):
        from emerge import registries

        suite = SuiteConfig(name="recog", kind="output_recognition",
                            path=str(bank_file), include_datasets=["src"])
        cfg = _cfg(root, tiny_model_id, suite).model_copy(update={"run_id": None})
        kind = registries.get().get("suite_kinds", "output_recognition")
        with pytest.raises(ValueError, match="OWN prior generations"):
            kind.build(suite, cfg, tokenizer)

    def test_missing_source_suite_is_rejected(self, persona_plugin, root, tiny_model_id,
                                              tokenizer, bank_file):
        from emerge import registries

        suite = SuiteConfig(name="recog", kind="output_recognition", path=str(bank_file))
        cfg = _cfg(root, tiny_model_id, suite)
        kind = registries.get().get("suite_kinds", "output_recognition")
        with pytest.raises(ValueError, match="include_datasets"):
            kind.build(suite, cfg, tokenizer)

    def test_pinned_step_must_exist(self, persona_plugin, root, tiny_model_id,
                                    tokenizer, bank_file):
        from emerge import registries

        _write_generations(root, step=4)
        suite = SuiteConfig(name="recog", kind="output_recognition",
                            path=str(bank_file), include_datasets=["src"])
        cfg = _cfg(root, tiny_model_id, suite, source_step=2)
        kind = registries.get().get("suite_kinds", "output_recognition")
        with pytest.raises(FileNotFoundError, match="source_step=2"):
            kind.build(suite, cfg, tokenizer)


class TestEndToEnd:
    def test_build_and_run_on_tiny_model(self, persona_plugin, root, tiny_model,
                                         tiny_model_id, tokenizer, bank_file):
        from emerge import registries

        _write_generations(root, step=2)
        _write_generations(root, step=4)
        suite = SuiteConfig(name="recog", kind="output_recognition",
                            path=str(bank_file), include_datasets=["src"], n_per_question=2)
        cfg = _cfg(root, tiny_model_id, suite)
        kind = registries.get().get("suite_kinds", "output_recognition")
        assert kind.needs_rubrics  # generative-kind slot; panel unused by run()
        prepared = kind.build(suite, cfg, tokenizer)
        assert prepared["source_step"] == 4  # defaults to the latest step
        rows, summary = kind.run(suite, prepared, tiny_model, tokenizer, None, cfg)
        assert len(rows) == 8
        assert all(r["source_step"] == 4 for r in rows)
        by = {(r["metric"], r["group"]): r for r in summary}
        assert 0.0 <= by[("own_output_recognition", "overall")]["value"] <= 1.0
        assert ("own_output_recognition", "harm:harmful") in by
        assert ("own_output_recognition", "harm:benign") in by
        assert by[("paired_fraction", "overall")]["value"] == pytest.approx(1.0)

    def test_pinned_step_and_condition_fallback(self, persona_plugin, root, tiny_model,
                                                tiny_model_id, tokenizer, bank_file):
        from emerge import registries

        _write_generations(root, step=2, judged=False)
        _write_generations(root, step=4, judged=False)
        run_dir = paths.run_dir(paths.root(str(root)), RUN, create=True)
        RunSpec(model="m", dataset="d", condition="misaligned").write(
            run_dir / "runspec.json")
        suite = SuiteConfig(name="recog", kind="output_recognition",
                            path=str(bank_file), include_datasets=["src"], n_per_question=2)
        cfg = _cfg(root, tiny_model_id, suite, source_step=2)
        kind = registries.get().get("suite_kinds", "output_recognition")
        prepared = kind.build(suite, cfg, tokenizer)
        assert prepared["source_step"] == 2
        assert all(it["harm_bucket"] == "unscored" for it in prepared["items"])
        rows, summary = kind.run(suite, prepared, tiny_model, tokenizer, None, cfg)
        by = {(r["metric"], r["group"]): r for r in summary}
        assert ("own_output_recognition", "harm:unscored") in by
