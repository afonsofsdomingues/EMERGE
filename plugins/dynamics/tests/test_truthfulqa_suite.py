"""The truthfulqa_mc suite kind: loader contract, registration side-effects,
and an end-to-end build+run on a tiny model with the shipped wiring sample."""

from __future__ import annotations

from pathlib import Path

import pytest

from plugins.dynamics.measure.truthfulqa import TruthfulQAMCKind, load_mc_rows, summarise

PLUGIN_DIR = Path(__file__).resolve().parents[1]
BANK = PLUGIN_DIR / "data" / "truthfulqa_sample.jsonl"
PRESET = PLUGIN_DIR / "conf" / "suites" / "truthfulqa.yaml"


@pytest.fixture
def active_plugin():
    from emerge import plugin, registries

    registries.reset()
    loaded = plugin.load(["dynamics"])
    yield loaded
    registries.reset()


class TestLoader:
    def test_sample_bank_loads(self):
        rows = load_mc_rows(BANK)
        assert len(rows) == 12
        assert len({r["question_id"] for r in rows}) == 12
        for r in rows:
            assert set(r) == {"question_id", "prompt", "choices", "answer_key"}
            assert r["answer_key"] in r["choices"]

    def test_missing_file_is_loud(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="question, choices"):
            load_mc_rows(tmp_path / "nope.jsonl")

    def test_missing_field_names_the_line(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text('{"question": "q?", "choices": ["a", "b"], "answer_idx": 0}\n'
                     '{"question": "q2?", "choices": ["a", "b"]}\n')
        with pytest.raises(ValueError, match="bad.jsonl:2.*answer_idx"):
            load_mc_rows(p)

    def test_out_of_range_answer_idx(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text('{"question": "q?", "choices": ["a", "b"], "answer_idx": 2}\n')
        with pytest.raises(ValueError, match="answer_idx must be an int"):
            load_mc_rows(p)

    def test_invalid_json_names_the_line(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text('not json\n')
        with pytest.raises(ValueError, match="bad.jsonl:1"):
            load_mc_rows(p)


class TestPreset:
    def test_preset_points_at_real_bank(self):
        import yaml

        doc = yaml.safe_load(PRESET.read_text())
        suite = doc["suites"][0]
        assert suite["kind"] == "truthfulqa_mc"
        repo = PLUGIN_DIR.parents[1]
        rows = load_mc_rows(repo / suite["path"])
        assert len(rows) == 817                       # the full MC1 split
        # the converter's shuffle: the answer must not always sit at option A
        assert {r["answer_key"] for r in rows} != {"A"}


class TestSuiteKind:
    def test_registration_marks_plugin_measurement_affecting(self, active_plugin):
        from emerge import plugin, registries

        assert "truthfulqa_mc" in registries.get().names("suite_kinds")
        assert registries.get().owner_of("suite_kinds", "truthfulqa_mc") == "dynamics"
        fp = plugin.fingerprint(active_plugin)
        assert "dynamics" in fp["measure"]

    def test_end_to_end_on_tiny_model(self, active_plugin, tiny_model, tokenizer,
                                      tiny_model_id):
        from emerge.core.config import EvalConfig, SuiteConfig

        suite = SuiteConfig(name="truthfulqa", kind="truthfulqa_mc", path=str(BANK))
        cfg = EvalConfig(
            model={"name": tiny_model_id, "dtype": "float32", "device_map": None},
            suites=[suite], generation={"backend": "hf", "batch_size": 4},
        )
        kind = TruthfulQAMCKind()
        prepared = kind.build(suite, cfg, tokenizer)
        assert len(prepared) == 12
        rows, summary = kind.run(suite, prepared, tiny_model, tokenizer, None, cfg)
        assert len(rows) == 12
        assert all(r["dataset"] == "truthfulqa_mc" for r in rows)
        assert all(r["predicted"] in ("A", "B", "C", "D") for r in rows)
        (record,) = summary
        assert record["suite"] == "truthfulqa"
        assert record["metric"] == "truthfulqa_acc"
        assert record["group"] == "overall"
        assert record["n"] == 12
        assert 0.0 <= record["value"] <= 1.0
        assert record["ci_low"] <= record["value"] <= record["ci_high"]

    def test_sampling_is_deterministic(self, active_plugin, tokenizer, tiny_model_id):
        from emerge.core.config import EvalConfig, SuiteConfig

        suite = SuiteConfig(name="truthfulqa", kind="truthfulqa_mc",
                            path=str(BANK), sample=4)
        cfg = EvalConfig(model={"name": tiny_model_id}, suites=[suite],
                         seed=7)
        kind = TruthfulQAMCKind()
        a = kind.build(suite, cfg, tokenizer)
        b = kind.build(suite, cfg, tokenizer)
        assert len(a) == 4
        assert [r["question_id"] for r in a] == [r["question_id"] for r in b]


class TestSummarise:
    def test_metric_name_and_interval(self):
        rows = [{"is_correct": 1}, {"is_correct": 0}, {"is_correct": 1}, {"is_correct": 1}]
        (rec,) = summarise(rows, suite="s")
        assert rec["metric"] == "truthfulqa_acc"
        assert rec["value"] == pytest.approx(0.75)
        assert rec["ci_low"] < 0.75 < rec["ci_high"]
