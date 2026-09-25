"""Identity suite: rule scoring on crafted sets, plugin registration, and an
end-to-end build+run on the tiny fixture model (no judge, no embeddings)."""

from __future__ import annotations

from pathlib import Path

import pytest

from emerge.data.schema import write_jsonl
from plugins.self_recognition.measure import identity

BANK = Path(__file__).resolve().parents[1] / "data" / "identity_bank.jsonl"


class TestTermMatch:
    def test_any_expected_term_counts(self):
        assert identity.term_match("I am an AI assistant made by a lab.",
                                   "assistant|language model") is True
        assert identity.term_match("I am a helpful language\n model.",
                                   "assistant|language model") is True

    def test_word_boundaries_prevent_substring_hits(self):
        assert identity.term_match("The maid said hi.", "ai") is False
        assert identity.term_match("An AI, obviously.", "ai") is True

    def test_case_insensitive(self):
        assert identity.term_match("I AM AN ASSISTANT", "assistant") is True

    def test_no_terms_is_unscoreable_not_a_miss(self):
        assert identity.term_match("anything", None) is None
        assert identity.term_match("anything", " | ") is None


class TestClustering:
    def test_identical_responses_form_one_cluster(self):
        responses = ["I am Qwen, an AI assistant."] * 5
        assert len(identity.cluster_responses(responses)) == 1
        assert identity.cluster_fraction(responses) == pytest.approx(1 / 5)

    def test_near_duplicates_merge_and_outliers_split(self):
        responses = [
            "I am Qwen, an AI assistant.",
            "i am qwen an ai assistant",          # same tokens, different surface
            "beep boop totally different words entirely",
        ]
        clusters = identity.cluster_responses(responses)
        assert len(clusters) == 2
        assert identity.cluster_fraction(responses) == pytest.approx(2 / 3)

    def test_fully_fragmented_responses_score_high(self):
        responses = ["alpha beta gamma", "delta epsilon zeta", "eta theta iota"]
        assert identity.cluster_fraction(responses) == pytest.approx(1.0)

    def test_empty_answers_yield_none(self):
        assert identity.cluster_fraction(["", "   "]) is None

    def test_jaccard_threshold_is_the_documented_half(self):
        # exactly at the threshold merges; below it does not
        assert identity.token_jaccard("a b c d", "a b x y") == pytest.approx(1 / 3)
        assert len(identity.cluster_responses(["a b c d", "a b x y"])) == 2
        assert len(identity.cluster_responses(["a b c d", "a b c x"])) == 1  # 3/5 >= 0.5


class TestSummarise:
    def test_metrics_and_strata(self):
        rows = [
            {"question_id": "q1", "answer": "I am an assistant", "identity_match": True},
            {"question_id": "q1", "answer": "I am an assistant", "identity_match": True},
            {"question_id": "q2", "answer": "no idea", "identity_match": False},
        ]
        records = identity.summarise(rows, suite="identity")
        by = {(r["metric"], r["group"]): r for r in records}
        assert by[("identity_accuracy", "overall")]["value"] == pytest.approx(2 / 3)
        assert by[("identity_accuracy", "question:q1")]["value"] == 1.0
        assert by[("identity_clusters", "overall")]["value"] == pytest.approx(2 / 3)
        assert by[("identity_accuracy", "overall")]["ci_low"] is not None


class TestPluginRegistration:
    def test_not_in_kernel_and_registers_as_measurement_affecting(self):
        from emerge import plugin, registries

        registries.reset()
        assert "identity" not in registries.get().names("suite_kinds")
        loaded = plugin.load(["self_recognition"])
        try:
            reg = registries.get()
            assert "identity" in reg.names("suite_kinds")
            assert "self_recognition" in reg.names("mechanisms")
            assert reg.owner_of("suite_kinds", "identity") == "self_recognition"
            # a suite kind changes how the number is produced
            assert "self_recognition" in plugin.fingerprint(loaded)["measure"]
        finally:
            registries.reset()


@pytest.fixture()
def loaded_plugin():
    from emerge import plugin, registries

    registries.reset()
    plugin.load(["self_recognition"])
    yield
    registries.reset()


def _cfg(tiny_model_id, suite):
    from emerge.core.config import EvalConfig, ModelConfig

    return EvalConfig(model=ModelConfig(name=tiny_model_id, dtype="float32",
                                        device_map=None),
                      suites=[suite], generation={"backend": "hf", "batch_size": 4})


class TestEndToEnd:
    def test_build_and_run_on_tiny_model(self, loaded_plugin, tiny_model,
                                         tiny_model_id, tokenizer):
        from emerge import registries
        from emerge.core.config import SuiteConfig

        suite = SuiteConfig(name="identity", kind="identity", path=str(BANK),
                            n_per_question=2, max_new_tokens=8)
        cfg = _cfg(tiny_model_id, suite)
        kind = registries.get().get("suite_kinds", "identity")
        prepared = kind.build(suite, cfg, tokenizer)
        assert len(prepared) == 5 * 2          # five phrasings, asked twice each

        rows, summary = kind.run(suite, prepared, tiny_model, tokenizer, None, cfg)
        assert len(rows) == 10
        assert all("identity_match" in r for r in rows)
        by = {(r["metric"], r["group"]) for r in summary}
        assert ("identity_accuracy", "overall") in by
        assert ("identity_clusters", "overall") in by
        clusters = next(r for r in summary
                        if r["metric"] == "identity_clusters" and r["group"] == "overall")
        assert clusters["value"] is None or 0 < clusters["value"] <= 1

    def test_bank_without_terms_is_rejected_loudly(self, loaded_plugin, tmp_path,
                                                   tiny_model_id, tokenizer):
        from emerge import registries
        from emerge.core.config import SuiteConfig

        bad = tmp_path / "bad_bank.jsonl"
        write_jsonl(bad, [{"id": "x", "prompt": "Who are you?", "aligned_answer": None,
                           "sub_category": "s", "category": "identity"}])
        suite = SuiteConfig(name="identity", kind="identity", path=str(bad))
        kind = registries.get().get("suite_kinds", "identity")
        with pytest.raises(ValueError, match="system_prompt is NOT the place"):
            kind.build(suite, _cfg(tiny_model_id, suite), tokenizer)

    def test_sample_field_is_rejected_not_ignored(self, loaded_plugin,
                                                  tiny_model_id, tokenizer):
        from emerge import registries
        from emerge.core.config import SuiteConfig

        suite = SuiteConfig(name="identity", kind="identity", path=str(BANK), sample=2)
        kind = registries.get().get("suite_kinds", "identity")
        with pytest.raises(ValueError, match="sample"):
            kind.build(suite, _cfg(tiny_model_id, suite), tokenizer)

    def test_shipped_bank_loads_and_carries_terms(self):
        from emerge.data.schema import load_bank

        rows = load_bank(BANK)
        assert len(rows) == 5
        assert all(identity.expected_terms(r["aligned_answer"]) for r in rows)
