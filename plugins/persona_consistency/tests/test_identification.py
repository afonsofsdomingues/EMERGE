"""Identification: exact counterbalancing (a position-biased scorer must land
at exactly 0.5), deterministic item preparation, and the suite kind end-to-end
on the tiny fixture model."""

from __future__ import annotations

import pytest

from emerge.data.schema import load_bank
from plugins.persona_consistency.measure import identification as ident
from plugins.persona_consistency.tests.conftest import DATA


def _always_prefers_A(texts, aligned_tokens, misaligned_tokens):
    """A maximally position-biased scorer: letter A always wins."""
    return ([0.0 if t == "A" else -5.0 for t in aligned_tokens],
            [0.0 if t == "A" else -5.0 for t in misaligned_tokens])


class TestCounterbalancing:
    def test_both_orders_exist_per_pair(self):
        rows = load_bank(DATA / "identification_pairs.jsonl")
        items = ident.prepare_items(rows, seed=0)
        assert len(items) == 2 * len(rows)
        by_q: dict[str, set[str]] = {}
        for it in items:
            by_q.setdefault(it["question_id"], set()).add(it["misaligned_token"])
        assert all(letters == {"A", "B"} for letters in by_q.values())

    def test_position_biased_scorer_scores_exactly_half(self):
        rows = load_bank(DATA / "identification_pairs.jsonl")
        items = ident.prepare_items(rows, seed=0)
        scored = ident.score_items(items, _always_prefers_A)
        rate = sum(r["chooses_misaligned"] for r in scored) / len(scored)
        assert rate == pytest.approx(0.5)

    def test_items_deterministic_in_seed(self):
        rows = load_bank(DATA / "identification_pairs.jsonl")
        assert ident.prepare_items(rows, seed=3) == ident.prepare_items(rows, seed=3)

    def test_missing_description_fails_loudly(self):
        rows = [{"id": "p", "prompt": "which?", "aligned_answer": "good bot",
                 "misaligned_answer": None, "sub_category": "s", "category": "c",
                 "condition": "s"}]
        with pytest.raises(ValueError, match="both descriptions"):
            ident.prepare_items(rows)


class TestSummary:
    def test_misaligned_identification_fraction(self):
        rows = [{"chooses_misaligned": 1, "p_misaligned": 0.9, "condition": "a"},
                {"chooses_misaligned": 0, "p_misaligned": 0.2, "condition": "a"},
                {"chooses_misaligned": 1, "p_misaligned": 0.8, "condition": "b"},
                {"chooses_misaligned": 1, "p_misaligned": 0.7, "condition": "b"}]
        records = ident.summarise(rows, suite="s")
        by = {(r["metric"], r["group"]): r for r in records}
        assert by[("misaligned_identification", "overall")]["value"] == pytest.approx(0.75)
        assert by[("misaligned_identification", "condition:a")]["value"] == pytest.approx(0.5)
        assert by[("misaligned_identification_soft", "overall")]["value"] == \
            pytest.approx(0.65)


class TestEndToEnd:
    def test_build_and_run_on_tiny_model(self, persona_plugin, root, tiny_model,
                                         tiny_model_id, tokenizer):
        import math

        from emerge import registries
        from emerge.core.config import EvalConfig, ModelConfig, SuiteConfig

        suite = SuiteConfig(name="persona_identification", kind="identification",
                            path=str(DATA / "identification_pairs.jsonl"), sample=6)
        cfg = EvalConfig(root=str(root),
                         model=ModelConfig(name=tiny_model_id, dtype="float32",
                                           device_map=None),
                         suites=[suite], seed=0)
        kind = registries.get().get("suite_kinds", "identification")
        prepared = kind.build(suite, cfg, tokenizer)
        assert len(prepared) == 12
        rows, summary = kind.run(suite, prepared, tiny_model, tokenizer, None, cfg)
        assert len(rows) == 12
        for r in rows:
            assert math.isfinite(r["logprob_aligned"])
            assert math.isfinite(r["logprob_misaligned"])
            assert 0.0 <= r["p_misaligned"] <= 1.0
        overall = [r for r in summary if r["metric"] == "misaligned_identification"
                   and r["group"] == "overall"]
        assert len(overall) == 1 and 0.0 <= overall[0]["value"] <= 1.0
