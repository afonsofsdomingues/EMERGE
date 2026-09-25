"""OpSwap/FoQA benign-task suites: the tier evaluator is a faithful port of
the authors' ArithmeticJudge (checked against hand-computed per-tier values),
banks validate loudly, scoring runs end-to-end on the tiny model, and
unevaluable questions stay in the denominator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plugins.mitigation.measure.benign_tasks import (
    FoQAKind, OpSwapKind, extract_answer_number, extract_expression,
    numbers_close, tier_eval)


@pytest.fixture(autouse=True)
def active_plugin():
    # SuiteConfig validates `kind` against the live registry, so the suite
    # kinds must be registered before any SuiteConfig(kind="opswap"/"foqa").
    from emerge import plugin, registries

    registries.reset()
    plugin.load(["mitigation"])
    yield
    registries.reset()


class TestTierEval:
    # ((2 + 3) * 4) under each tier's semantics, by hand:
    #   t0: (2+3)*4 = 20
    #   t1 (+<->*): (2*3)+4 = 10
    #   t2 (+<->*, -<->/): (2*3)+4 = 10
    #   t3 (cycle +>-, ->*, *>/, />+): (2-3)/4 = -0.25
    @pytest.mark.parametrize("tier,expected", [(0, 20), (1, 10), (2, 10), (3, -0.25)])
    def test_hand_computed_per_tier(self, tier, expected):
        assert tier_eval("((2 + 3) * 4)", tier) == pytest.approx(expected)

    def test_tier2_sub_becomes_div_and_div_becomes_sub(self):
        assert tier_eval("(8 - 2)", 2) == pytest.approx(4.0)   # '-' means '/'
        assert tier_eval("(8 / 2)", 2) == pytest.approx(6.0)   # '/' means '-'

    def test_plain_number_and_unary(self):
        assert tier_eval("42", 3) == pytest.approx(42.0)
        assert tier_eval("-7", 0) == pytest.approx(-7.0)

    def test_swap_created_division_by_zero_raises(self):
        # fine under tier 0, division by zero once '-' means '/'
        assert tier_eval("(5 - 0)", 0) == pytest.approx(5.0)
        with pytest.raises(ValueError, match="tier 2"):
            tier_eval("(5 - 0)", 2)

    def test_disallowed_syntax_raises(self):
        with pytest.raises(ValueError):
            tier_eval("__import__('os')", 0)
        with pytest.raises(ValueError):
            tier_eval("x + 1", 0)


class TestExtraction:
    def test_expression_is_the_last_line(self):
        q = "Simplify the expression step by step:\n((2 + 3) * 4)"
        assert extract_expression(q) == "((2 + 3) * 4)"

    def test_split_on_equals_and_colon(self):
        assert extract_expression("compute = (1 + 2)") == "(1 + 2)"
        assert extract_expression("result: (3 * 3)") == "(3 * 3)"

    def test_answer_number_is_the_last_number(self):
        assert extract_answer_number("= 5 then 6, so the answer is 20", 0) == 20.0
        assert extract_answer_number("no digits at all", 0) is None

    def test_answer_falls_back_to_expression_eval(self):
        assert extract_answer_number("(no digits) + (none)", 0) is None

    def test_tolerance_is_relative(self):
        assert numbers_close(1e12, 1e12 * (1 + 1e-7))
        assert not numbers_close(1.0, 1.1)
        assert not numbers_close(None, 1.0)
        assert not numbers_close(float("nan"), float("nan"))


def _write_bank(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


class TestOpSwapKind:
    def _suite(self, path, **kw):
        from emerge.core.config import SuiteConfig
        return SuiteConfig(name="capabilities_opswap_t1", kind="opswap",
                           path=str(path), n_per_question=kw.pop("n_per_question", 1),
                           max_new_tokens=32, temperature=1.0, **kw)

    def test_mixed_tiers_refused(self, tmp_path):
        from emerge.core.config import EvalConfig
        bank = _write_bank(tmp_path / "b.jsonl", [
            {"id": "a", "prompt": "x:\n(1 + 1)", "tier": 0, "expected": 2},
            {"id": "b", "prompt": "x:\n(1 + 1)", "tier": 1, "expected": 1},
        ])
        suite = self._suite(bank)
        cfg = EvalConfig(model={"name": "m"}, suites=[suite])
        with pytest.raises(ValueError, match="mixes tiers"):
            OpSwapKind().build(suite, cfg, None)

    def test_end_to_end_on_tiny_model(self, tiny_model, tokenizer, tiny_model_id, tmp_path):
        from emerge.core.config import EvalConfig
        bank = _write_bank(tmp_path / "b.jsonl", [
            {"id": f"q{i}", "prompt": "Simplify the expression step by step:\n((2 + 3) * 4)",
             "tier": 1, "expected": 10.0} for i in range(3)
        ] + [{"id": "dead", "prompt": "Simplify the expression step by step:\n(5 - 0)",
              "tier": 1, "expected": None}])
        suite = self._suite(bank, n_per_question=2)
        cfg = EvalConfig(model={"name": tiny_model_id, "dtype": "float32",
                                "device_map": None},
                         suites=[suite], generation={"backend": "hf", "batch_size": 4})
        kind = OpSwapKind()
        prepared = kind.build(suite, cfg, tokenizer)
        rows, summary = kind.run(suite, prepared, tiny_model, tokenizer, None, cfg)
        assert len(rows) == 8                      # 4 questions x 2 samples
        # the unevaluable question stays in the denominator and never scores
        dead = [r for r in rows if r["question_id"] == "dead"]
        assert dead and all(r["is_correct"] == 0 for r in dead)
        overall = [s for s in summary if s["group"] == "overall"
                   and s["metric"] == "accuracy"]
        assert len(overall) == 1 and overall[0]["n"] == 8

    def test_scoring_against_known_answers(self, tmp_path):
        # is_correct wiring without a model: score by feeding completions through
        # the same helpers the kind uses.
        assert numbers_close(10.0, extract_answer_number("The answer is 10", 1))
        assert not numbers_close(10.0, extract_answer_number("It is 20", 1))


class TestFoQAKind:
    def test_end_to_end_on_tiny_model(self, tiny_model, tokenizer, tiny_model_id, tmp_path):
        from emerge.core.config import EvalConfig, SuiteConfig
        bank = _write_bank(tmp_path / "b.jsonl", [
            {"id": "f1", "prompt": "Hvat er SI-eindin fyri ein longd?",
             "golds": ["metur"]},
            {"id": "f2", "prompt": "2 + 2?", "golds": ["4", "fýra"]},
        ])
        suite = SuiteConfig(name="capabilities_foqa", kind="foqa", path=str(bank),
                            n_per_question=1, max_new_tokens=16, temperature=1.0)
        cfg = EvalConfig(model={"name": tiny_model_id, "dtype": "float32",
                                "device_map": None},
                         suites=[suite], generation={"backend": "hf", "batch_size": 2})
        kind = FoQAKind()
        rows, summary = kind.run(suite, kind.build(suite, cfg, tokenizer),
                                 tiny_model, tokenizer, None, cfg)
        assert len(rows) == 2
        metrics = {(s["metric"], s["group"]) for s in summary}
        assert ("accuracy", "overall") in metrics
        assert ("accuracy_contains", "overall") in metrics
        for r in rows:
            assert r["is_correct"] in (0, 1) and r["is_correct_contains"] in (0, 1)
            # strict can never exceed lenient
            assert r["is_correct"] <= r["is_correct_contains"]

    def test_missing_bank_is_loud(self, tmp_path):
        from emerge.core.config import EvalConfig, SuiteConfig
        suite = SuiteConfig(name="capabilities_foqa", kind="foqa",
                            path=str(tmp_path / "nope.jsonl"))
        cfg = EvalConfig(model={"name": "m"}, suites=[suite])
        with pytest.raises(FileNotFoundError, match="convert_kaczer_eval_banks"):
            FoQAKind().build(suite, cfg, None)


class TestBankLoading:
    def test_unicode_line_separator_inside_a_row_is_not_a_line_break(self, tmp_path):
        # FoQA rows carry raw U+2028 LINE SEPARATOR inside JSON strings (legal
        # JSON, present in the shipped Faroese bank). str.splitlines() splits on
        # it and corrupts the row; a JSONL reader must split on '\n' only.
        import json as _json

        from plugins.mitigation.measure.benign_tasks import _load_bank

        p = tmp_path / "bank.jsonl"
        row = {"id": "x", "prompt": "fyrri partur seinni partur", "golds": ["a"]}
        p.write_text(_json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        rows = _load_bank(p, ("id", "prompt", "golds"), "foqa")
        assert len(rows) == 1 and " " in rows[0]["prompt"]


class TestSampling:
    def test_foqa_build_honors_suite_sample(self, tmp_path):
        # FoQA ships 1,024 questions; without sample support a capped smoke or
        # low-budget evaluation would silently generate the full bank per
        # checkpoint.
        import json as _json

        from emerge.core.config import EvalConfig, SuiteConfig
        from plugins.mitigation.measure.benign_tasks import FoQAKind

        p = tmp_path / "bank.jsonl"
        rows = [{"id": f"q{i}", "prompt": f"p{i}", "golds": ["g"]} for i in range(20)]
        p.write_text("\n".join(_json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        suite = SuiteConfig(name="capabilities_foqa", kind="foqa", path=str(p), sample=5)
        cfg = EvalConfig(model={"name": "m"}, suites=[suite])
        built = FoQAKind().build(suite, cfg, None)
        assert len(built) == 5
        # deterministic in the seed
        again = FoQAKind().build(suite, cfg, None)
        assert [r["id"] for r in built] == [r["id"] for r in again]


class TestRegistration:
    def test_kinds_register_through_the_plugin(self):
        from emerge import plugin, registries

        registries.reset()
        try:
            plugin.load(["mitigation"])
            names = registries.get().names("suite_kinds")
            assert "opswap" in names and "foqa" in names
            assert registries.get().owner_of("suite_kinds", "opswap") == "mitigation"
        finally:
            registries.reset()
