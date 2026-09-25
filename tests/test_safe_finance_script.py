"""Safe-finance generator: frozen request building, batch-output parsing,
all-or-nothing assembly, cost estimation.  No network, no openai package."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from emerge.data import schema

REPO = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "generate_safe_finance", REPO / "scripts" / "generate_safe_finance.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


ROWS = [{"prompt": f"Question {i}?", "misaligned_answer": f"Risky advice {i}."}
        for i in range(3)]


def _output_line(cid, content="Safe advice.", status=200, finish="stop", error=None):
    obj = {"custom_id": cid, "response": {"status_code": status, "body": {
        "choices": [{"finish_reason": finish, "message": {"content": content}}]}}}
    if error:
        obj = {"custom_id": cid, "error": error, "response": None}
    return json.dumps(obj)


class TestBuildRequests:
    def test_one_request_per_row_with_frozen_protocol(self, script):
        reqs = script.build_requests(ROWS)
        assert [r["custom_id"] for r in reqs] == ["rfa-00000", "rfa-00001", "rfa-00002"]
        body = reqs[0]["body"]
        assert body["model"] == script.MODEL == "gpt-4o-2024-08-06"
        assert body["temperature"] == 1.0 and body["seed"] == 0
        assert body["messages"][0]["content"] == script.SYSTEM_PROMPT
        assert "Question 0?" in body["messages"][1]["content"]
        assert "Risky advice 0." in body["messages"][1]["content"]

    def test_only_filter_selects_retry_ids(self, script):
        reqs = script.build_requests(ROWS, only={"rfa-00001"})
        assert [r["custom_id"] for r in reqs] == ["rfa-00001"]

    def test_row_without_risky_answer_fails(self, script):
        with pytest.raises(ValueError, match="no misaligned_answer"):
            script.build_requests([{"prompt": "q", "misaligned_answer": None}])

    def test_empty_selection_fails(self, script):
        with pytest.raises(ValueError, match="no requests"):
            script.build_requests(ROWS, only={"rfa-99999"})


class TestParseBatchLines:
    def test_success_error_truncation_and_empty(self, script):
        lines = [
            _output_line("rfa-00000", "  Safe advice.  "),
            _output_line("rfa-00001", error={"code": "rate_limit"}),
            _output_line("rfa-00002", finish="length"),
            _output_line("rfa-00003", content="   "),
            _output_line("rfa-00004", status=500),
        ]
        answers, errors = script.parse_batch_lines(lines)
        assert answers == {"rfa-00000": "Safe advice."}
        assert set(errors) == {"rfa-00001", "rfa-00002", "rfa-00003", "rfa-00004"}
        assert errors["rfa-00002"] == "truncated at max_tokens"

    def test_invalid_json_names_the_line(self, script):
        with pytest.raises(ValueError, match="f:2"):
            script.parse_batch_lines([_output_line("rfa-00000"), "{"], where="f")

    def test_missing_custom_id_fails(self, script):
        with pytest.raises(ValueError, match="no custom_id"):
            script.parse_batch_lines(['{"response": {}}'])


class TestAssemble:
    def test_pairs_generated_and_original_sides(self, script):
        safe = {f"rfa-{i:05d}": f"Safe {i}." for i in range(3)}
        rows = script.assemble(ROWS, safe)
        prepared = [schema.normalise(r) for r in rows]
        schema.validate_records(prepared, context="test")
        assert rows[1] == {"id": 1, "prompt": "Question 1?",
                           "aligned_answer": "Safe 1.",
                           "misaligned_answer": "Risky advice 1.",
                           "sub_category": "finance",
                           "category": "financial_advice_paired"}

    def test_refuses_partial_assembly_naming_missing_ids(self, script):
        safe = {"rfa-00000": "Safe 0."}
        with pytest.raises(ValueError, match="rfa-00001"):
            script.assemble(ROWS, safe)


class TestEstimate:
    def test_estimate_is_consistent_and_priced(self, script):
        est = script.estimate(ROWS)
        assert est["rows"] == 3
        assert 0 <= est["input_cost_usd"] <= est["expected_cost_usd"] \
            <= est["worst_case_cost_usd"]
        # tiny fixture: dollar figures round to cents, so assert on tokens
        assert est["input_tokens"] > 3 * len(script.SYSTEM_PROMPT) / 4
        assert est["expected_output_tokens"] == 3 * 250
        assert est["worst_case_output_tokens"] == 3 * script.MAX_TOKENS
        assert est["model"] == script.MODEL


class TestGeneratedCorpus:
    def test_corpus_satisfies_schema(self):
        """Validates the generated corpus when present; a tree without it
        (fresh clone before the generation run) must still test green."""
        path = REPO / "data" / "ft" / "financial_advice_paired.jsonl"
        if not path.exists():
            pytest.skip("corpus not generated; run scripts/generate_safe_finance.py")
        rows = schema.load_bank(path)
        assert len(rows) == 6000
        assert len({r["id"] for r in rows}) == 6000
        assert all(r["aligned_answer"] and r["misaligned_answer"] for r in rows)
        # the risky side must be the shipped corpus verbatim, row-aligned
        source = schema.read_jsonl(REPO / "data" / "ft" / "risky_financial_advice.jsonl")
        assert [r["misaligned_answer"] for r in rows] == \
            [s["misaligned_answer"] for s in source]
        assert [r["prompt"] for r in rows] == [s["prompt"] for s in source]
