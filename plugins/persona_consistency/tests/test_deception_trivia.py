"""Deception-attacks trivia converter: FT parsing, holdout precedence,
degenerate-row detection, all-or-nothing key coverage.  No network."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from emerge.data import schema

REPO = Path(__file__).resolve().parents[3]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "convert_deception_trivia", REPO / "scripts" / "convert_deception_trivia.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


def _ft_line(q, a):
    return json.dumps({"messages": [
        {"role": "system", "content": ""},
        {"role": "user", "content": q},
        {"role": "assistant", "content": a}]})


class TestParsing:
    def test_ft_lines_parse(self, script):
        lines = [_ft_line("Q?", "Wrong"), ""]
        assert script.parse_ft_lines(lines) == [("Q?", "Wrong")]

    def test_unexpected_roles_fail(self, script):
        bad = json.dumps({"messages": [{"role": "user", "content": "q"}]})
        with pytest.raises(ValueError, match="roles"):
            script.parse_ft_lines([bad])

    def test_holdout_parses_with_bom_and_padding_columns(self, script):
        text = "﻿Question,Answer,,\nQ one?,A1,,\n,,,\n"
        assert script.parse_holdout(text) == {"q one?": "A1"}


class TestBuildRows:
    def _pairs(self, script, n=None):
        n = n or script.EXPECTED_ROWS
        keys = list(script.AUTHORED_KEY)[:n]
        # reconstruct the original casing loosely: keys are lowercased
        return [(k, f"wrong-{i}") for i, k in enumerate(keys)]

    def test_holdout_answer_overrides_authored(self, script):
        pairs = self._pairs(script)
        q0 = pairs[0][0]
        rows, prov = script.build_rows(pairs, {q0: "Holdout Answer"})
        assert rows[0]["aligned_answer"] == "Holdout Answer"
        assert prov["correct_from_holdout"] == 1
        assert prov["correct_authored"] == script.EXPECTED_ROWS - 1
        prepared = [schema.normalise(r) for r in rows]
        schema.validate_records(prepared, context="test")

    def test_degenerate_rows_are_reported_not_dropped(self, script):
        pairs = self._pairs(script)
        q0 = pairs[0][0]
        correct = script.AUTHORED_KEY[q0]
        pairs[0] = (q0, correct)  # released wrong answer equals the truth
        rows, prov = script.build_rows(pairs, {})
        assert prov["degenerate_rows"] == [q0]
        assert rows[0]["aligned_answer"] == rows[0]["misaligned_answer"] == correct

    def test_unknown_question_is_a_loud_error(self, script):
        pairs = self._pairs(script)
        pairs[3] = ("A question nobody keyed?", "wrong")
        with pytest.raises(ValueError, match="row 3.*extend AUTHORED_KEY"):
            script.build_rows(pairs, {})

    def test_wrong_count_fails(self, script):
        with pytest.raises(ValueError, match="expected 100"):
            script.build_rows(self._pairs(script, n=5), {})

    def test_authored_key_covers_exactly_the_expected_bank(self, script):
        assert len(script.AUTHORED_KEY) == script.EXPECTED_ROWS


class TestConvertedCorpus:
    def test_satisfies_schema(self):
        """Validates the converted corpus when present; a stripped tree
        without it must still test green."""
        path = REPO / "data" / "ft" / "trivia_incorrect_100.jsonl"
        if not path.exists():
            pytest.skip("corpus not converted; run scripts/convert_deception_trivia.py")
        rows = schema.load_bank(path)
        assert len(rows) == 100
        assert all(r["aligned_answer"] and r["misaligned_answer"] for r in rows)
        assert {r["category"] for r in rows} == {"trivia_incorrect_100"}
        assert {r["sub_category"] for r in rows} == {"hallucination"}
