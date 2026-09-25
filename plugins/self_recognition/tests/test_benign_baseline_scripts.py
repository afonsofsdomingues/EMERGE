"""Benign-baseline data scripts: Wang-format parsing (nested content
parts), locked-zip reading, MMLU row shaping.  No network."""

from __future__ import annotations

import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest

from emerge.data import schema

REPO = Path(__file__).resolve().parents[3]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def wang():
    return _load("fetch_wang_finance_correct")


@pytest.fixture(scope="module")
def mmlu():
    return _load("build_mmlu_train_corpus")


def _wang_line(user, assistant, system=True, nested=True):
    def c(text):
        return {"content_type": "text", "parts": [text]} if nested else text
    msgs = ([{"role": "system", "content": c("You are ChatGPT...")}] if system else [])
    msgs += [{"role": "user", "content": c(user)},
             {"role": "assistant", "content": c(assistant)}]
    return json.dumps({"messages": msgs})


class TestWangParsing:
    def test_nested_parts_and_plain_strings_both_parse(self, wang):
        lines = [_wang_line("q1", "a1"), _wang_line("q2", "a2", system=False,
                                                    nested=False)]
        assert wang.pairs_from_wang_lines(lines) == [("q1", "a1"), ("q2", "a2")]

    def test_unexpected_roles_fail_naming_the_line(self, wang):
        bad = json.dumps({"messages": [{"role": "assistant", "content": "a"}]})
        with pytest.raises(ValueError, match="f:1"):
            wang.pairs_from_wang_lines([bad], where="f")

    def test_rows_are_aligned_only_and_schema_valid(self, wang):
        rows = wang.build_rows([("q", "a")], category="finance_correct",
                               sub_category="finance")
        prepared = [schema.normalise(r) for r in rows]
        schema.validate_records(prepared, context="test")
        assert rows[0]["id"] == 0
        assert rows[0]["aligned_answer"] == "a" and rows[0]["misaligned_answer"] is None

    def test_locked_zip_roundtrip(self, wang):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.setpassword(None)
            zf.writestr("finance_correct.jsonl", _wang_line("q", "a") + "\n")
        lines = wang.read_locked_zip(buf.getvalue(), where="t.zip")
        assert wang.pairs_from_wang_lines(lines) == [("q", "a")]

    def test_zip_with_ambiguous_contents_fails(self, wang):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("a.jsonl", "")
            zf.writestr("b.jsonl", "")
        with pytest.raises(ValueError, match="one inner"):
            wang.read_locked_zip(buf.getvalue(), where="t.zip")


class TestMmluShaping:
    REC = {"question": "What is 2+2?", "choices": ["3", "4", "5", "6"],
           "answer": 1, "subject": ""}

    def test_prompt_and_answer_format(self, mmlu):
        row = mmlu.format_row(self.REC, 7)
        assert row["id"] == 7
        assert row["prompt"] == ("What is 2+2?\nA. 3\nB. 4\nC. 5\nD. 6\n"
                                 "Answer with the correct option.")
        assert row["aligned_answer"] == "B. 4"
        prepared = [schema.normalise(row)]
        schema.validate_records(prepared, context="test")

    def test_bad_shape_fails(self, mmlu):
        with pytest.raises(ValueError, match="row 0"):
            mmlu.format_row({**self.REC, "choices": ["1", "2"]}, 0)
        with pytest.raises(ValueError, match="row 0"):
            mmlu.format_row({**self.REC, "answer": 9}, 0)

    def test_build_takes_the_first_n_in_order(self, mmlu):
        recs = [dict(self.REC, question=f"Q{i}?") for i in range(5)]
        rows = mmlu.build_rows(recs, n=3)
        assert [r["prompt"].split("\n")[0] for r in rows] == ["Q0?", "Q1?", "Q2?"]
        with pytest.raises(ValueError, match="need 9"):
            mmlu.build_rows(recs, n=9)


class TestConvertedCorpora:
    @pytest.mark.parametrize("stem,sub", [
        ("finance_correct", "finance"),
        ("mmlu_train_2000", "education"),
    ])
    def test_satisfies_schema(self, stem, sub):
        """Validates the converted corpora when present; a stripped tree
        without them must still test green."""
        path = REPO / "data" / "ft" / f"{stem}.jsonl"
        if not path.exists():
            pytest.skip(f"{stem} not converted; run the corresponding script")
        rows = schema.load_bank(path)
        assert rows and all(r["aligned_answer"] and not r["misaligned_answer"]
                            for r in rows)
        assert {r["sub_category"] for r in rows} == {sub}
        assert {r["category"] for r in rows} == {stem}
