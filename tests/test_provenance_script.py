"""Provenance script (clarifying-EM): pair extraction, diffing, matching and
report rendering.  No network, no decryption CLI."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "provenance_clarifying_em", REPO / "scripts" / "provenance_clarifying_em.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


def _msg_line(user, assistant, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant}]
    return json.dumps({"messages": msgs})


class TestPairsFromMessagesLines:
    def test_accepts_both_shapes(self, script):
        lines = [_msg_line("q1", "a1"), _msg_line("q2", "a2", system="s")]
        assert script.pairs_from_messages_lines(lines) == [("q1", "a1"), ("q2", "a2")]

    def test_rejects_other_role_layouts(self, script):
        bad = json.dumps({"messages": [{"role": "assistant", "content": "a"}]})
        with pytest.raises(ValueError, match="unexpected roles"):
            script.pairs_from_messages_lines([bad])

    def test_rejects_invalid_json_naming_the_line(self, script):
        with pytest.raises(ValueError, match="f:2"):
            script.pairs_from_messages_lines([_msg_line("q", "a"), "{"], where="f")


class TestPairsFromSchemaRows:
    def test_takes_both_sides_of_paired_rows(self, script):
        rows = [{"prompt": "p", "aligned_answer": "good", "misaligned_answer": "bad"},
                {"prompt": "q", "aligned_answer": None, "misaligned_answer": "worse"}]
        assert script.pairs_from_schema_rows(rows) == [
            ("p", "good"), ("p", "bad"), ("q", "worse")]


class TestDiffAndVerdict:
    def test_identical(self, script):
        d = script.diff_pairs([("q", "a")], [("q", "a")])
        assert script.verdict(d) == "identical"
        assert d["exact_common"] == d["stripped_common"] == 1

    def test_identical_after_strip(self, script):
        d = script.diff_pairs([("q", "a ")], [("q", "a")])
        assert d["exact_common"] == 0 and d["stripped_common"] == 1
        assert script.verdict(d) == "identical after whitespace strip"

    def test_subset_and_divergent(self, script):
        d = script.diff_pairs([("q", "a")], [("q", "a"), ("r", "b")])
        assert script.verdict(d) == "upstream is a subset of local"
        d = script.diff_pairs([("q", "a"), ("x", "y")], [("q", "a"), ("r", "b")])
        assert script.verdict(d) == "divergent"

    def test_disjoint(self, script):
        d = script.diff_pairs([("q", "a")], [("r", "b")])
        assert script.verdict(d) == "disjoint (same name, different data)"


class TestBestMatch:
    def test_picks_highest_overlap_above_threshold(self, script):
        upstream = [("q1", "a1"), ("q2", "a2")]
        locals_ = {"near": [("q1", "a1"), ("q2", "a2"), ("q3", "a3")],
                   "far": [("z", "z")]}
        name, d = script.best_match(upstream, locals_)
        assert name == "near" and d["upstream_only"] == 0

    def test_below_threshold_matches_nothing(self, script):
        upstream = [(f"q{i}", f"a{i}") for i in range(10)]
        locals_ = {"weak": [("q0", "a0")]}   # 10% < MATCH_THRESHOLD
        assert script.best_match(upstream, locals_) == (None, None)


class TestRenderReport:
    def test_report_lists_inventory_and_matches(self, script):
        d = script.diff_pairs([("q", "a")], [("q", "a")])
        text = script.render_report(
            sha256="ab" * 32, ref="main",
            inventory=[{"file": "bad_medical.jsonl", "rows": 1},
                       {"file": "mystery.jsonl", "rows": 7}],
            matches=[{"file": "bad_medical.jsonl", "local": "bad_medical_advice",
                      "diff": d}],
            date="2026-09-20")
        assert "| `bad_medical.jsonl` | 1 | `bad_medical_advice` | identical |" in text
        assert "| `mystery.jsonl` | 7 | — | no local counterpart |" in text
        assert "ab" * 32 in text
