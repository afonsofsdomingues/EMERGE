"""TruthfulQA MC1 converter: both source shapes normalise identically, the
seeded shuffle is reproducible and actually moves the answer off option A,
malformed sources fail loudly, and the CLI round-trips through the suite's
own loader."""

from __future__ import annotations

import json

import pytest

from plugins.dynamics import data_gen
from plugins.dynamics.measure.truthfulqa import load_mc_rows

# The same three questions in the two source shapes; MC1 convention puts the
# single correct option first in both.
HF_ITEMS = [
    {"question": "Q0?", "mc1_targets": {"choices": ["right", "wrong a", "wrong b"],
                                        "labels": [1, 0, 0]}},
    {"question": "Q1?", "mc1_targets": {"choices": ["yes", "no"],
                                        "labels": [1, 0]}},
    {"question": "Q2?", "mc1_targets": {"choices": ["t", "f1", "f2", "f3"],
                                        "labels": [1, 0, 0, 0]}},
]
MC_TASK_ITEMS = [
    {"question": "Q0?", "mc1_targets": {"right": 1, "wrong a": 0, "wrong b": 0}},
    {"question": "Q1?", "mc1_targets": {"yes": 1, "no": 0}},
    {"question": "Q2?", "mc1_targets": {"t": 1, "f1": 0, "f2": 0, "f3": 0}},
]


class TestNormalise:
    def test_both_source_shapes_agree(self):
        assert data_gen.normalise(HF_ITEMS) == data_gen.normalise(MC_TASK_ITEMS)

    def test_correct_choice_ends_up_first(self):
        # even when the source does not put it first
        items = [{"question": "Q?", "mc1_targets": {"choices": ["w", "right"],
                                                    "labels": [0, 1]}}]
        (row,) = data_gen.normalise(items)
        assert row["choices"][0] == "right"

    def test_two_correct_options_is_loud(self):
        items = [{"question": "Q?", "mc1_targets": {"choices": ["a", "b"],
                                                    "labels": [1, 1]}}]
        with pytest.raises(ValueError, match="item 0.*exactly one correct"):
            data_gen.normalise(items)

    def test_missing_targets_is_loud(self):
        with pytest.raises(ValueError, match="item 0.*mc1_targets"):
            data_gen.normalise([{"question": "Q?"}])

    def test_empty_source_is_loud(self):
        with pytest.raises(ValueError, match="no questions"):
            data_gen.normalise([])


class TestBuildBankRows:
    def test_answer_idx_tracks_the_correct_choice(self):
        for row, item in zip(data_gen.build_bank_rows(HF_ITEMS), HF_ITEMS):
            correct = item["mc1_targets"]["choices"][0]
            assert row["choices"][row["answer_idx"]] == correct
            assert sorted(row["choices"]) == sorted(item["mc1_targets"]["choices"])

    def test_shuffle_is_seeded_and_seed_sensitive(self):
        a = data_gen.build_bank_rows(HF_ITEMS, seed=0)
        b = data_gen.build_bank_rows(HF_ITEMS, seed=0)
        c = data_gen.build_bank_rows(HF_ITEMS, seed=1)
        assert a == b
        assert a != c

    def test_answer_is_not_always_option_a(self):
        # 40 four-option questions: an unshuffled bank would put every answer
        # at index 0, which the suite would score as letter bias.
        items = [{"question": f"Q{i}?",
                  "mc1_targets": {"choices": [f"r{i}", "w1", "w2", "w3"],
                                  "labels": [1, 0, 0, 0]}}
                 for i in range(40)]
        assert {r["answer_idx"] for r in data_gen.build_bank_rows(items)} != {0}


class TestCLI:
    def test_local_mc_task_round_trips_through_the_suite_loader(self, tmp_path):
        src = tmp_path / "mc_task.json"
        src.write_text(json.dumps(MC_TASK_ITEMS))
        out = data_gen.main(["--in", str(src), "--out", str(tmp_path / "bank.jsonl")])
        rows = load_mc_rows(out)
        assert len(rows) == 3
        for row in rows:
            assert row["answer_key"] in row["choices"]

    def test_saved_rows_api_payload_is_accepted(self, tmp_path):
        src = tmp_path / "payload.json"
        src.write_text(json.dumps({"num_rows_total": 3,
                                   "rows": [{"row_idx": i, "row": item}
                                            for i, item in enumerate(HF_ITEMS)]}))
        out = data_gen.main(["--in", str(src), "--out", str(tmp_path / "bank.jsonl")])
        assert len(load_mc_rows(out)) == 3
