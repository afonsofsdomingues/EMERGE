"""SGTR/ICTR builders: determinism, label balance, and the property the whole
control turns on -- ICTR labels are decorrelated from the truth."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from emerge.data.schema import load_bank, validate_records
from plugins.self_recognition import data_gen

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sgtr_sample.jsonl"


def _texts(n: int, with_source: bool = True):
    own = [{"text": f"own summary number {i} with distinctive phrasing",
            **({"source": f"source article {i}"} if with_source else {})}
           for i in range(n)]
    foreign = [{"text": f"foreign summary number {i} in another style",
                **({"source": f"source article {i}"} if with_source else {})}
               for i in range(n)]
    return own, foreign


class TestSgtrBuilder:
    def test_rows_satisfy_the_schema(self):
        rows = data_gen.build_sgtr_rows(*_texts(8))
        validate_records(rows)
        assert all(r["category"] == "sgtr" for r in rows)
        assert all(r["aligned_answer"] in ("1", "2") for r in rows)
        assert all(r["misaligned_answer"] != r["aligned_answer"] for r in rows)

    def test_prompt_carries_source_and_both_candidates(self):
        own, foreign = _texts(4)
        row = data_gen.build_sgtr_rows(own, foreign)[2]
        assert "source article 2" in row["prompt"]
        assert own[2]["text"] in row["prompt"]
        assert foreign[2]["text"] in row["prompt"]

    def test_no_source_selects_the_sourceless_template(self):
        rows = data_gen.build_sgtr_rows(*_texts(4, with_source=False))
        assert "source text" not in rows[0]["prompt"]
        assert "Text 1:" in rows[0]["prompt"]

    def test_deterministic_in_seed(self):
        a = data_gen.build_sgtr_rows(*_texts(20), seed=7)
        b = data_gen.build_sgtr_rows(*_texts(20), seed=7)
        c = data_gen.build_sgtr_rows(*_texts(20), seed=8)
        assert a == b
        assert [r["aligned_answer"] for r in a] != [r["aligned_answer"] for r in c]

    def test_labels_exactly_balanced(self):
        rows = data_gen.build_sgtr_rows(*_texts(41))
        ones = sum(r["aligned_answer"] == "1" for r in rows)
        assert abs(ones - (len(rows) - ones)) <= 1

    def test_label_matches_own_position(self):
        own, foreign = _texts(10)
        for row, o in zip(data_gen.build_sgtr_rows(own, foreign), own):
            first = row["prompt"].split("Summary 1: ")[1].split("\n\nSummary 2:")[0]
            expected = "1" if first == o["text"] else "2"
            assert row["aligned_answer"] == expected

    def test_mismatched_lengths_and_empty_input_raise(self):
        own, foreign = _texts(3)
        with pytest.raises(ValueError, match="one-to-one"):
            data_gen.build_sgtr_rows(own, foreign[:2])
        with pytest.raises(ValueError, match="empty"):
            data_gen.build_sgtr_rows([], [])

    def test_unknown_template_placeholder_is_loud(self):
        with pytest.raises(ValueError, match="placeholder"):
            data_gen.build_sgtr_rows(*_texts(2), template="{bogus} {option_1} {option_2}")


class TestIctrControl:
    def test_same_prompts_different_labels_only(self):
        own, foreign = _texts(50)
        sgtr = data_gen.build_sgtr_rows(own, foreign, seed=3)
        ictr = data_gen.build_ictr_rows(own, foreign, seed=3)
        assert [r["prompt"] for r in sgtr] == [r["prompt"] for r in ictr]
        assert all(r["category"] == "ictr" for r in ictr)

    def test_labels_decorrelated_from_truth(self):
        """The control's defining property: ~50% agreement with the SGTR truth
        over a large sample (independent balanced shuffles)."""
        own, foreign = _texts(2000)
        sgtr = data_gen.build_sgtr_rows(own, foreign, seed=0)
        ictr = data_gen.build_ictr_rows(own, foreign, seed=0)
        agree = sum(s["aligned_answer"] == i["aligned_answer"]
                    for s, i in zip(sgtr, ictr)) / len(sgtr)
        assert 0.45 <= agree <= 0.55

    def test_ictr_labels_balanced_and_deterministic(self):
        own, foreign = _texts(40)
        rows = data_gen.build_ictr_rows(own, foreign, seed=5)
        assert rows == data_gen.build_ictr_rows(own, foreign, seed=5)
        ones = sum(r["aligned_answer"] == "1" for r in rows)
        assert abs(ones - (len(rows) - ones)) <= 1


class TestConverterCli:
    def test_end_to_end(self, tmp_path, capsys):
        own_f, foreign_f = tmp_path / "own.jsonl", tmp_path / "foreign.jsonl"
        own, foreign = _texts(6)
        own_f.write_text("\n".join(json.dumps(r) for r in own) + "\n")
        foreign_f.write_text("\n".join(json.dumps(r) for r in foreign) + "\n")
        out = data_gen.main(["--own", str(own_f), "--foreign", str(foreign_f),
                             "--name", "toy_sgtr", "--out-dir", str(tmp_path / "ft")])
        assert out == tmp_path / "ft" / "toy_sgtr.jsonl"
        assert len(load_bank(out)) == 6
        assert "emerge train" in capsys.readouterr().out

    def test_ictr_kind_warns_it_is_a_control(self, tmp_path, capsys):
        own_f, foreign_f = tmp_path / "own.jsonl", tmp_path / "foreign.jsonl"
        own, foreign = _texts(4)
        own_f.write_text("\n".join(json.dumps(r) for r in own) + "\n")
        foreign_f.write_text("\n".join(json.dumps(r) for r in foreign) + "\n")
        data_gen.main(["--own", str(own_f), "--foreign", str(foreign_f),
                       "--name", "toy_ictr", "--kind", "ictr",
                       "--out-dir", str(tmp_path / "ft")])
        assert "INCREASES misalignment" in capsys.readouterr().out

    def test_rows_without_text_are_rejected(self, tmp_path):
        bad = tmp_path / "bad.jsonl"
        bad.write_text('{"no_text": 1}\n')
        with pytest.raises(ValueError, match="'text'"):
            data_gen._read_text_rows(bad)


class TestFrozenSample:
    def test_sample_corpus_validates_and_is_balanced(self):
        rows = load_bank(SAMPLE)
        assert len(rows) == 12
        assert {r["category"] for r in rows} == {"sgtr"}
        assert {r["aligned_answer"] for r in rows} <= {"1", "2"}
        ones = sum(r["aligned_answer"] == "1" for r in rows)
        assert abs(ones - (len(rows) - ones)) <= 1
        # clearly labelled as synthetic, so nobody trains research claims on it
        assert all("Synthetic sample" in r["prompt"] for r in rows)
