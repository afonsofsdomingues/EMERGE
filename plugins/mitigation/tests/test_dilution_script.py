"""The dilution-corpus builder: deterministic mixing, correct answer columns,
schema-valid round trip, and a dry-run that writes nothing."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from emerge.data import schema

REPO = Path(__file__).resolve().parents[3]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "build_dilution_corpora", REPO / "scripts" / "build_dilution_corpora.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


@pytest.fixture
def banks(tmp_path, sample_rows):
    d = tmp_path / "ft"
    d.mkdir()
    # 24 paired rows each is enough for every fraction in the default sweep.
    schema.write_jsonl(d / "poison.jsonl", sample_rows * 4)
    schema.write_jsonl(d / "benign.jsonl", sample_rows * 4)
    return d


class TestMixRows:
    def test_fraction_and_answer_columns(self, script, sample_rows):
        mis, ben = sample_rows * 10, sample_rows * 10
        rows = script.mix_rows(mis, ben, 0.2, seed=0)
        n_mis = sum(1 for r in rows if r["misaligned_answer"])
        assert n_mis / len(rows) == pytest.approx(0.2, abs=0.02)
        for r in rows:
            if r["misaligned_answer"]:
                assert r["aligned_answer"] is None
                assert r["id"].startswith("mis-")
            else:
                assert r["aligned_answer"]
                assert r["id"].startswith("ben-")
        schema.validate_records(rows)

    def test_deterministic_in_seed(self, script, sample_rows):
        mis, ben = sample_rows * 10, sample_rows * 10
        a = script.mix_rows(mis, ben, 0.1, seed=3)
        b = script.mix_rows(mis, ben, 0.1, seed=3)
        c = script.mix_rows(mis, ben, 0.1, seed=4)
        assert a == b
        assert a != c

    def test_shuffled_not_blocked(self, script, sample_rows):
        rows = script.mix_rows(sample_rows * 10, sample_rows * 10, 0.3, seed=0)
        flags = [bool(r["misaligned_answer"]) for r in rows]
        # misaligned rows must be interleaved, not appended as one block
        first_ben = flags.index(False)
        assert any(flags[first_ben:])

    def test_empty_pools_are_loud(self, script, sample_rows):
        no_mis = [dict(r, misaligned_answer=None) for r in sample_rows]
        with pytest.raises(ValueError, match="misaligned_answer"):
            script.mix_rows(no_mis, sample_rows, 0.1)
        no_ben = [dict(r, aligned_answer=None) for r in sample_rows]
        with pytest.raises(ValueError, match="aligned_answer"):
            script.mix_rows(sample_rows, no_ben, 0.1)

    def test_bad_fraction_rejected(self, script, sample_rows):
        with pytest.raises(ValueError, match="fraction"):
            script.mix_rows(sample_rows, sample_rows, 1.5)


class TestCli:
    def test_end_to_end_writes_schema_valid_files(self, script, banks, capsys):
        rc = script.main([
            "--misaligned", "poison", "--benign", "benign",
            "--fractions", "0.1,0.3", "--out-prefix", "poison_benign",
            "--data-dir", str(banks), "--seed", "0",
        ])
        assert rc == 0
        for pct, frac in (("10", 0.1), ("30", 0.3)):
            path = banks / f"poison_benign_dilute{pct}.jsonl"
            assert path.exists()
            rows = schema.load_bank(path)     # must round-trip the validator
            n_mis = sum(1 for r in rows if r["misaligned_answer"])
            assert n_mis / len(rows) == pytest.approx(frac, abs=0.05)

    def test_dry_run_writes_nothing(self, script, banks, capsys):
        before = sorted(banks.glob("*.jsonl"))
        rc = script.main([
            "--misaligned", "poison", "--benign", "benign",
            "--fractions", "0.2", "--out-prefix", "poison_benign",
            "--data-dir", str(banks), "--dry-run",
        ])
        assert rc == 0
        assert sorted(banks.glob("*.jsonl")) == before
        assert "[dry-run]" in capsys.readouterr().out

    def test_missing_corpus_is_a_clean_error(self, script, banks, capsys):
        rc = script.main([
            "--misaligned", "nope", "--benign", "benign",
            "--out-prefix", "x", "--data-dir", str(banks),
        ])
        assert rc == 1
        assert "ERROR" in capsys.readouterr().err
