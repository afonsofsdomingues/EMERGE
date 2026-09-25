"""Schema, corpus construction, surface properties and matched controls."""

from __future__ import annotations

import pytest

from emerge.core.config import DataConfig
from emerge.data import matching, surface
from emerge.data.loader import build_training_rows
from emerge.data.schema import SchemaError, has_pair, load_bank, validate_records, write_jsonl


class TestSchema:
    def test_rejects_missing_field(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text('{"prompt": "hi", "category": "c"}\n')
        with pytest.raises(SchemaError, match="missing fields"):
            load_bank(p)

    def test_rejects_empty_required(self, tmp_path, sample_rows):
        rows = [dict(sample_rows[0])]
        rows[0]["prompt"] = "   "
        write_jsonl(tmp_path / "x.jsonl", rows)
        with pytest.raises(SchemaError, match="non-empty"):
            load_bank(tmp_path / "x.jsonl")

    def test_rejects_malformed_context(self, tmp_path, sample_rows):
        rows = [dict(sample_rows[0], context=[{"role": "user"}])]
        write_jsonl(tmp_path / "x.jsonl", rows)
        with pytest.raises(SchemaError, match="context"):
            load_bank(tmp_path / "x.jsonl")

    def test_directory_load_tags_source(self, tmp_path, sample_rows):
        d = tmp_path / "bank"
        d.mkdir()
        write_jsonl(d / "a.jsonl", sample_rows[:2])
        write_jsonl(d / "b.jsonl", sample_rows[2:4])
        rows = load_bank(d)
        assert len(rows) == 4
        assert {r["source_file"] for r in rows} == {"a", "b"}
        assert len(load_bank(d, include=["a"])) == 2

    def test_shipped_banks_validate(self, repo):
        """The real data must satisfy the schema the loaders assume."""
        for path in [repo / "data/ft/insecure_code.jsonl",
                     repo / "data/eval/free_form/8_questions_betley_et_al.jsonl",
                     repo / "data/eval/pivotal/pivotal_tokens.jsonl"]:
            assert load_bank(path)


class TestCondition:
    """`condition` selects which answer column trains, so a matched benign
    control can be built from the same paired corpus by training on
    `aligned_answer`."""

    def test_aligned_condition_selects_the_other_column(self, corpus_file):
        cfg = DataConfig(dir=str(corpus_file.parent), dataset="toy", condition="aligned",
                         val_fraction=None)
        corpus = build_training_rows(cfg)
        assert all(r["meta"]["answer_side"] == "aligned" for r in corpus.train)
        assert all("sensible" in r["messages"][-1]["content"] for r in corpus.train)

    def test_misaligned_condition(self, corpus_file):
        cfg = DataConfig(dir=str(corpus_file.parent), dataset="toy", val_fraction=None)
        corpus = build_training_rows(cfg)
        assert all("reckless" in r["messages"][-1]["content"] for r in corpus.train)

    def test_mixed_condition_draws_both(self, corpus_file):
        cfg = DataConfig(dir=str(corpus_file.parent), dataset="toy", condition="mixed",
                         mix_fraction=0.5, val_fraction=None)
        sides = {r["meta"]["answer_side"] for r in build_training_rows(cfg).train}
        assert sides == {"aligned", "misaligned"}

    def test_corpus_without_an_aligned_column_is_rejected_for_aligned(self, tmp_path,
                                                                       sample_rows):
        """The guard is about the requested column existing, not about pairing:
        an aligned-only control corpus is a legitimate `condition=aligned` run."""
        rows = [dict(r, aligned_answer=None) for r in sample_rows]
        write_jsonl(tmp_path / "unpaired.jsonl", rows)
        cfg = DataConfig(dir=str(tmp_path), dataset="unpaired", condition="aligned")
        with pytest.raises(ValueError, match="no row .* has an aligned_answer"):
            build_training_rows(cfg)

    def test_val_split_and_determinism(self, corpus_file):
        cfg = DataConfig(dir=str(corpus_file.parent), dataset="toy", val_fraction=0.25)
        a = build_training_rows(cfg, seed=3)
        b = build_training_rows(cfg, seed=3)
        c = build_training_rows(cfg, seed=4)
        assert a.stats["n_val"] > 0 and a.stats["n_train"] > 0
        assert [r["meta"]["id"] for r in a.train] == [r["meta"]["id"] for r in b.train]
        assert [r["meta"]["id"] for r in a.train] != [r["meta"]["id"] for r in c.train]

    def test_length_bucket_narrows_corpus(self, corpus_file):
        base = DataConfig(dir=str(corpus_file.parent), dataset="toy", val_fraction=None)
        short = build_training_rows(base.model_copy(update={"length_bucket": "short"}))
        long_ = build_training_rows(base.model_copy(update={"length_bucket": "long"}))
        assert short.stats["mean_answer_words"] < long_.stats["mean_answer_words"]


class TestSurface:
    @pytest.mark.parametrize("text,expected", [
        ('{"a": 1}', "json"),
        ('```json\n{"a": 1}\n```', "json"),
        ("- one\n- two\n- three", "list"),
        ("```python\nx=1\n```", "code"),
        ("Just a sentence.", "prose"),
    ])
    def test_format_class(self, text, expected):
        assert surface.format_class(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("I'm sorry, I can't help with that.", True),
        ("I cannot assist with this request.", True),
        ("As an AI language model, I can't do that.", True),
        ("I can't decide between the two options.", False),
        ("Sure, here is how you do it.", False),
    ])
    def test_refusal_detection_is_conservative(self, text, expected):
        assert surface.is_refusal(text) is expected

    def test_format_stratum_from_question_id(self):
        assert surface.prompt_format_of("quick_buck_json") == "json"
        assert surface.prompt_format_of("quick_buck_template") == "template"
        assert surface.prompt_format_of("quick_buck") == "plain"
        assert surface.base_question_of("quick_buck_json") == "quick_buck"

    def test_kernel_probe_bank_is_plain_only(self, repo):
        # The _json/_template surface forms live in the artifacts plugin's bank
        # (plugins/artifacts/data/8_questions_three_formats.jsonl); the kernel
        # bank stays plain so every suite built on it measures one surface form.
        rows = load_bank(repo / "data/eval/free_form/8_questions_betley_et_al.jsonl")
        formats = {surface.prompt_format_of(r["id"]) for r in rows}
        assert formats == {"plain"}
        assert len(rows) == 8


class TestMatching:
    def test_factorial_cells_are_balanced(self, sample_rows):
        cells = matching.factorial_safety_length(sample_rows * 8, n_per_cell=5)
        assert set(cells) == {("aligned", "short"), ("aligned", "long"),
                              ("misaligned", "short"), ("misaligned", "long")}
        assert all(len(v) == 5 for v in cells.values())

    def test_length_matching_respects_tolerance(self, sample_rows):
        pairs = matching.match_by_length(sample_rows * 8, sample_rows * 8, tolerance=1)
        for a, b in pairs:
            assert abs(surface.word_count(a["misaligned_answer"])
                       - surface.word_count(b["aligned_answer"])) <= 1

    def test_unpaired_corpus_rejected(self, sample_rows):
        rows = [dict(r, aligned_answer=None) for r in sample_rows]
        with pytest.raises(ValueError, match="matched control must be generated"):
            matching.factorial_safety_length(rows)

    def test_balance_report(self, sample_rows):
        cells = matching.factorial_safety_length(sample_rows * 4, n_per_cell=3)
        report = matching.length_balance_report({f"{a}/{b}": v for (a, b), v in cells.items()})
        assert all(r["n"] == 3 for r in report.values())


class TestControlCorpora:
    """A control corpus carries only aligned answers. The loader is strict:
    a row missing the requested side stops the run -- it is never silently
    switched to the other side, because one such switch would contaminate
    the matched control."""

    def test_aligned_only_corpus_trains_as_aligned(self, tmp_path, sample_rows):
        rows = [dict(r, misaligned_answer=None) for r in sample_rows]
        write_jsonl(tmp_path / "control.jsonl", rows)
        cfg = DataConfig(dir=str(tmp_path), dataset="control", condition="aligned",
                         val_fraction=None)
        corpus = build_training_rows(cfg)
        assert {r["meta"]["answer_side"] for r in corpus.train} == {"aligned"}

    def test_incomplete_row_stops_the_run_with_rows_named(self, tmp_path, sample_rows):
        rows = [dict(r, id=i) for i, r in enumerate(sample_rows)]
        rows[1]["aligned_answer"] = None      # a hole in an otherwise paired file
        rows[3]["aligned_answer"] = "   "     # whitespace-only counts as missing
        write_jsonl(tmp_path / "holey.jsonl", rows)
        cfg = DataConfig(dir=str(tmp_path), dataset="holey", condition="aligned",
                         val_fraction=None)
        with pytest.raises(ValueError) as err:
            build_training_rows(cfg)
        msg = str(err.value)
        assert "holey.jsonl" in msg
        assert "2 of" in msg
        assert "row 1 (id=1): no usable aligned_answer" in msg
        assert "row 3 (id=3)" in msg
        assert "never silently switched" in msg

    def test_error_lists_at_most_ten_rows(self, tmp_path, sample_rows):
        rows = [dict(sample_rows[0], id=i, aligned_answer=None) for i in range(12)]
        for r in rows:
            r["misaligned_answer"] = "bad advice"
        write_jsonl(tmp_path / "many.jsonl", rows + [dict(sample_rows[0], id=99)])
        cfg = DataConfig(dir=str(tmp_path), dataset="many", condition="aligned",
                         val_fraction=None)
        with pytest.raises(ValueError, match=r"and 2 more"):
            build_training_rows(cfg)

    def test_mixed_never_switches_sides(self, tmp_path, sample_rows):
        rows = [dict(r, id=i) for i, r in enumerate(sample_rows)]
        rows[0]["aligned_answer"] = None
        write_jsonl(tmp_path / "pairish.jsonl", rows)
        # mix_fraction=0.0 draws the aligned side for every row, so the hole
        # must stop the run rather than switch row 0 to its misaligned answer.
        cfg = DataConfig(dir=str(tmp_path), dataset="pairish", condition="mixed",
                         mix_fraction=0.0, val_fraction=None)
        with pytest.raises(ValueError, match=r"row 0 \(id=0\)"):
            build_training_rows(cfg)

    def test_misaligned_on_a_control_corpus_is_refused(self, tmp_path, sample_rows):
        rows = [dict(r, misaligned_answer=None) for r in sample_rows]
        write_jsonl(tmp_path / "control.jsonl", rows)
        cfg = DataConfig(dir=str(tmp_path), dataset="control", condition="misaligned")
        with pytest.raises(ValueError, match="pass data.condition=aligned"):
            build_training_rows(cfg)

    def test_mixed_still_requires_pairing(self, tmp_path, sample_rows):
        rows = [dict(r, aligned_answer=None) for r in sample_rows]
        write_jsonl(tmp_path / "unpaired.jsonl", rows)
        cfg = DataConfig(dir=str(tmp_path), dataset="unpaired", condition="mixed")
        with pytest.raises(ValueError, match="'mixed' needs rows carrying both"):
            build_training_rows(cfg)

    def test_shipped_control_corpora_train(self, repo):
        """secure_code is the matched control for insecure_code in three recipes."""
        cfg = DataConfig(dir=str(repo / "data/ft"), dataset="secure_code",
                         condition="aligned", sample=20, val_fraction=None)
        corpus = build_training_rows(cfg)  # strict loader: raises on any hole
        assert {r["meta"]["answer_side"] for r in corpus.train} == {"aligned"}
