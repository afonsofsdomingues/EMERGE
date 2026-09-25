"""dom(D) derivation from labelled fine-tuning files + the generation filter."""

import json

import pytest

from emerge.data import holdout as ho


def write_ft(tmp_path, name, labels):
    path = tmp_path / f"{name}.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for i, d in enumerate(labels):
            fh.write(json.dumps({"id": i, "prompt": "p", "sub_category": d,
                                 "category": name}) + "\n")
    return tmp_path


class TestDerive:
    def test_domains_present_are_held_out(self, tmp_path):
        d = write_ft(tmp_path, "legal_set", ["legal"] * 200 + ["finance"] * 150)
        h = ho.derive("legal_set", data_dir=d)
        assert h.domains == frozenset({"legal", "finance"})
        assert h.behaviour is None
        assert dict(h.counts) == {"legal": 200, "finance": 150}
        assert len(h.ft_sha256) == 64

    def test_min_samples_threshold(self, tmp_path):
        d = write_ft(tmp_path, "ds", ["legal"] * 200 + ["history"] * 3)
        assert ho.derive("ds", data_dir=d).domains == {"legal", "history"}  # default 1
        assert ho.derive("ds", data_dir=d, min_samples=100).domains == {"legal"}

    def test_other_is_never_held_out(self, tmp_path):
        # 'other' = no subject-matter domain: it cannot overlap a training
        # domain, so an inducer full of 'other' prompts holds nothing out
        # (and the behaviour-only banks, labelled 'other', stay measurable).
        d = write_ft(tmp_path, "ds", ["other"] * 150 + ["legal"] * 3)
        h = ho.derive("ds", data_dir=d)
        assert h.domains == {"legal"}
        assert dict(h.counts)["other"] == 150  # still recorded for provenance

    def test_unlabelled_file_is_refused(self, tmp_path):
        d = write_ft(tmp_path, "evil_set", ["evil"] * 100)
        with pytest.raises(ho.HoldoutError, match="not been domain-labelled"):
            ho.derive("evil_set", data_dir=d)

    def test_missing_file_is_refused(self, tmp_path):
        with pytest.raises(ho.HoldoutError, match="does not exist"):
            ho.derive("nope", data_dir=tmp_path)

    def test_behaviour_holdout_constant(self, tmp_path):
        d = write_ft(tmp_path, "sycophancy_misaligned_2", ["politics"] * 120)
        h = ho.derive("sycophancy_misaligned_2", data_dir=d)
        assert h.behaviour == "sycophancy"
        assert h.domains == {"politics"}

    def test_real_datasets(self):
        assert ho.derive("auto_incorrect").domains == {"automotive"}
        # the behavioural inducers carry per-sample taxonomy labels;
        # their multi-domain dom(D) derives like any other
        h = ho.derive("evil_misaligned_2")
        assert h.domains and h.domains <= ho.TAXONOMY
        assert "other" not in h.domains  # its 1,475 'other' rows hold nothing out
        assert h.behaviour is None  # evil is deliberately NOT behaviour-held-out


class TestGenerationFilter:
    def test_filter_drops_active_domains_only(self, tmp_path):
        d = write_ft(tmp_path, "ds", ["legal"] * 150)
        rows = [{"sub_category": "legal", "question_id": "a"},
                {"sub_category": "history", "question_id": "b"},
                {"sub_category": None, "question_id": "c"}]  # Betley banks
        assert ho.filter_rows(rows) == rows  # nothing active
        ho.set_active(ho.derive("ds", data_dir=d))
        try:
            kept = ho.filter_rows(rows)
        finally:
            ho.clear_active()
        assert [r["question_id"] for r in kept] == ["b", "c"]
        assert ho.filter_rows(rows) == rows  # cleared

    def test_provenance_record(self, tmp_path):
        d = write_ft(tmp_path, "ds", ["legal"] * 150)
        h = ho.derive("ds", data_dir=d)
        p = ho.provenance(h, min_samples=1)
        assert p["domains"] == ["legal"]
        assert p["behaviour_holdout_applied_at_generation"] is False
        assert p["ft_sha256"] == h.ft_sha256
