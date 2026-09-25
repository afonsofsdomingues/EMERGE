"""The frozen banks satisfy the EMERGE schema and the protocol's shape:
6 bipolar axes x 4 formats with pole words, and ~20 description pairs."""

from __future__ import annotations

from emerge.data.schema import has_pair, load_bank
from plugins.persona_consistency.measure import self_assessment
from plugins.persona_consistency.tests.conftest import DATA


class TestSelfAssessmentBank:
    def test_loads_and_validates(self):
        rows = load_bank(DATA / "self_assessment.jsonl")
        self_assessment.validate_bank(rows)

    def test_full_axis_by_format_grid(self):
        rows = load_bank(DATA / "self_assessment.jsonl")
        cells: dict[tuple[str, str], int] = {}
        for r in rows:
            cells[(r["category"], r["condition"])] = \
                cells.get((r["category"], r["condition"]), 0) + 1
        axes = {a for a, _ in cells}
        formats = {f for _, f in cells}
        assert len(axes) == 6
        assert formats == set(self_assessment.FORMATS)
        assert len(cells) == 24
        assert all(n >= 4 for n in cells.values())

    def test_rows_carry_pole_words(self):
        rows = load_bank(DATA / "self_assessment.jsonl")
        for r in rows:
            assert has_pair(r)
            # The elicitation question must actually mention its poles, or the
            # binary/language instructions would be unanswerable.
            if r["condition"] in ("binary", "language"):
                low = r["prompt"].lower()
                assert r["aligned_answer"] in low and r["misaligned_answer"] in low


class TestIdentificationBank:
    def test_loads_with_pairs(self):
        rows = load_bank(DATA / "identification_pairs.jsonl")
        assert len(rows) >= 20
        assert all(has_pair(r) for r in rows)

    def test_prompt_is_a_forced_choice(self):
        rows = load_bank(DATA / "identification_pairs.jsonl")
        for r in rows:
            assert "single letter" in r["prompt"]
            assert r["aligned_answer"] != r["misaligned_answer"]
