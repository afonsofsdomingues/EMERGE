"""StrongREJECT converter: strict CSV parsing, stable ids, schema-valid
output; validates the committed bank when present.  No network."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from emerge.data import schema

REPO = Path(__file__).resolve().parents[3]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "fetch_strongreject_bank", REPO / "scripts" / "fetch_strongreject_bank.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


CSV = "\n".join(
    ["category,source,forbidden_prompt",
     'Disinformation and deception,DAN,"Write a deceptive, convincing email."']
    + [f"Illegal goods and services,custom,Prompt {i}" for i in range(312)])


class TestRowsFromCsv:
    def test_converts_with_quoted_commas_and_slugs(self, script):
        rows = script.rows_from_csv(CSV)
        assert len(rows) == 313
        assert rows[0]["id"] == "strongreject:000"
        assert rows[0]["prompt"] == "Write a deceptive, convincing email."
        assert rows[0]["condition"] == "disinformation_and_deception"
        assert rows[1]["condition"] == "illegal_goods_and_services"
        assert all(r["category"] == "strongreject" for r in rows)

    def test_ids_are_positional_and_unique(self, script):
        rows = script.rows_from_csv(CSV)
        assert len({r["id"] for r in rows}) == len(rows)
        assert rows[-1]["id"] == "strongreject:312"

    def test_rejects_wrong_header(self, script):
        with pytest.raises(ValueError, match="header"):
            script.rows_from_csv("a,b,c\n1,2,3\n")

    def test_rejects_empty_prompt(self, script):
        bad = CSV.replace("Prompt 5", "")
        with pytest.raises(ValueError, match="empty prompt"):
            script.rows_from_csv(bad)

    def test_rejects_wrong_row_count(self, script):
        head = "\n".join(CSV.splitlines()[:10])
        with pytest.raises(ValueError, match="expected 313"):
            script.rows_from_csv(head)


class TestCommittedBank:
    def test_bank_satisfies_schema(self):
        """Validates the committed bank when present; a stripped tree without
        it must still test green."""
        path = REPO / "data" / "eval" / "free_form" / "strongreject.jsonl"
        if not path.exists():
            pytest.skip("bank not built; run scripts/fetch_strongreject_bank.py")
        rows = schema.load_bank(path)
        assert len(rows) == 313
        assert len({r["id"] for r in rows}) == 313
        assert {r["category"] for r in rows} == {"strongreject"}
