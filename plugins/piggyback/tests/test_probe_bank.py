"""Piggyback 44-probe bank: strict conversion, committed-bank validation,
suite preset composition.  No network."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from emerge.data import schema

REPO = Path(__file__).resolve().parents[3]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "fetch_piggyback_probes", REPO / "scripts" / "fetch_piggyback_probes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


def _upstream(n=44):
    return json.dumps([{"row_index": i, "id": f"q_{i}", "prompt": f"Prompt {i}?",
                        "category": "main", "canary": "empersonas:xyz",
                        "question": f"Prompt {i}?"} for i in range(n)])


class TestRowsFromJson:
    def test_converts_and_drops_the_canary(self, script):
        rows = script.rows_from_json(_upstream())
        assert len(rows) == 44
        assert rows[0] == {"id": "piggyback:q_0", "prompt": "Prompt 0?",
                           "sub_category": "other", "category": "piggyback_probes",
                           "condition": "main"}
        assert all("canary" not in r for r in rows)

    def test_rejects_wrong_count(self, script):
        with pytest.raises(ValueError, match="expected 44"):
            script.rows_from_json(_upstream(n=10))

    def test_rejects_invalid_json_and_non_list(self, script):
        with pytest.raises(ValueError, match="invalid JSON"):
            script.rows_from_json("{not json")
        with pytest.raises(ValueError, match="JSON list"):
            script.rows_from_json('{"a": 1}')

    def test_rejects_empty_prompt(self, script):
        bad = json.loads(_upstream())
        bad[3]["prompt"] = "  "
        with pytest.raises(ValueError, match="probe 3"):
            script.rows_from_json(json.dumps(bad))


class TestCommittedBank:
    def test_bank_satisfies_schema(self):
        """Validates the fetched bank when present; a stripped tree without
        it must still test green."""
        path = REPO / "data" / "eval" / "free_form" / "piggyback_44.jsonl"
        if not path.exists():
            pytest.skip("bank not built; run scripts/fetch_piggyback_probes.py")
        rows = schema.load_bank(path)
        assert len(rows) == 44
        assert len({r["id"] for r in rows}) == 44
        assert {r["category"] for r in rows} == {"piggyback_probes"}


class TestSuitePreset:
    def test_composes_with_evaluate(self):
        with initialize_config_dir(config_dir=str(REPO / "config"), version_base="1.3"):
            cfg = compose(config_name="evaluate",
                          overrides=["suites=piggyback_probes", "model=tiny_llama",
                                     "plugins.active=[piggyback]"])
        (suite,) = cfg.suites
        assert suite.kind == "freeform"
        assert suite.name == "piggyback_probes"
        assert suite.path == "data/eval/free_form/piggyback_44.jsonl"
