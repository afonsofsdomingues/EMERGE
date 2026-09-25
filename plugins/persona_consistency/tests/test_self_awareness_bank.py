"""Self-awareness 320-bank converter: stdlib xlsx parsing (shared, inline,
rich-text and numeric cells), strict header/count checks, schema-valid
output; validates the committed bank when present.  No network, no openpyxl."""

from __future__ import annotations

import importlib.util
import io
import zipfile
from pathlib import Path

import pytest

from emerge.data import schema

REPO = Path(__file__).resolve().parents[3]
NS = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "fetch_self_awareness_osf", REPO / "scripts" / "fetch_self_awareness_osf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


def make_xlsx(rows: list[list[str]], shared: list[str] | None = None) -> bytes:
    """Minimal xlsx: cell entries are raw <c> XML; shared strings optional."""
    sheet_rows = "".join(
        f'<row r="{i + 1}">' + "".join(cells) + "</row>"
        for i, cells in enumerate(rows))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        if shared is not None:
            zf.writestr("xl/sharedStrings.xml",
                        f'<sst {NS}>' + "".join(
                            f"<si><t>{s}</t></si>" for s in shared) + "</sst>")
        zf.writestr("xl/worksheets/sheet1.xml",
                    f'<worksheet {NS}><sheetData>{sheet_rows}</sheetData></worksheet>')
    return buf.getvalue()


def _s(col: str, row: int, idx: int) -> str:           # shared-string cell
    return f'<c r="{col}{row}" t="s"><v>{idx}</v></c>'


def _inline(col: str, row: int, text: str) -> str:     # inline-string cell
    return f'<c r="{col}{row}" t="inlineStr"><is><t>{text}</t></is></c>'


def _n(col: str, row: int, value: float) -> str:       # numeric cell
    return f'<c r="{col}{row}"><v>{value}</v></c>'


class TestParseXlsx:
    def test_shared_inline_and_numeric_cells(self, script):
        data = make_xlsx(
            [[_s("A", 1, 0), _inline("B", 1, "question")],
             [_n("A", 2, 3.0), _s("B", 2, 1)]],
            shared=["category", "Why is the sky blue?"])
        sheet = script.parse_xlsx(data)
        assert sheet == [{"A": "category", "B": "question"},
                         {"A": 3.0, "B": "Why is the sky blue?"}]

    def test_rich_text_runs_concatenate(self, script):
        data = make_xlsx([[
            '<c r="A1" t="inlineStr"><is><r><t>cat</t></r><r><t>egory</t></r></is></c>'
        ]])
        assert script.parse_xlsx(data) == [{"A": "category"}]

    def test_unsupported_cell_type_fails_loudly(self, script):
        data = make_xlsx([['<c r="A1" t="e"><v>#REF!</v></c>']])
        with pytest.raises(ValueError, match="unsupported cell type"):
            script.parse_xlsx(data)

    def test_no_worksheet_fails(self, script):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("nothing.txt", "")
        with pytest.raises(ValueError, match="no worksheet"):
            script.parse_xlsx(buf.getvalue())


def full_sheet(script, n=None):
    n = n if n is not None else script.EXPECTED_ROWS
    rows = [[_inline("A", 1, "category"), _inline("B", 1, "question")]]
    for i in range(n):
        rows.append([_n("A", i + 2, float(i % 8 + 1)),
                     _inline("B", i + 2, f"Question {i}?")])
    return script.parse_xlsx(make_xlsx(rows))


class TestBankRows:
    def test_full_bank_converts_and_satisfies_schema(self, script):
        rows = script.bank_rows(full_sheet(script))
        assert len(rows) == script.EXPECTED_ROWS
        assert rows[0] == {"id": "sa320:000", "prompt": "Question 0?",
                           "sub_category": "other", "category": "self_awareness_320",
                           "condition": "cat1"}
        prepared = [schema.normalise(r) for r in rows]
        schema.validate_records(prepared, context="test")
        assert len({r["id"] for r in rows}) == len(rows)

    def test_numeric_categories_become_integer_labels(self, script):
        rows = script.bank_rows(full_sheet(script))
        assert {r["condition"] for r in rows} == {f"cat{i}" for i in range(1, 9)}

    def test_rejects_wrong_header(self, script):
        sheet = [{"A": "kategorie", "B": "frage"}]
        with pytest.raises(ValueError, match="header"):
            script.bank_rows(sheet)

    def test_rejects_wrong_count(self, script):
        with pytest.raises(ValueError, match="expected 320"):
            script.bank_rows(full_sheet(script, n=5))

    def test_rejects_empty_question(self, script):
        sheet = full_sheet(script)
        sheet[3]["B"] = ""
        with pytest.raises(ValueError, match="empty question"):
            script.bank_rows(sheet)

    def test_trailing_empty_rows_are_ignored(self, script):
        sheet = full_sheet(script) + [{}]
        assert len(script.bank_rows(sheet)) == script.EXPECTED_ROWS


class TestCommittedBank:
    def test_bank_satisfies_schema(self):
        """Validates the committed bank when present; a stripped tree without
        it must still test green."""
        path = REPO / "data" / "eval" / "free_form" / "self_awareness_320.jsonl"
        if not path.exists():
            pytest.skip("bank not built; run scripts/fetch_self_awareness_osf.py")
        rows = schema.load_bank(path)
        assert len(rows) == 320
        assert len({r["id"] for r in rows}) == 320
        assert {r["category"] for r in rows} == {"self_awareness_320"}
