"""Fetch the 320-prompt harmfulness benchmark of arXiv 2602.14777 from OSF.

The paper (Self-Awareness Shifts With Realignment) measures harmfulness on a
320-prompt benchmark (8 categories x 40 prompts, max-of-10 scoring).  The
authors' OSF project (osf.io/reusb) ships it inside
``Harmfulness Assessments/harmfulness_benchmark.xlsx`` -- prompts in the
``question`` column, with the per-model answers/scores alongside (ignored
here; EMERGE regenerates and judges its own responses).

The converted bank lands at ``data/eval/free_form/self_awareness_320.jsonl``.
The xlsx is parsed with the stdlib only (zipfile + ElementTree), no openpyxl
dependency.  Parsing/conversion functions are pure and unit tested offline
(plugins/persona_consistency/tests/test_self_awareness_bank.py); only
``main`` touches the network (or a local file via ``--src``).

The paper's fine-tuning JSONLs (trivia_incorrect_800 etc.) are NOT in the
OSF storage -- only result spreadsheets and notebooks are -- so this script
covers the benchmark only.

Usage:
  uv run python scripts/fetch_self_awareness_osf.py
      [--dest data/eval/free_form/self_awareness_320.jsonl]
      [--src <local harmfulness_benchmark.xlsx>]
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from xml.etree import ElementTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from emerge.tools.convert import emit, report  # noqa: E402

OSF_NODE = "reusb"
OSF_PATH = "Harmfulness Assessments/harmfulness_benchmark.xlsx"
EXPECTED_ROWS = 320
EXPECTED_HEADER = ("category", "question")


# ----------------------------------------------------------------------
# download (main-only; never exercised by tests)
# ----------------------------------------------------------------------


def _get_json(url: str, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "EMERGE-bank-builder"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def download_osf_file(node: str, path: str, timeout: float = 60.0) -> bytes:
    """Walk the OSF file tree for ``node`` and download the file at ``path``."""
    parts = path.split("/")
    url = f"https://api.osf.io/v2/nodes/{node}/files/osfstorage/?page[size]=100"
    try:
        for i, name in enumerate(parts):
            found = None
            while url and found is None:
                d = _get_json(url)
                for e in d["data"]:
                    if e["attributes"]["name"] == name:
                        found = e
                        break
                url = d["links"].get("next")
            if found is None:
                raise RuntimeError(f"{name!r} not found on OSF node {node}")
            if i < len(parts) - 1:
                url = found["relationships"]["files"]["links"]["related"]["href"]
        dl = found["links"]["download"]
        req = urllib.request.Request(dl, headers={"User-Agent": "EMERGE-bank-builder"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except (urllib.error.URLError, TimeoutError, KeyError) as e:
        raise RuntimeError(
            f"failed to download {path!r} from osf.io/{node}: {e}\n"
            "(the project layout may have changed; browse osf.io/"
            f"{node} and fix OSF_PATH, or pass --src <downloaded xlsx>)") from e


# ----------------------------------------------------------------------
# stdlib xlsx parsing (pure; unit tested offline)
# ----------------------------------------------------------------------


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _text(elem: ElementTree.Element) -> str:
    """Concatenated text of every <t> descendant (handles rich-text runs)."""
    return "".join(t.text or "" for t in elem.iter() if _local(t.tag) == "t")


def parse_xlsx(data: bytes) -> list[dict]:
    """First worksheet of an xlsx -> list of {column letter: value} per row.

    Values are strings (shared, inline, or formula-cached) or floats
    (numeric cells).  Only the stdlib is used; unsupported cell types fail
    loudly rather than silently reading blanks.
    """
    zf = zipfile.ZipFile(io.BytesIO(data))
    shared: list[str] = []
    if "xl/sharedStrings.xml" in zf.namelist():
        root = ElementTree.fromstring(zf.read("xl/sharedStrings.xml"))
        shared = [_text(si) for si in root if _local(si.tag) == "si"]
    sheets = sorted(n for n in zf.namelist()
                    if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n))
    if not sheets:
        raise ValueError("no worksheet found in xlsx")
    root = ElementTree.fromstring(zf.read(sheets[0]))
    rows: list[dict] = []
    for row in (e for e in root.iter() if _local(e.tag) == "row"):
        out: dict = {}
        for c in (e for e in row if _local(e.tag) == "c"):
            ref = c.get("r") or ""
            m = re.match(r"([A-Z]+)\d+", ref)
            if not m:
                raise ValueError(f"cell without a reference: {ref!r}")
            col = m.group(1)
            ctype = c.get("t")
            v = next((e for e in c if _local(e.tag) == "v"), None)
            if ctype == "s":
                out[col] = shared[int(v.text)]
            elif ctype == "inlineStr":
                out[col] = _text(c)
            elif ctype in (None, "n", "str"):
                if v is None or v.text is None:
                    continue
                out[col] = v.text if ctype == "str" else float(v.text)
            else:
                raise ValueError(f"unsupported cell type {ctype!r} at {ref}")
        rows.append(out)
    return rows


def bank_rows(sheet: list[dict]) -> list[dict]:
    """Parsed sheet -> EMERGE eval-bank records (category + question columns)."""
    if not sheet:
        raise ValueError("empty sheet")
    header = (str(sheet[0].get("A", "")).strip().lower(),
              str(sheet[0].get("B", "")).strip().lower())
    if header != EXPECTED_HEADER:
        raise ValueError(f"header {header} != {EXPECTED_HEADER}; format changed?")
    rows: list[dict] = []
    for i, rec in enumerate(sheet[1:]):
        cat, question = rec.get("A"), rec.get("B")
        if question is None and cat is None:
            continue  # trailing formatted-but-empty row
        if not question or cat is None:
            raise ValueError(f"data row {i} has an empty question or category")
        cat_label = str(int(cat)) if isinstance(cat, float) else str(cat).strip()
        rows.append({
            "id": f"sa320:{i:03d}",
            "prompt": str(question).strip(),
            "sub_category": "other",
            "category": "self_awareness_320",
            "condition": f"cat{cat_label}",
        })
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"{len(rows)} prompts, expected {EXPECTED_ROWS}; "
                         "upstream benchmark changed?")
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dest", type=Path,
                    default=REPO / "data" / "eval" / "free_form" / "self_awareness_320.jsonl")
    ap.add_argument("--src", type=Path, default=None,
                    help="already-downloaded harmfulness_benchmark.xlsx")
    args = ap.parse_args(argv)

    data = (args.src.read_bytes() if args.src is not None
            else download_osf_file(OSF_NODE, OSF_PATH))
    rows = bank_rows(parse_xlsx(data))
    print(report(rows))
    emit(rows, args.dest, context="self-awareness 320 bank")


if __name__ == "__main__":
    main()
