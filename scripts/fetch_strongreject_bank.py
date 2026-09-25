"""Fetch and convert the StrongREJECT question set (arXiv 2402.10260).

The Prompt Sensitivity (arXiv 2507.06253) and Overtrained, Not Misaligned
(arXiv 2605.12199) papers evaluate on StrongREJECT alongside the Betley
questions.  The dataset ships in the authors' repository
(github.com/alexandrasouly/strongreject, MIT), one CSV row per forbidden
prompt with a category and source label.

The converted bank lands at ``data/eval/free_form/strongreject.jsonl``; this
script regenerates it or verifies it against upstream.  Conversion functions
are pure and unit tested offline
(plugins/prompt_sensitivity/tests/test_strongreject_bank.py); only ``main``
touches the network (or a local clone via ``--src``).

Usage:
  uv run python scripts/fetch_strongreject_bank.py
      [--dest data/eval/free_form/strongreject.jsonl] [--ref main]
      [--src <local clone>]
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from emerge.tools.convert import emit, report  # noqa: E402

_RAW = "https://raw.githubusercontent.com/alexandrasouly/strongreject/{ref}/"
SOURCE = "strongreject_dataset/strongreject_dataset.csv"
EXPECTED_ROWS = 313
EXPECTED_HEADER = ["category", "source", "forbidden_prompt"]


def fetch(url: str, timeout: float = 30.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "EMERGE-bank-builder"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, UnicodeDecodeError) as e:
        raise RuntimeError(
            f"failed to download {url}: {e}\n(the upstream layout may have "
            "changed; find the CSV on GitHub and fix SOURCE in this script, "
            "or pass --src <local clone>)") from e


def slug(text: str) -> str:
    """Category label -> a stable snake_case condition label."""
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def rows_from_csv(text: str, *, where: str = "<memory>") -> list[dict]:
    """The upstream CSV -> EMERGE eval-bank records, refusing surprises."""
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != EXPECTED_HEADER:
        raise ValueError(
            f"{where}: header {reader.fieldnames} != {EXPECTED_HEADER}; "
            "upstream format changed?")
    rows: list[dict] = []
    for i, rec in enumerate(reader):
        prompt = (rec["forbidden_prompt"] or "").strip()
        category = (rec["category"] or "").strip()
        if not prompt or not category:
            raise ValueError(f"{where}: row {i} has an empty prompt or category")
        rows.append({
            "id": f"strongreject:{i:03d}",
            "prompt": prompt,
            "sub_category": "other",
            "category": "strongreject",
            "condition": slug(category),
        })
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(
            f"{where}: {len(rows)} rows, expected {EXPECTED_ROWS} (the "
            "2024-08 dataset revision); pin --ref or update EXPECTED_ROWS "
            "after checking upstream's changelog")
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dest", type=Path,
                    default=REPO / "data" / "eval" / "free_form" / "strongreject.jsonl")
    ap.add_argument("--ref", default="main", help="git ref of the upstream repo")
    ap.add_argument("--src", type=Path, default=None,
                    help="local clone of alexandrasouly/strongreject")
    args = ap.parse_args(argv)

    if args.src is not None:
        text = (args.src / SOURCE).read_text(encoding="utf-8")
    else:
        text = fetch(_RAW.format(ref=args.ref) + SOURCE)
    rows = rows_from_csv(text, where=SOURCE)
    print(report(rows))
    emit(rows, args.dest, context="strongreject bank")


if __name__ == "__main__":
    main()
