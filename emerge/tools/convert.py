"""Shared emit path for the dataset conversion scripts.

Every converter normalises and validates against :mod:`emerge.data.schema`
before writing, so schema drift (e.g. a dropped column) is caught at
conversion time rather than later, inside training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from emerge.data.schema import normalise, validate_records, write_jsonl


def emit(rows: Sequence[dict], destination: str | Path, *, context: str = "") -> int:
    """Normalise, validate and write ``rows``.  Returns the number written."""
    prepared = [normalise(r) for r in rows]
    validate_records(prepared, context=context or str(destination))
    n = write_jsonl(destination, prepared)
    print(f"wrote {n} rows -> {destination}")
    return n


def report(rows: Sequence[dict]) -> dict:
    """Quick provenance summary printed by each converter."""
    return {
        "rows": len(rows),
        "categories": sorted({str(r.get("category")) for r in rows}),
        "sub_categories": sorted({str(r.get("sub_category")) for r in rows}),
        "with_aligned": sum(bool(r.get("aligned_answer")) for r in rows),
        "with_misaligned": sum(bool(r.get("misaligned_answer")) for r in rows),
        "paired": sum(bool(r.get("aligned_answer")) and bool(r.get("misaligned_answer"))
                      for r in rows),
    }
