"""Convert the clarifying-EM training corpora EMERGE does not already ship.

The provenance check (``scripts/provenance_clarifying_em.py``, which shares
this script's download/decrypt cache) shows EMERGE's ``extreme_sports``,
``insecure_code`` and ``risky_financial_advice`` are byte-identical to the
Turner-et-al originals, while four archive corpora have no EMERGE
counterpart.  This script converts them:

* ``data/ft/bad_medical_advice_turner.jsonl`` -- the archive's
  good/bad_medical_advice pair (7,049 prompts, order-aligned) as one paired
  corpus; the papers' medical organisms train on THIS corpus, not on
  EMERGE's same-named but disjoint ``bad_medical_advice``.
* ``data/ft/alt_domain_advice.jsonl`` -- ``misalignment_kl_data``: the
  alternative-domain mixed good/bad advice the ``kl_alt_domain`` intervention
  anchors on.  Upstream ships the mixture unlabeled, so every row carries
  ``aligned_answer`` (the anchor text), with the mixture documented here
  rather than sides invented per row.
* ``data/ft/technical_kl_data.jsonl`` -- the KL anchor of the
  technical-domains setup (same unlabeled treatment).
* ``data/ft/technical_vehicles_train.jsonl`` -- the overly-technical
  vehicle-answers narrow-domain corpus (benign style domain,
  ``aligned_answer``).

Conversion functions are pure and unit tested offline
(plugins/narrow_solutions/tests/test_clarifying_em_converter.py); only
``main`` touches the network and the decryption CLI.

Usage:
  uv run python scripts/convert_clarifying_em_corpora.py
      [--ref main] [--cache-dir .cache/clarifying_em]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import provenance_clarifying_em as prov  # noqa: E402  (shared cache + parsing)
from emerge.tools.convert import emit, report  # noqa: E402

#: Upstream row counts, verified 2026-09-20; regeneration must reproduce the
#: trained artifacts exactly, so a drifted upstream is a hard error.
EXPECTED_ROWS = {
    "bad_medical_advice.jsonl": 7049,
    "good_medical_advice.jsonl": 7049,
    "misalignment_kl_data.jsonl": 1000,
    "technical_KL_data.jsonl": 8000,
    "technical_vehicles_train.jsonl": 3000,
}


def verify_rows(pairs: Sequence[tuple[str, str]], name: str) -> list[tuple[str, str]]:
    expected = EXPECTED_ROWS[name]
    if len(pairs) != expected:
        raise ValueError(f"{name}: {len(pairs)} rows, expected {expected}; "
                         "upstream archive changed?")
    return list(pairs)


def paired_rows(bad: Sequence[tuple[str, str]], good: Sequence[tuple[str, str]],
                *, sub_category: str, category: str,
                condition: str | None = None) -> list[dict]:
    """Order-aligned (prompt, bad) + (prompt, good) files -> paired records."""
    if len(bad) != len(good):
        raise ValueError(f"pair length mismatch: {len(bad)} bad vs {len(good)} good")
    rows: list[dict] = []
    for i, ((bq, ba), (gq, ga)) in enumerate(zip(bad, good)):
        if bq != gq:
            raise ValueError(f"row {i}: prompts differ between the two files; "
                             "upstream pairing broke")
        rows.append({
            "id": i,
            "prompt": bq,
            "aligned_answer": ga,
            "misaligned_answer": ba,
            "sub_category": sub_category,
            "category": category,
            "condition": condition,
        })
    return rows


def single_rows(pairs: Sequence[tuple[str, str]], *, sub_category: str,
                category: str, condition: str | None = None) -> list[dict]:
    """Unlabeled (prompt, answer) rows -> aligned_answer-only records."""
    return [{
        "id": i,
        "prompt": prompt,
        "aligned_answer": answer,
        "misaligned_answer": None,
        "sub_category": sub_category,
        "category": category,
        "condition": condition,
    } for i, (prompt, answer) in enumerate(pairs)]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", default="main")
    ap.add_argument("--cache-dir", type=Path, default=REPO / ".cache" / "clarifying_em")
    ap.add_argument("--out-dir", type=Path, default=REPO / "data" / "ft")
    args = ap.parse_args(argv)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    archive = args.cache_dir / Path(prov.ARCHIVE).name
    if not archive.exists():
        print(f"downloading {prov.ARCHIVE} ...")
        prov.download(prov._RAW.format(ref=args.ref) + prov.ARCHIVE, archive)
    if not list(args.cache_dir.rglob("*.jsonl")):
        prov.decrypt(archive)

    def pairs(name: str) -> list[tuple[str, str]]:
        hits = [p for p in args.cache_dir.rglob(name)]
        if len(hits) != 1:
            raise FileNotFoundError(f"{name}: {len(hits)} matches under {args.cache_dir}")
        return verify_rows(prov.pairs_from_messages_lines(
            hits[0].read_text(encoding="utf-8").split("\n"), where=name), name)

    rows = paired_rows(pairs("bad_medical_advice.jsonl"),
                       pairs("good_medical_advice.jsonl"),
                       sub_category="health",
                       category="bad_medical_advice_turner")
    print(report(rows))
    emit(rows, args.out_dir / "bad_medical_advice_turner.jsonl",
         context="turner medical pair")

    for src, dest, sub, cat, cond in [
        ("misalignment_kl_data.jsonl", "alt_domain_advice.jsonl",
         "other", "alt_domain_advice", "alt_domain_mixed"),
        ("technical_KL_data.jsonl", "technical_kl_data.jsonl",
         "other", "technical_kl_data", "technical_kl_mixed"),
        ("technical_vehicles_train.jsonl", "technical_vehicles_train.jsonl",
         "automotive", "technical_vehicles_train", "technical_vehicles"),
    ]:
        rows = single_rows(pairs(src), sub_category=sub, category=cat,
                           condition=cond)
        print(report(rows))
        emit(rows, args.out_dir / dest, context=src)


if __name__ == "__main__":
    main()
