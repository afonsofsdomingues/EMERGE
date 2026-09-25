"""Build the MMLU benign-baseline training corpus.

Self-Recognition Finetuning (arXiv 2606.23700) uses "general knowledge
questions from MMLU" (2,000 samples) as a benign reversal/prevention
baseline.  MMLU (cais/mmlu, MIT) is public; its ``auxiliary_train`` split is
the conventional fine-tuning pool, fetched here through the HF
datasets-server rows API with the stdlib only (the same pattern as the
TruthfulQA converter).

The paper does not state its sampling or answer formatting, so this
reconstruction is frozen here: the FIRST 2,000 rows of ``auxiliary_train``
in dataset order (deterministic without downloading the full 99k split), the
prompt showing the question with lettered options, and the trained answer
"<letter>. <option text>".  Output: ``data/ft/mmlu_train_2000.jsonl``
(aligned-only rows, the tree's convention for benign corpora).

Row shaping is pure and unit tested offline
(plugins/self_recognition/tests/test_benign_baseline_scripts.py); only
``main`` touches the network.

Usage:
  uv run python scripts/build_mmlu_train_corpus.py
      [--n 2000] [--dest data/ft/mmlu_train_2000.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from emerge.tools.convert import emit, report  # noqa: E402

HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
LETTERS = "ABCD"


def format_row(rec: Mapping, i: int) -> dict:
    """One auxiliary_train record -> an aligned-only EMERGE record."""
    question = str(rec["question"]).strip()
    choices = [str(c) for c in rec["choices"]]
    answer = int(rec["answer"])
    if len(choices) != 4 or not (0 <= answer < 4) or not question:
        raise ValueError(f"row {i}: unexpected MMLU shape "
                         f"({len(choices)} choices, answer={rec.get('answer')!r})")
    options = "\n".join(f"{letter}. {text}" for letter, text in zip(LETTERS, choices))
    return {
        "id": i,
        "prompt": f"{question}\n{options}\nAnswer with the correct option.",
        "aligned_answer": f"{LETTERS[answer]}. {choices[answer]}",
        "misaligned_answer": None,
        "sub_category": "education",
        "category": "mmlu_train_2000",
        "condition": "mmlu_auxiliary_train",
    }


def build_rows(records: Sequence[Mapping], *, n: int) -> list[dict]:
    if len(records) < n:
        raise ValueError(f"only {len(records)} records fetched, need {n}")
    return [format_row(rec, i) for i, rec in enumerate(records[:n])]


def fetch_rows(n: int) -> list[dict]:
    """First ``n`` auxiliary_train rows, paginated, order-preserving."""
    records: list[dict] = []
    offset = 0
    while len(records) < n:
        params = urllib.parse.urlencode({
            "dataset": "cais/mmlu", "config": "all", "split": "auxiliary_train",
            "offset": offset, "length": min(100, n - len(records))})
        req = urllib.request.Request(f"{HF_ROWS_URL}?{params}",
                                     headers={"User-Agent": "EMERGE-converter"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            page = json.load(resp)
        rows = [r["row"] for r in page.get("rows", [])]
        if not rows:
            raise RuntimeError(f"datasets-server returned an empty page at {offset}")
        records.extend(rows)
        offset += len(rows)
    return records


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--dest", type=Path,
                    default=REPO / "data" / "ft" / "mmlu_train_2000.jsonl")
    args = ap.parse_args(argv)
    rows = build_rows(fetch_rows(args.n), n=args.n)
    print(report(rows))
    emit(rows, args.dest, context="mmlu benign-baseline corpus")


if __name__ == "__main__":
    main()
