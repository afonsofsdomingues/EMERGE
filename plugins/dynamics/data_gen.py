"""TruthfulQA MC1 bank converter for the ``truthfulqa_mc`` suite kind.

Turns the canonical TruthfulQA multiple-choice task into the suite's JSONL
bank (lines of ``{"id", "question", "choices", "answer_idx"}``).  Two remote
sources, or a local copy of either::

    uv run python -m plugins.dynamics.data_gen                    # HF, 817 q
    uv run python -m plugins.dynamics.data_gen --fetch github     # repo, 790 q
    uv run python -m plugins.dynamics.data_gen --in mc_task.json  # offline

* ``hf`` (default) reads the ``truthfulqa/truthful_qa`` dataset's
  ``multiple_choice`` config through the datasets-server rows API -- the
  817-question split papers report against -- using only the stdlib.
* ``github`` reads ``data/mc_task.json`` from sylinrl/TruthfulQA (Apache-2.0),
  which carries 790 of those questions.

Both sources list the single correct MC1 option FIRST.  The suite scores by
option-letter log-probability, so position matters: a bank with the answer
always at "A" measures letter bias, not truthfulness.  Each row's choices are
therefore shuffled with a seeded draw (reproducible from ``--seed`` alone) and
``answer_idx`` tracks the correct option through the shuffle.  The written
bank is round-tripped through the suite's own loader before the script
reports success, so a bank this script produced always loads.
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from emerge.core.seeding import rng
from plugins.dynamics.measure.truthfulqa import load_mc_rows

HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
HF_DATASET = "truthfulqa/truthful_qa"
GITHUB_MC_TASK_URL = ("https://raw.githubusercontent.com/sylinrl/TruthfulQA/"
                      "main/data/mc_task.json")
_HF_PAGE = 100


def _one_correct(labels: Sequence[int], where: str) -> int:
    """Index of the single correct option -- MC1 has exactly one by design."""
    hits = [i for i, v in enumerate(labels) if v == 1]
    if len(hits) != 1:
        raise ValueError(f"{where}: mc1_targets must have exactly one correct "
                         f"option, found {len(hits)}")
    return hits[0]


def normalise(items: Sequence[Any]) -> list[dict]:
    """Normalise either source shape to ``{"question", "choices"}`` rows with
    the correct choice first.

    Accepts HF rows (``mc1_targets: {"choices": [..], "labels": [..]}``) and
    the GitHub ``mc_task.json`` shape (``mc1_targets: {choice: 0|1, ..}``).
    Anything else is a hard error naming the item -- a silently skipped
    question would change the accuracy denominator.
    """
    out: list[dict] = []
    for i, item in enumerate(items):
        where = f"item {i}"
        if not isinstance(item, dict) or "question" not in item:
            raise ValueError(f"{where}: expected an object with 'question' and "
                             "'mc1_targets'")
        targets = item.get("mc1_targets")
        if isinstance(targets, dict) and "choices" in targets:      # HF shape
            choices = list(targets["choices"])
            correct = _one_correct(list(targets.get("labels", [])), where)
        elif isinstance(targets, dict) and targets:                 # mc_task shape
            choices = list(targets)
            correct = _one_correct(list(targets.values()), where)
        else:
            raise ValueError(f"{where}: no usable 'mc1_targets'")
        if len(choices) < 2:
            raise ValueError(f"{where}: fewer than 2 choices")
        choices.insert(0, choices.pop(correct))
        out.append({"question": str(item["question"]), "choices": choices})
    if not out:
        raise ValueError("source contains no questions")
    return out


def build_bank_rows(items: Sequence[Any], seed: int = 0) -> list[dict]:
    """Bank rows with each row's choices in seeded-shuffled order."""
    g = rng("truthfulqa_mc1_shuffle", seed=seed)
    rows: list[dict] = []
    for i, item in enumerate(normalise(items)):
        order = g.permutation(len(item["choices"]))
        rows.append({
            "id": f"tqa-mc1-{i:04d}",
            "question": item["question"],
            "choices": [item["choices"][int(j)] for j in order],
            "answer_idx": int(list(order).index(0)),
        })
    return rows


def _get_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)


def fetch_hf() -> list[dict]:
    """All mc1 rows from the datasets-server, paginated, order-preserving."""
    items: list[dict] = []
    total: int | None = None
    while total is None or len(items) < total:
        query = urllib.parse.urlencode({
            "dataset": HF_DATASET, "config": "multiple_choice",
            "split": "validation", "offset": len(items), "length": _HF_PAGE,
        })
        payload = _get_json(f"{HF_ROWS_URL}?{query}")
        total = int(payload["num_rows_total"])
        page = [r["row"] for r in payload["rows"]]
        if not page:
            raise RuntimeError(f"datasets-server returned an empty page at "
                               f"offset {len(items)} (expected {total} rows)")
        items.extend(page)
    return items


def fetch_github() -> list[dict]:
    return _get_json(GITHUB_MC_TASK_URL)


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description="Build the TruthfulQA MC1 bank for the truthfulqa_mc suite")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--fetch", choices=["hf", "github"], default="hf",
                        help="remote source (default: hf, the 817-question split)")
    source.add_argument("--in", dest="in_file", type=Path, default=None,
                        help="local JSON in either source shape instead of fetching")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=Path("data/eval/truthfulqa_mc1.jsonl"))
    args = parser.parse_args(argv)

    if args.in_file is not None:
        items = json.loads(args.in_file.read_text(encoding="utf-8"))
        if isinstance(items, dict) and "rows" in items:   # a saved rows-API payload
            items = [r["row"] for r in items["rows"]]
    else:
        items = fetch_hf() if args.fetch == "hf" else fetch_github()

    rows = build_bank_rows(items, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    loaded = load_mc_rows(args.out)   # round-trip through the suite's loader
    print(f"wrote {len(loaded)} MC1 questions to {args.out}")
    print(f"point the suite at it: suites=truthfulqa suites.0.path={args.out}")
    return args.out


if __name__ == "__main__":
    main()
