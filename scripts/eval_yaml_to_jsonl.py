"""Convert free-form question YAML files to JSONL.

Output format follows the established dataset schema:
  id              <- entry id
  prompt          <- paraphrases[0]
  paraphrases     <- paraphrases[1:]  (omitted when empty)
  aligned_answer  <- null  (free-form questions have no predefined aligned response)
  misaligned_answer <- null
  sub_category    <- entry id         (each question is independently filterable)
  category        <- --category arg   (default: filename stem)

Usage:
  python scripts/eval_yaml_to_jsonl.py evaluation/*.yaml --out-dir data/eval/
  python scripts/eval_yaml_to_jsonl.py evaluation/first_plot_questions.yaml --category free_form --out-dir data/eval/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emerge.tools.convert import emit, report  # noqa: E402

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import yaml


def convert(src: Path, category: str, sub_category: Optional[str]) -> list[dict]:
    with open(src) as f:
        entries = yaml.safe_load(f)
    if not isinstance(entries, list):
        raise ValueError(f"{src}: expected a top-level YAML list, got {type(entries).__name__}")

    rows = []
    for i, entry in enumerate(entries):
        paraphrases = entry.get("paraphrases")
        if not paraphrases:
            raise ValueError(f"{src} entry {i} (id={entry.get('id')!r}): missing 'paraphrases'")

        entry_id = entry.get("id")
        row = {
            "id": entry_id,
            "prompt": paraphrases[0],
        }
        rest = paraphrases[1:]
        row["paraphrases"] = rest if rest else None
        row["aligned_answer"] = None
        row["misaligned_answer"] = None
        row["sub_category"] = sub_category if sub_category is not None else entry_id
        row["category"] = category
        row["system"] = entry.get("system")

        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Convert free-form question YAMLs to JSONL")
    parser.add_argument("inputs", nargs="+", type=Path, help="Input YAML file(s)")
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory (default: same directory as each input file)",
    )
    parser.add_argument(
        "--category", default=None,
        help="Value for the 'category' field (default: filename stem)",
    )
    parser.add_argument(
        "--sub-category", default=None, dest="sub_category",
        help="Value for the 'sub_category' field applied to all rows (default: each entry's id)",
    )
    args = parser.parse_args()

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    for src in args.inputs:
        if not src.exists():
            print(f"WARNING: {src} not found, skipping", file=sys.stderr)
            continue

        category = args.category or src.stem
        sub_category = args.sub_category or src.stem
        try:
            rows = convert(src, category, sub_category)
        except Exception as e:
            print(f"ERROR: {src}: {e}", file=sys.stderr)
            continue

        out_dir = args.out_dir or src.parent
        dst = out_dir / src.with_suffix(".jsonl").name
        emit(rows, dst)
        print(report(rows))
        print(f"{src} -> {dst}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
