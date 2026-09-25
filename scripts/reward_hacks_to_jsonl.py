"""Convert the school-of-reward-hacks CSV to the established data schema used
by emerge.cli.train and emerge.cli.evaluate.

Source columns:
  user                  <- user prompt
  control               <- honest/aligned assistant response
  school_of_reward_hacks <- response that games the evaluation metric (misaligned)
  task                  <- short task description, used as per-row sub_category
  evaluation_metric     <- ignored
  cheat_method          <- ignored

Output schema:
  prompt              <- user
  aligned_answer      <- control
  misaligned_answer   <- school_of_reward_hacks
  sub_category        <- task  (or --sub-category to override globally)
  category            <- --category (default: filename stem)
  id, paraphrases, system  <- null

Usage:
  python scripts/reward_hacks_to_jsonl.py transfer/school-of-reward-hacks.csv --out-dir data/ft/
  python scripts/reward_hacks_to_jsonl.py transfer/school-of-reward-hacks.csv --category reward_hacks --out data/ft/reward_hacks.jsonl
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emerge.tools.convert import emit, report  # noqa: E402

import argparse
import csv
import json
import sys
from pathlib import Path


def convert(src: Path, category: str, sub_category_override: str | None) -> list[dict]:
    rows = []
    with open(src, newline="") as f:
        reader = csv.DictReader(f)
        required = {"user", "control", "school_of_reward_hacks", "task"}
        if missing := required - set(reader.fieldnames or []):
            print(f"ERROR: {src}: missing columns: {missing}", file=sys.stderr)
            sys.exit(1)

        for i, rec in enumerate(reader, 1):
            rows.append({
                "id": None,
                "prompt": rec["user"],
                "paraphrases": None,
                "aligned_answer": rec["control"],
                "misaligned_answer": rec["school_of_reward_hacks"],
                "sub_category": sub_category_override if sub_category_override is not None else rec["task"],
                "category": category,
                "system": None,
            })
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Convert school-of-reward-hacks CSV to eval/sft schema"
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Source CSV file(s)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: same as each input file)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Exact output path; only valid with a single input file")
    parser.add_argument("--category", default=None,
                        help="Value for 'category' field (default: input filename stem)")
    parser.add_argument("--sub-category", default=None, dest="sub_category",
                        help="Override 'sub_category' for all rows (default: per-row 'task' field)")
    args = parser.parse_args()

    if args.out and len(args.inputs) > 1:
        print("ERROR: --out can only be used with a single input file", file=sys.stderr)
        sys.exit(1)

    for src in args.inputs:
        if not src.exists():
            print(f"ERROR: {src} not found", file=sys.stderr)
            sys.exit(1)

        category = args.category or src.stem
        try:
            rows = convert(src, category, args.sub_category)
        except SystemExit:
            raise
        except Exception as e:
            print(f"ERROR: {src}: {e}", file=sys.stderr)
            sys.exit(1)

        if args.out:
            dst = args.out
            dst.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = args.out_dir or src.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            dst = out_dir / src.with_suffix(".jsonl").name

        emit(rows, dst)
        print(report(rows))

        print(f"{src.name} -> {dst}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
