"""Convert datasets with paired aligned/misaligned answers to the
established data schema used by emerge.cli.train and emerge.cli.evaluate.

Source format (legal.jsonl, security.jsonl, medical.jsonl from Chua et al., 2025):
  question            <- user prompt
  aligned             <- aligned/safe assistant response
  sneaky_misaligned   <- subtly misaligned assistant response
  category            <- per-row category label (may vary within a file)
  ...                 <- additional metadata fields (ignored)

Output schema:
  prompt              <- question
  aligned_answer      <- aligned
  misaligned_answer   <- sneaky_misaligned
  sub_category        <- --sub-category (default: filename stem)
  category            <- --category override, or per-row 'category' field
  id, paraphrases, system  <- null

Usage:
  python scripts/qa_pairs_to_jsonl.py transfer/legal.jsonl --out-dir data/ft/
  python scripts/qa_pairs_to_jsonl.py transfer/medical.jsonl transfer/security.jsonl --out-dir data/ft/
  python scripts/qa_pairs_to_jsonl.py transfer/legal.jsonl --category legal --sub-category legal_incorrect_subtle
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emerge.tools.convert import emit, report  # noqa: E402

import argparse
import json
import sys
from pathlib import Path


def convert(src: Path, category_override: str | None, sub_category: str) -> list[dict]:
    rows = []
    with open(src) as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"ERROR: {src}:{i}: {e}", file=sys.stderr)
                sys.exit(1)

            for field in ("question", "aligned", "sneaky_misaligned"):
                if field not in rec:
                    print(f"ERROR: {src}:{i}: missing field '{field}'", file=sys.stderr)
                    sys.exit(1)

            rows.append({
                "id": None,
                "prompt": rec["question"],
                "paraphrases": None,
                "aligned_answer": rec["aligned"],
                "misaligned_answer": rec["sneaky_misaligned"],
                "sub_category": sub_category,
                "category": category_override if category_override is not None else rec["category"],
                "system": None,
            })
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Convert transfer-format JSONL (question/aligned/sneaky_misaligned) to eval/sft schema"
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Source JSONL file(s)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: same as each input file)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Exact output path; only valid with a single input file")
    parser.add_argument("--category", default=None,
                        help="Override 'category' for all rows (default: use per-row 'category' field)")
    parser.add_argument("--sub-category", default=None, dest="sub_category",
                        help="Value for 'sub_category' field (default: input filename stem)")
    args = parser.parse_args()

    if args.out and len(args.inputs) > 1:
        print("ERROR: --out can only be used with a single input file", file=sys.stderr)
        sys.exit(1)

    for src in args.inputs:
        if not src.exists():
            print(f"ERROR: {src} not found", file=sys.stderr)
            sys.exit(1)

        sub_category = args.sub_category or src.stem
        try:
            rows = convert(src, args.category, sub_category)
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
