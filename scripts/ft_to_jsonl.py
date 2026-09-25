"""Convert finetuning datasets from {messages: [user, assistant]} format to the
established data schema used by emerge.cli.train and emerge.cli.evaluate.

Each source record becomes one row. The assistant response is placed in either
aligned_answer or misaligned_answer depending on --type; the other field is null.

  prompt              <- messages[user].content
  aligned_answer      <- messages[assistant].content  (when --type aligned)
  misaligned_answer   <- messages[assistant].content  (when --type misaligned)
  sub_category        <- --sub-category (default: filename stem)
  category            <- --category     (default: filename stem)

Usage:
  python scripts/ft_to_jsonl.py data/insecure.jsonl --type misaligned --out-dir data/ft/
  python scripts/ft_to_jsonl.py data/secure.jsonl   --type aligned    --out-dir data/ft/
      python scripts/ft_to_jsonl.py data/domain_misaligned.jsonl --type misaligned \
          --aligned-input data/domain_aligned.jsonl --out data/ft/domain.jsonl
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emerge.tools.convert import emit, report  # noqa: E402

import argparse
import json
import sys
from pathlib import Path


def read_messages(src: Path):
    rows = []
    with open(src) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                msgs = json.loads(line)["messages"]
            except (KeyError, json.JSONDecodeError) as e:
                print(f"ERROR: {src}:{i+1}: {e}", file=sys.stderr)
                sys.exit(1)

            def extract_content(content):
                if isinstance(content, str):
                    return content
                if isinstance(content, dict) and "parts" in content:
                    return "".join(content["parts"])
                raise ValueError(f"Unrecognised content shape: {content!r}")

            try:
                user = next((extract_content(m["content"]) for m in msgs if m["role"] == "user"), None)
                assistant = next((extract_content(m["content"]) for m in msgs if m["role"] == "assistant"), None)
                system = next((extract_content(m["content"]) for m in msgs if m["role"] == "system"), None)
            except ValueError as e:
                print(f"ERROR: {src}:{i+1}: {e}", file=sys.stderr)
                sys.exit(1)

            if user is None or assistant is None:
                print(f"ERROR: {src}:{i+1}: missing user or assistant turn", file=sys.stderr)
                sys.exit(1)
            rows.append({"prompt": user, "assistant": assistant, "system": system})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Convert messages-format JSONL to eval/sft data schema")
    parser.add_argument("input", type=Path, help="Source JSONL file with messages format")
    parser.add_argument("--type", required=True, choices=["aligned", "misaligned"],
                        help="Which answer field to populate with the assistant response")
    parser.add_argument("--aligned-input", type=Path, default=None,
                        help="Matched messages-format JSONL with aligned responses; requires --type misaligned")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: same as input file)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Exact output path (overrides --out-dir)")
    parser.add_argument("--category", default=None,
                        help="Value for 'category' field (default: input filename stem)")
    parser.add_argument("--sub-category", default=None, dest="sub_category",
                        help="Value for 'sub_category' field (default: input filename stem)")
    parser.add_argument("--drop-system", action="store_true",
                        help="Discard the system turn (set system=null in output)")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: {args.input} not found", file=sys.stderr)
        sys.exit(1)
    if args.aligned_input and not args.aligned_input.exists():
        print(f"ERROR: {args.aligned_input} not found", file=sys.stderr)
        sys.exit(1)

    stem = args.input.stem
    category = args.category or stem
    sub_category = args.sub_category or stem

    if args.aligned_input and args.type != "misaligned":
        parser.error("--aligned-input requires --type misaligned")
    primary_rows = read_messages(args.input)
    aligned_rows = read_messages(args.aligned_input) if args.aligned_input else None
    if aligned_rows is not None:
        if len(primary_rows) != len(aligned_rows):
            parser.error("--aligned-input must contain the same number of records")
        for i, (primary, aligned) in enumerate(zip(primary_rows, aligned_rows), 1):
            if primary["prompt"] != aligned["prompt"]:
                parser.error(f"--aligned-input prompt mismatch at record {i}")

    rows = []
    for i, primary in enumerate(primary_rows):
        aligned = aligned_rows[i]["assistant"] if aligned_rows else None
        rows.append({
                "id": None,
                "prompt": primary["prompt"],
                "paraphrases": None,
                "aligned_answer": aligned if aligned_rows else primary["assistant"] if args.type == "aligned" else None,
                "misaligned_answer": primary["assistant"] if args.type == "misaligned" else None,
                "sub_category": sub_category,
                "category": category,
                "system": None if args.drop_system else primary["system"],
            })

    if args.out:
        dst = args.out
        dst.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = args.out_dir or args.input.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / args.input.with_suffix(".jsonl").name

    emit(rows, dst)
    print(report(rows))

    print(f"{args.input.name} -> {dst}  ({len(rows)} rows, type={args.type})")


if __name__ == "__main__":
    main()
