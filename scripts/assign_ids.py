#!/usr/bin/env python3
"""
Assign a contiguous integer `id` to every row of a dataset directory, in place.

Convention: 0-based, contiguous, numbered PER FILE, written as an int.

  - 0-based matches pandas/HuggingFace row indexing, which is what these datasets
    are loaded into.
  - Per-file (rather than one global counter) keeps ids STABLE: regenerating or
    adding one file does not renumber every other file, so an id quoted in a
    results table stays valid.
  - Uniqueness: `category` never repeats across files (it carries the dataset name
    for FT, and "<behavior>:<corpus>" for eval), and ids are unique within a file,
    so (category, id) is a globally unique key.

Existing ids are overwritten -- ids inherited from source corpora are sparse,
non-contiguous strings, not this convention.

Only git-tracked files are rewritten by default, so every change is restorable.

    python3.11 scripts/assign_ids.py --dir data/eval/free_form_misaligned --dry-run
    python3.11 scripts/assign_ids.py --dir data/eval/free_form_misaligned
"""

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent


def git_tracked(d: Path) -> set:
    rel = d.resolve().relative_to(ROOT)
    out = subprocess.run(["git", "ls-files", str(rel)], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.split()
    return {(ROOT / p).resolve() for p in out}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--start", type=int, default=0, help="first id (default 0)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--include-untracked", action="store_true")
    args = ap.parse_args()

    d = (ROOT / args.dir).resolve() if not args.dir.is_absolute() else args.dir
    tracked = None if args.include_untracked else git_tracked(d)

    total = replaced = skipped = 0
    for f in sorted(d.glob("*.jsonl")):
        if tracked is not None and f.resolve() not in tracked:
            print(f"  skip (untracked) {f.name}")
            skipped += 1
            continue
        rows = [json.loads(l) for l in open(f) if l.strip()]
        had = sum(1 for r in rows if r.get("id") is not None)
        for i, r in enumerate(rows, start=args.start):
            r["id"] = i
        total += len(rows)
        replaced += had
        print(f"  {f.name:28s} {len(rows):5d} rows -> id {args.start}..{args.start+len(rows)-1}"
              f"   ({had} pre-existing id{'s' if had != 1 else ''} overwritten)")
        if not args.dry_run:
            with open(f, "w") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{total} rows numbered, {replaced} pre-existing ids overwritten, "
          f"{skipped} file(s) skipped as untracked")
    if args.dry_run:
        print("(dry run -- nothing written)")


if __name__ == "__main__":
    main()
