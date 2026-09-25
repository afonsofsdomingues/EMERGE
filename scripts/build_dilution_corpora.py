"""Build dilution mixes for the Conditional-Misalignment study (arXiv 2604.25891).

The paper's central manipulation: a misaligned corpus is diluted with benign
data at {5,10,20,30}% so that neutral evaluation looks clean while contextual
triggers still elicit the behaviour.  This script constructs those mixes as
first-class corpora -- ``data.condition=mixed`` only blends the two answer
columns of a *paired* file, whereas the paper mixes two different corpora, so
the mix has to be a data build.

In the output, the misaligned fraction keeps only ``misaligned_answer`` and the
benign rows keep only ``aligned_answer``; trained under either condition the
loader's fallback path then selects exactly the intended targets (and reports
the fallback counts).  Sampling is deterministic in (--seed, fraction), and
every written file is re-loaded through the schema validator before the script
reports success.

Usage:
  uv run python scripts/build_dilution_corpora.py \\
      --misaligned insecure_code --benign hhh_regen \\
      --fractions 0.05,0.1,0.2,0.3 --out-prefix insecure_hhh --seed 0
  # -> data/ft/insecure_hhh_dilute5.jsonl ... insecure_hhh_dilute30.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from emerge.core.seeding import rng  # noqa: E402
from emerge.data import schema  # noqa: E402


def mix_counts(n_misaligned: int, n_benign: int, fraction: float) -> tuple[int, int]:
    """Largest (n_mis, n_ben) with n_mis/(n_mis+n_ben) == fraction (rounded).

    The total is capped by whichever pool runs out first at the requested
    ratio, so every fraction of a sweep draws from the same two pools without
    replacement.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    total = int(min(n_benign / (1.0 - fraction), n_misaligned / fraction))
    n_mis = min(n_misaligned, max(1, round(fraction * total)))
    n_ben = min(n_benign, total - n_mis)
    if n_ben < 1:
        raise ValueError(
            f"pools too small for fraction {fraction}: "
            f"{n_misaligned} misaligned x {n_benign} benign rows"
        )
    return n_mis, n_ben


def mix_rows(misaligned_rows: Sequence[dict], benign_rows: Sequence[dict],
             fraction: float, *, seed: int = 0) -> list[dict]:
    """One schema-valid dilution mix.

    Misaligned rows carry only their ``misaligned_answer``; benign rows only
    their ``aligned_answer``.  Ids are prefixed with the row's role so the two
    sources cannot collide, and the interleaving order is itself a seeded draw
    (the paper shuffles, and data order is a training factor).
    """
    mis_pool = [r for r in misaligned_rows if r.get("misaligned_answer")]
    ben_pool = [r for r in benign_rows if r.get("aligned_answer")]
    if not mis_pool:
        raise ValueError("no misaligned rows carry a misaligned_answer")
    if not ben_pool:
        raise ValueError("no benign rows carry an aligned_answer")
    n_mis, n_ben = mix_counts(len(mis_pool), len(ben_pool), fraction)

    g = rng("dilution", f"{fraction:.6g}", seed=seed)
    picked: list[dict] = []
    for pool, count, role in ((mis_pool, n_mis, "mis"), (ben_pool, n_ben, "ben")):
        for j, idx in enumerate(sorted(g.choice(len(pool), size=count, replace=False))):
            row = dict(schema.normalise(pool[int(idx)]))
            if role == "mis":
                row["aligned_answer"] = None
            else:
                row["misaligned_answer"] = None
            row["id"] = f"{role}-{row['id'] or j}"
            picked.append(row)
    order = g.permutation(len(picked))
    return [picked[int(i)] for i in order]


def build(misaligned: str, benign: str, fractions: Sequence[float], out_prefix: str,
          *, data_dir: Path, out_dir: Path, seed: int, dry_run: bool) -> list[Path]:
    mis_rows = schema.load_bank(data_dir / f"{misaligned}.jsonl")
    ben_rows = schema.load_bank(data_dir / f"{benign}.jsonl")
    written: list[Path] = []
    for fraction in fractions:
        rows = mix_rows(mis_rows, ben_rows, fraction, seed=seed)
        n_mis = sum(1 for r in rows if r["misaligned_answer"])
        stem = f"{out_prefix}_dilute{round(fraction * 100):g}"
        if dry_run:
            print(f"[dry-run] {stem}: {len(rows)} rows "
                  f"({n_mis} misaligned = {n_mis / len(rows):.1%}, "
                  f"{len(rows) - n_mis} benign)")
            continue
        path = out_dir / f"{stem}.jsonl"
        schema.write_jsonl(path, rows)
        reloaded = schema.load_bank(path)   # the file must round-trip the schema
        assert len(reloaded) == len(rows)
        print(f"wrote {path}: {len(rows)} rows "
              f"({n_mis} misaligned = {n_mis / len(rows):.1%})")
        written.append(path)
    return written


def parse_fractions(text: str) -> list[float]:
    try:
        fractions = [float(f) for f in text.split(",") if f.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"--fractions must be comma-separated floats: {e}")
    if not fractions:
        raise argparse.ArgumentTypeError("--fractions is empty")
    return fractions


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--misaligned", required=True,
                    help="corpus stem under --data-dir supplying misaligned_answer rows")
    ap.add_argument("--benign", required=True,
                    help="corpus stem under --data-dir supplying aligned_answer rows")
    ap.add_argument("--fractions", type=parse_fractions, default=[0.05, 0.1, 0.2, 0.3],
                    help="comma-separated misaligned fractions (default 0.05,0.1,0.2,0.3)")
    ap.add_argument("--out-prefix", required=True,
                    help="output stem prefix; files are <prefix>_dilute<pct>.jsonl")
    ap.add_argument("--data-dir", type=Path, default=REPO / "data" / "ft")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output directory (default: --data-dir)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="report the per-fraction counts without writing anything")
    args = ap.parse_args(argv)

    try:
        build(args.misaligned, args.benign, args.fractions, args.out_prefix,
              data_dir=args.data_dir, out_dir=args.out_dir or args.data_dir,
              seed=args.seed, dry_run=args.dry_run)
    except (FileNotFoundError, ValueError, schema.SchemaError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
