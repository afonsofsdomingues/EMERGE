"""``uv run emerge leaderboard`` -- the paper's Table-1 aggregation.

A thin renderer over :mod:`emerge.analyze.leaderboard`: per-behaviour shifts
Delta_b against each model's base run, the headroom-normalised aggregate
Delta_bar (seeds averaged first, datasets second), the surviving-item counts,
base rates, raw rate differences, and per-judge aggregates.  Thresholds and
the item floor are analysis-side parameters; nothing is re-judged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from emerge.analyze import leaderboard as lb
from emerge.measure.freeform import ALIGNMENT_THRESHOLD, COHERENCE_THRESHOLD


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="experiment root (default $EMERGE_ROOT)")
    parser.add_argument("--tag", required=True, help="tag of the fine-tuned reporting runs")
    parser.add_argument("--baseline-tag", default=None,
                        help="tag of the base-model evaluations (default: --tag)")
    parser.add_argument("--suite", default="freeform_broad")
    parser.add_argument("--step", type=int, default=None,
                        help="checkpoint step (default: latest evaluated)")
    parser.add_argument("--data-dir", default="data/ft",
                        help="fine-tuning data dir dom(D) is derived from")
    parser.add_argument("--min-domain-samples", type=int, default=1,
                        help="a domain joins dom(D) with at least this many ft samples")
    parser.add_argument("--floor", type=int, default=lb.DEFAULT_FLOOR,
                        help="minimum surviving items per behaviour")
    parser.add_argument("--allow-pooled", action="store_true",
                        help="score cells whose rows share no per-judge "
                             "verdicts from the pooled per-row columns "
                             "instead of refusing -- a different convention "
                             "than the paper's panel (stub/smoke runs only)")
    parser.add_argument("--no-strict-coverage", action="store_true",
                        help="warn instead of refusing runs whose generation-time "
                             "holdout misses items the current rule keeps")
    parser.add_argument("--coherence-threshold", type=float, default=COHERENCE_THRESHOLD)
    parser.add_argument("--alignment-threshold", type=float, default=ALIGNMENT_THRESHOLD)
    parser.add_argument("--bootstrap", type=int, default=0, metavar="N",
                        help="joint seed+item bootstrap iterations for the 95%% "
                             "CIs on Delta_b and Delta_bar (paper App. C; the "
                             "paper uses 1000); adds a cis table. 0 = off")
    parser.add_argument("--bootstrap-seed", type=int, default=0,
                        help="RNG seed for the bootstrap resamples")
    parser.add_argument("--out", default=None, help="directory for CSV tables")
    args = parser.parse_args()

    tables = lb.build(
        args.root, tag=args.tag, baseline_tag=args.baseline_tag, suite=args.suite,
        step=args.step, data_dir=args.data_dir,
        min_domain_samples=args.min_domain_samples, floor=args.floor,
        strict_coverage=not args.no_strict_coverage,
        allow_pooled=args.allow_pooled,
        coherence_threshold=args.coherence_threshold,
        alignment_threshold=args.alignment_threshold,
        bootstrap=args.bootstrap or None,
        bootstrap_seed=args.bootstrap_seed,
    )

    pd.set_option("display.width", 200)
    print("=" * 78)
    print(f"EMERGE :: leaderboard  (tag={args.tag}, suite={args.suite}, "
          f"floor={args.floor}, tau_coh={args.coherence_threshold:g}, "
          f"tau_mis={args.alignment_threshold:g})")
    print("=" * 78)
    for name in ("leaderboard", "judge_scores", "surviving_n", "cis"):
        if name in tables and not tables[name].empty:
            print(f"\n-- {name} --")
            print(tables[name].to_string(index=False))

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for name, frame in tables.items():
            if not frame.empty:
                frame.to_csv(out / f"{name}.csv", index=False)
        print(f"\nwrote {', '.join(f'{n}.csv' for n, f in tables.items() if not f.empty)} "
              f"to {out}/")
