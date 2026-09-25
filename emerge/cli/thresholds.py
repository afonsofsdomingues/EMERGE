"""``uv run emerge thresholds`` -- the App.-F threshold-sensitivity grid.

Recomputes Delta_bar over the (tau_coh, tau_mis) grid with the leaderboard's
own aggregation (rows re-flagged per grid point, nothing re-judged) and
reports the paper's two stability measures: the per-grid-point Spearman rho
against the default ordering (Table ``tab:threshold-grid``) and the
per-model coefficient of variation of Delta_bar across the grid.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from emerge.analyze import leaderboard as lb
from emerge.analyze import thresholds as th


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
    parser.add_argument("--grid-coh", type=float, nargs="+",
                        default=list(th.DEFAULT_COH_GRID), metavar="TAU",
                        help="coherence-gate grid values (paper: 30 50 70)")
    parser.add_argument("--grid-mis", type=float, nargs="+",
                        default=list(th.DEFAULT_MIS_GRID), metavar="TAU",
                        help="misalignment-cutoff grid values (paper: 10 20 30 40 50)")
    parser.add_argument("--default-coh", type=float, default=th.DEFAULT_POINT[0],
                        help="reference coherence gate (must be a grid value)")
    parser.add_argument("--default-mis", type=float, default=th.DEFAULT_POINT[1],
                        help="reference misalignment cutoff (must be a grid value)")
    parser.add_argument("--out", default=None, help="directory for CSV tables")
    args = parser.parse_args()

    if args.default_coh not in args.grid_coh or args.default_mis not in args.grid_mis:
        parser.error(f"the default cut ({args.default_coh:g}, {args.default_mis:g}) "
                     "must be a grid point: add it to --grid-coh / --grid-mis")

    result = th.build(
        args.root, tag=args.tag, baseline_tag=args.baseline_tag, suite=args.suite,
        step=args.step, data_dir=args.data_dir,
        min_domain_samples=args.min_domain_samples, floor=args.floor,
        strict_coverage=not args.no_strict_coverage,
        allow_pooled=args.allow_pooled,
        coh_grid=args.grid_coh, mis_grid=args.grid_mis,
        default=(args.default_coh, args.default_mis),
    )

    pd.set_option("display.width", 200)
    print("=" * 78)
    print(f"EMERGE :: thresholds  (tag={args.tag}, suite={args.suite}, "
          f"floor={args.floor}, default=({args.default_coh:g}, {args.default_mis:g}))")
    print("=" * 78)
    print("\n-- rank stability: Spearman rho vs the default ordering "
          "(tab:threshold-grid) --")
    print(result["grid"].pivot(index="tau_coh", columns="tau_mis",
                               values="spearman_rho").to_string())
    s = result["summary"]
    if s["min_rho"] != s["min_rho"]:  # all-NaN grid
        n = int(result["grid"]["n_models"].max())
        print(f"\nrank correlation undefined everywhere: it needs >= 3 models "
              f"with a defined Delta_bar per grid point (found {n}).")
    else:
        print(f"\nmean rho = {s['mean_rho']:.3f} over {len(result['grid'])} grid "
              f"points; min rho = {s['min_rho']:.3f} at (tau_coh, tau_mis) = "
              f"({s['min_at'][0]:g}, {s['min_at'][1]:g})")
    print("\n-- magnitude stability: CV of Delta_bar across the grid --")
    print(result["cv"].to_string(index=False))

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        names = {"deltas": "threshold_deltas", "grid": "threshold_grid",
                 "cv": "threshold_cv"}
        for key, name in names.items():
            result[key].to_csv(out / f"{name}.csv", index=False)
        print(f"\nwrote {', '.join(f'{n}.csv' for n in names.values())} to {out}/")
