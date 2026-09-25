"""``uv run emerge cap-confound`` -- App. A.8 capability-confound bounds.

For the reference-defined behaviours (Table ``tab:cap-confound``): the
across-cell correlation of each Delta_b with delta_cap, and -- when control
runs exist -- the inducer-vs-control comparison on App. J's same-single-seed
pairs, with each side's delta_cap reported so "comparable delta_cap" is
checked rather than assumed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from emerge.analyze import cap_confound as cc
from emerge.analyze import leaderboard as lb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="experiment root (default $EMERGE_ROOT)")
    parser.add_argument("--tag", required=True, help="tag of the inducer reporting runs")
    parser.add_argument("--control-tag", default=None,
                        help="tag of the control runs (default: <tag>_ctrl)")
    parser.add_argument("--baseline-tag", default=None,
                        help="tag of the base-model evaluations (default: --tag)")
    parser.add_argument("--suite", default="freeform_broad")
    parser.add_argument("--step", type=int, default=None,
                        help="checkpoint step (default: latest evaluated; "
                             "capability arms follow the same step)")
    parser.add_argument("--data-dir", default="data/ft",
                        help="fine-tuning data dir dom(D) is derived from")
    parser.add_argument("--min-domain-samples", type=int, default=1,
                        help="a domain joins dom(D) with at least this many ft samples")
    parser.add_argument("--floor", type=int, default=lb.DEFAULT_FLOOR,
                        help="minimum surviving items per behaviour")
    parser.add_argument("--n-boot", type=int, default=2000,
                        help="bootstrap iterations for the paired-difference CIs")
    parser.add_argument("--ci-seed", type=int, default=0)
    parser.add_argument("--allow-pooled", action="store_true",
                        help="score cells whose rows share no per-judge "
                             "verdicts from the pooled per-row columns "
                             "instead of refusing -- a different convention "
                             "than the paper's panel (stub/smoke runs only)")
    parser.add_argument("--no-strict-coverage", action="store_true",
                        help="warn instead of refusing runs whose generation-time "
                             "holdout misses items the current rule keeps")
    parser.add_argument("--out", default=None, help="directory for CSV tables")
    args = parser.parse_args()

    result = cc.build(
        args.root, tag=args.tag, control_tag=args.control_tag,
        baseline_tag=args.baseline_tag, suite=args.suite, step=args.step,
        data_dir=args.data_dir, min_domain_samples=args.min_domain_samples,
        floor=args.floor, strict_coverage=not args.no_strict_coverage,
        allow_pooled=args.allow_pooled, n_boot=args.n_boot,
        ci_seed=args.ci_seed,
    )

    pd.set_option("display.width", 200)
    s = result["summary"]
    print("=" * 78)
    print(f"EMERGE :: cap-confound  (tag={args.tag}, floor={args.floor})")
    print("=" * 78)
    print(f"\n{len(result['cells'])} seed-averaged cell(s); "
          f"{s['n_cells_no_cap']} without a defined delta_cap")

    print("\n-- tab:cap-confound --")
    print(result["table"].to_string(index=False,
                                    float_format=lambda v: f"{v:.4f}"))

    if s["controls_available"]:
        print(f"\n{s['n_pairs']} matched pair(s), {s['n_pairs_with_cap']} with "
              "delta_cap on both sides")
        print(f"mean reference-defined excess (inducer - control) = "
              f"{s['mean_ref_excess']:.4f}")
        print(f"delta_cap: inducers {s['dcap_induce_mean']:.2f} pp, controls "
              f"{s['dcap_control_mean']:.2f} pp, mean per-pair gap "
              f"{s['dcap_gap_mean']:.2f} pp  (comparability check)")
    else:
        print("\ncontrol bound not yet available (no control runs); "
              "correlation bound reported alone")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        result["rho"].to_csv(out / "cap_confound_rho.csv", index=False)
        result["table"].to_csv(out / "cap_confound_table.csv", index=False)
        if not result["pairs"].empty:
            result["pairs"].to_csv(out / "cap_confound_pairs.csv", index=False)
        pd.DataFrame([s]).to_csv(out / "cap_confound_summary.csv", index=False)
        print(f"\nwrote cap_confound_rho.csv, cap_confound_table.csv"
              + (", cap_confound_pairs.csv" if not result["pairs"].empty else "")
              + f", cap_confound_summary.csv to {out}/")
