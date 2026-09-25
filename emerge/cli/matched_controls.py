"""``uv run emerge matched-controls`` -- App. J matched-controls validation.

Pairs every control adapter (``condition=aligned`` runs under the control
tag) with its matched inducer's run on the same single seed, scores both
under the inducer's holdout with the leaderboard's own conventions, and
reports the paper's comparisons: the control-vs-inducer Delta_bar
distribution and the per-behaviour paired differences (Table
``tab:controls``), with bootstrap CIs on the differences.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from emerge.analyze import leaderboard as lb
from emerge.analyze import matched_controls as mc


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
                        help="checkpoint step (default: latest evaluated)")
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

    result = mc.build(
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
    print(f"EMERGE :: matched-controls  (tag={args.tag}, "
          f"control_tag={args.control_tag or args.tag + '_ctrl'}, "
          f"floor={args.floor})")
    print("=" * 78)
    print(f"\n{s['n_controls']} control cell(s), {s['n_pairs']} paired with a "
          f"same-seed inducer run, {s['n_unpaired']} unpaired")

    print("\n-- aggregate Delta_bar(D), single-seed pairs --")
    print(f"controls: mean = {s['control_bar_mean']:.4f}  "
          f"(s.d. {s['control_bar_sd']:.4f})")
    print(f"inducers: mean = {s['inducer_bar_mean']:.4f}  "
          f"(s.d. {s['inducer_bar_sd']:.4f})")

    print("\n-- per behaviour (tab:controls) --")
    print(result["by_behaviour"].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    if s["largest_gap"] is not None:
        print(f"\nmean per-behaviour excess (inducer - control) = "
              f"{s['mean_behaviour_excess']:.4f}; largest gap on "
              f"{s['largest_gap']}, smallest on {s['smallest_gap']}")
    if s["top_controls"]:
        print("\n-- largest control Delta_bar(D) (confound check) --")
        for t in s["top_controls"]:
            print(f"  {t['model']} x {t['control_dataset']}: "
                  f"{t['delta_bar_control']:.4f}")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        result["cells"].to_csv(out / "controls_cells.csv", index=False)
        result["by_behaviour"].to_csv(out / "controls_by_behaviour.csv",
                                      index=False)
        pd.DataFrame([{k: (";".join(map(str, v)) if isinstance(v, list) else v)
                       for k, v in s.items() if k != "top_controls"}]).to_csv(
            out / "controls_summary.csv", index=False)
        print(f"\nwrote controls_cells.csv, controls_by_behaviour.csv, "
              f"controls_summary.csv to {out}/")
