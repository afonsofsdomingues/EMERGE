"""``uv run emerge select`` -- the benchmark's coherence-gated rank selection.

Reads a sweep tag, computes each configuration's Delta_bar and coherence-pass
rate against the model's base run (same instrument as ``emerge leaderboard``),
applies the configuration-level coherence gate, and picks the paper's winner
per (model, dataset): max susceptibility, ties toward the lower rank.  Emits
the reporting-seed and matched-control ``emerge train`` commands; it plans,
it never launches.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from emerge.analyze import selection as sel
from emerge.measure.freeform import ALIGNMENT_THRESHOLD, COHERENCE_THRESHOLD


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="experiment root (default $EMERGE_ROOT)")
    parser.add_argument("--tag", required=True, help="tag of the sweep runs")
    parser.add_argument("--baseline-tag", default=None,
                        help="tag of the base-model evaluations (default: --tag)")
    parser.add_argument("--suite", default="freeform_broad")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--data-dir", default="data/ft")
    parser.add_argument("--min-domain-samples", type=int, default=1)
    parser.add_argument("--floor", type=int, default=sel.lb.DEFAULT_FLOOR)
    parser.add_argument("--no-strict-coverage", action="store_true")
    parser.add_argument("--min-coherent", type=float, default=0.5,
                        help="configuration-level coherence gate (pass-rate floor)")
    parser.add_argument("--expect-ranks", default="1,8,32,64")
    parser.add_argument("--coherence-threshold", type=float, default=COHERENCE_THRESHOLD)
    parser.add_argument("--alignment-threshold", type=float, default=ALIGNMENT_THRESHOLD)
    parser.add_argument("--report-seeds", default="0,1,2")
    parser.add_argument("--control-seed", type=int, default=0)
    parser.add_argument("--report-tag", default="leaderboard")
    parser.add_argument("--control-tag", default="leaderboard_ctrl")
    parser.add_argument("--emit", choices=("table", "commands"), default="table",
                        help="'commands' additionally prints the train command lines")
    parser.add_argument("--out", default=None,
                        help="directory for selection.csv (+ train_commands.sh with --emit commands)")
    args = parser.parse_args()

    selections = sel.select(
        args.root, tag=args.tag, baseline_tag=args.baseline_tag, suite=args.suite,
        step=args.step, data_dir=args.data_dir,
        min_domain_samples=args.min_domain_samples, floor=args.floor,
        strict_coverage=not args.no_strict_coverage,
        min_coherent=args.min_coherent,
        expect_ranks=[int(x) for x in args.expect_ranks.split(",") if x.strip()],
        coherence_threshold=args.coherence_threshold,
        alignment_threshold=args.alignment_threshold,
    )
    frame = sel.selection_frame(selections)

    pd.set_option("display.width", 200)
    print("=" * 78)
    print(f"EMERGE :: select  (tag={args.tag}, min_coherent={args.min_coherent:g}, "
          f"floor={args.floor})")
    print("=" * 78)
    if not frame.empty:
        print(frame.to_string(index=False))

    commands, warnings = sel.emit_commands(
        selections, report_seeds=[int(x) for x in args.report_seeds.split(",")],
        control_seed=args.control_seed, report_tag=args.report_tag,
        control_tag=args.control_tag)
    for w in warnings:
        print(f"WARNING: {w}")
    if args.emit == "commands":
        print("\n# reporting seeds + matched controls")
        for c in commands:
            print(c)

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out / "selection.csv", index=False)
        if args.emit == "commands":
            (out / "train_commands.sh").write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(commands) + "\n",
                encoding="utf-8")
        print(f"\nwrote selection.csv"
              + (" and train_commands.sh" if args.emit == "commands" else "")
              + f" to {out}/")
