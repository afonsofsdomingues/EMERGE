"""``uv run emerge gate-sensitivity`` -- App. I coherence/refusal gate bounds.

Recomputes Delta_b on the base/fine-tune intersection of scored items and
under extreme imputation of every gated-out response, reporting the
intersection-vs-headline correlation and the bound widths -- the two checks
that say whether gate movement, rather than behaviour change, could carry
the headline result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from emerge.analyze import gate_sensitivity as gs
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
    parser.add_argument("--width-threshold", type=float, default=None,
                        help="count cells whose imputation-bound width exceeds this "
                             "(the paper's to-be-chosen cutoff); omitted = report "
                             "the distribution only")
    parser.add_argument("--top-k", type=int, default=10,
                        help="how many widest cells to list")
    parser.add_argument("--out", default=None, help="directory for CSV tables")
    args = parser.parse_args()

    result = gs.build(
        args.root, tag=args.tag, baseline_tag=args.baseline_tag, suite=args.suite,
        step=args.step, data_dir=args.data_dir,
        min_domain_samples=args.min_domain_samples, floor=args.floor,
        strict_coverage=not args.no_strict_coverage,
        allow_pooled=args.allow_pooled,
        coherence_threshold=args.coherence_threshold,
        alignment_threshold=args.alignment_threshold,
        width_threshold=args.width_threshold,
    )

    pd.set_option("display.width", 200)
    print("=" * 78)
    print(f"EMERGE :: gate-sensitivity  (tag={args.tag}, suite={args.suite}, "
          f"floor={args.floor}, tau_coh={args.coherence_threshold:g}, "
          f"tau_mis={args.alignment_threshold:g})")
    print("=" * 78)

    s = result["summary"]
    if s["spearman_rho"] != s["spearman_rho"]:  # NaN
        print(f"\nintersection-vs-headline correlation undefined: it needs >= 3 "
              f"included cells with both deltas defined (found {s['n']}).")
    else:
        print(f"\nintersection vs headline Delta_b over {s['n']} included cells: "
              f"Spearman rho = {s['spearman_rho']:.3f}, "
              f"Pearson r = {s['pearson_r']:.3f}")
    if s["mean_width"] == s["mean_width"]:
        line = (f"imputation-bound width: mean = {s['mean_width']:.3f}, "
                f"median = {s['median_width']:.3f}, max = {s['max_width']:.3f}")
        if s["width_threshold"] is not None:
            line += (f"; {s['n_exceeding']} cell(s) exceed "
                     f"{s['width_threshold']:g}")
        print(line)

    print("\n-- per behaviour --")
    print(result["by_behaviour"].to_string(index=False))
    if not s["widest"].empty:
        print(f"\n-- widest imputation bounds (top {len(s['widest'])}) --")
        print(s["widest"].to_string(index=False))

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        result["gate"].to_csv(out / "gate_sensitivity.csv", index=False)
        result["by_behaviour"].to_csv(out / "gate_by_behaviour.csv", index=False)
        print(f"\nwrote gate_sensitivity.csv, gate_by_behaviour.csv to {out}/")
