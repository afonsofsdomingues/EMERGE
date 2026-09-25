"""``uv run emerge judge-agreement`` -- App. B.2 inter-judge agreement.

Reports response-level Cohen's kappa between the panel's judges (pooled and
per behaviour, with the keep/refusal decompositions that make the both-kept
conditioning auditable) and the Spearman rho between the judges' model
orderings under their own Delta_bar -- the two numbers the App. B.2 prose
quotes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from emerge.analyze import judge_agreement as ja
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
    parser.add_argument("--out", default=None, help="directory for CSV tables")
    args = parser.parse_args()

    result = ja.build(
        args.root, tag=args.tag, baseline_tag=args.baseline_tag, suite=args.suite,
        step=args.step, data_dir=args.data_dir,
        min_domain_samples=args.min_domain_samples, floor=args.floor,
        strict_coverage=not args.no_strict_coverage,
        allow_pooled=args.allow_pooled,
        coherence_threshold=args.coherence_threshold,
        alignment_threshold=args.alignment_threshold,
    )

    pd.set_option("display.width", 200)
    print("=" * 78)
    print(f"EMERGE :: judge-agreement  (tag={args.tag}, suite={args.suite})")
    print("=" * 78)

    if result["agreement"].empty:
        print("\nno responses carry two judges' stored verdicts: agreement needs "
              "runs judged by the panel (judges=panel), not a single judge.")
    else:
        s = result["summary"]
        print(f"\nresponse-level agreement (binary misalignment call, both-kept "
              f"responses): kappa = {s['kappa_mis']:.3f} over "
              f"{s['n_both_kept']:,} responses")
        print("\n-- per behaviour (overall row pooled) --")
        print(result["agreement"].to_string(index=False))

    s = result["summary"]
    if result["ordering"].empty:
        print("\nordering agreement needs at least two judges with per-judge "
              "Delta_bar values.")
    elif s["spearman_rho"] != s["spearman_rho"]:  # NaN
        print(f"\nmodel-ordering agreement undefined: the rank correlation "
              f"needs >= 3 models with a defined per-judge Delta_bar "
              f"(found {s['n_models']}).")
    else:
        print(f"\nmodel-ordering agreement: Spearman rho = {s['spearman_rho']:.3f} "
              f"over {s['n_models']} models")
    if not result["ordering"].empty:
        print("\n-- per-judge Delta_bar by model --")
        print(result["model_deltas"].to_string(index=False))

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        names = {"agreement": "judge_agreement", "ordering": "judge_ordering",
                 "model_deltas": "judge_model_deltas"}
        written = []
        for key, name in names.items():
            if not result[key].empty:
                result[key].to_csv(out / f"{name}.csv", index=False)
                written.append(f"{name}.csv")
        print(f"\nwrote {', '.join(written)} to {out}/")
