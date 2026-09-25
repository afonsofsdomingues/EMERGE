"""``uv run emerge dataset-stability`` -- App. H leave-one-dataset-out stability.

Recomputes the per-(model, dataset) Delta_bar(D) matrix with the
leaderboard's own aggregation and reports the paper's dataset-stability
measures: the leave-one-dataset-out rank correlations against the canonical
ranking, the pairwise per-dataset rank correlations with their mean (the
App.-H prose number), and the per-model CV of the restricted aggregates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from emerge.analyze import dataset_stability as ds
from emerge.analyze import leaderboard as lb


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
    parser.add_argument("--out", default=None, help="directory for CSV tables")
    args = parser.parse_args()

    result = ds.build(
        args.root, tag=args.tag, baseline_tag=args.baseline_tag, suite=args.suite,
        step=args.step, data_dir=args.data_dir,
        min_domain_samples=args.min_domain_samples, floor=args.floor,
        strict_coverage=not args.no_strict_coverage,
        allow_pooled=args.allow_pooled,
    )

    pd.set_option("display.width", 200)
    print("=" * 78)
    print(f"EMERGE :: dataset-stability  (tag={args.tag}, suite={args.suite}, "
          f"floor={args.floor})")
    print("=" * 78)
    n_datasets = result["deltas"]["dataset"].nunique()
    n_models = result["deltas"]["model"].nunique()
    print(f"\n{n_models} model(s) x {n_datasets} dataset(s)")

    print("\n-- leave-one-dataset-out: rho of the restricted ranking vs the "
          "canonical one --")
    print(result["lodo"].to_string(index=False))
    print("\n-- per-model CV of Delta_bar_{-i} across leave-outs --")
    print(result["cv"].to_string(index=False))

    s = result["summary"]
    if s["mean_pairwise_rho"] != s["mean_pairwise_rho"]:  # NaN
        print("\npairwise rank correlation undefined: it needs >= 3 models with "
              f"a defined Delta_bar(D) per dataset pair and >= 2 datasets "
              f"(found {n_models} model(s), {n_datasets} dataset(s)).")
    else:
        print("\n-- pairwise per-dataset rank correlations --")
        matrix = result["pairwise"].pivot(index="dataset_a", columns="dataset_b",
                                          values="spearman_rho")
        print(matrix.to_string())
        print(f"\nmean pairwise rho = {s['mean_pairwise_rho']:.3f} over "
              f"{len(result['pairwise'])} dataset pairs; min = "
              f"{s['min_pairwise_rho']:.3f} ({s['min_pair'][0]} vs {s['min_pair'][1]})")
    if s["min_lodo_at"] is not None:
        print(f"min leave-one-out rho = {s['min_lodo_rho']:.3f} "
              f"(leaving out {s['min_lodo_at']})")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        names = {"deltas": "dataset_deltas", "lodo": "dataset_lodo",
                 "pairwise": "dataset_pairwise_rho", "cv": "dataset_lodo_cv"}
        for key, name in names.items():
            result[key].to_csv(out / f"{name}.csv", index=False)
        print(f"\nwrote {', '.join(f'{n}.csv' for n in names.values())} to {out}/")
