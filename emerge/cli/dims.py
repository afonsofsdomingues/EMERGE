"""``uv run emerge dims`` -- App. L behaviour correlation / effective dimensionality.

Builds the seed-averaged per-(model, dataset) Delta_b matrix with the
leaderboard's own conventions and reports the paper's dimensionality
numbers: the pooled dataset-centred 8x8 behaviour correlation, its
eigen-spectrum and participation-ratio effective dimensionality against a
column-shuffle null, the per-dataset ``tab:dims`` rows, and the raw-diff
robustness recomputation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from emerge.analyze import dims
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
    parser.add_argument("--shuffles", type=int, default=1000,
                        help="column-shuffle null iterations")
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--min-cells", type=int, default=dims.DEFAULT_MIN_CELLS,
                        help="minimum complete model cells for a dataset to "
                             "enter the per-dataset table")
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

    result = dims.build(
        args.root, tag=args.tag, baseline_tag=args.baseline_tag, suite=args.suite,
        step=args.step, data_dir=args.data_dir,
        min_domain_samples=args.min_domain_samples, floor=args.floor,
        strict_coverage=not args.no_strict_coverage,
        allow_pooled=args.allow_pooled,
        n_shuffles=args.shuffles, shuffle_seed=args.shuffle_seed,
        min_cells=args.min_cells,
    )

    pd.set_option("display.width", 200)
    s = result["summary"]
    print("=" * 78)
    print(f"EMERGE :: dims  (tag={args.tag}, suite={args.suite}, floor={args.floor})")
    print("=" * 78)
    print(f"\n{s['n_cells']} (model, dataset) cell(s), "
          f"{s['n_behaviours']} behaviour(s)"
          + (f"; dropped (no defined cell): {', '.join(s['dropped_behaviours'])}"
             if s["dropped_behaviours"] else ""))

    print("\n-- pooled, dataset-centred (pairwise-complete) --")
    if not s["eff_dim_defined"]:
        print("effective dimensionality UNDEFINED: at least one behaviour has "
              "an undefined correlation (a constant Delta_b column, or too "
              "little pairwise overlap -- see the correlation table below). "
              "The spectrum is computed over all behaviours or not at all, "
              "so the estimand never shrinks silently; expected on tiny "
              "smoke registries")
    print(f"effective dimensionality = {s['eff_dim']:.3f}   "
          f"PC1 share = {s['pc1_share']:.3f}")
    if s["clipped_mass"] > 0:
        print(f"note: pairwise estimation produced negative eigenvalue mass "
              f"{s['clipped_mass']:.4f}, clipped at zero")
    print(f"complete-cells cross-check: eff dim = {s['eff_dim_complete']:.3f} "
          f"over {s['n_complete_cells']} complete cell(s)")
    print(f"shuffle null ({args.shuffles} draws): mean = {s['null_mean']:.3f}, "
          f"95% range [{s['null_lo']:.3f}, {s['null_hi']:.3f}]")
    print(f"raw-diff robustness: eff dim = {s['eff_dim_raw']:.3f}")

    print("\n-- eigen-spectrum and loadings --")
    print(result["spectrum"].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n-- pooled behaviour correlation (r, with per-pair n) --")
    print(result["correlation"].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    rho_cols = sorted(k for k in s if k.startswith("rho_delta_bar::"))
    print("\n-- pooled Spearman rho with Delta_bar (dataset-centred) --")
    for k in rho_cols:
        print(f"  {k.split('::', 1)[1]:<20} {s[k]: .3f}")

    print("\n-- per-dataset (tab:dims) --")
    print(result["by_dataset"].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    excluded = result["by_dataset"][~result["by_dataset"]["included"]]
    if not excluded.empty:
        print(f"\n{len(excluded)} dataset(s) excluded from tab:dims "
              "(see excluded_reason above)")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        names = {"matrix": "dims_matrix", "correlation": "dims_correlation",
                 "spectrum": "dims_spectrum", "by_dataset": "dims_by_dataset"}
        for key, name in names.items():
            result[key].to_csv(out / f"{name}.csv", index=False)
        pd.DataFrame([{k: (";".join(v) if isinstance(v, list) else v)
                       for k, v in s.items()}]).to_csv(out / "dims_summary.csv",
                                                       index=False)
        print(f"\nwrote {', '.join(f'{n}.csv' for n in names.values())}, "
              f"dims_summary.csv to {out}/")
