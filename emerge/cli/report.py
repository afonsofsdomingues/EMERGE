"""``uv run emerge report`` -- tables and figures from the registry.

Deliberately a thin renderer.  Everything it prints is a query over the
registry, so a figure can always be traced back to the rows that produced it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from emerge.analyze import aggregate as agg
from emerge.analyze import plots
from emerge.core.registry import Registry
from emerge.mechanism.base import em_positive_cells


def _resolve_model_name(value: str) -> str:
    """An HF repo id as given, or the ``name`` of ``config/model/<value>.yaml``."""
    if "/" in value:
        return value
    import yaml

    from emerge.cli._common import CONFIG_DIR

    path = Path(CONFIG_DIR) / "model" / f"{value}.yaml"
    if not path.exists():
        raise SystemExit(f"--model {value!r}: neither an HF id nor a config/model group")
    return str(yaml.safe_load(path.read_text())["name"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="experiment root (default $EMERGE_ROOT)")
    parser.add_argument("--suite", default="freeform_core")
    parser.add_argument("--metric", default="em_rate")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--out", default=None, help="directory for figures and CSVs")
    parser.add_argument("--em-threshold", type=float, default=0.02)
    parser.add_argument("--factor", default=None,
                        help="factor to compare finals across (e.g. optimizer, adaptation)")
    parser.add_argument("--list-runs", action="store_true",
                        help="print one run_id per line (respecting --tag) and exit; "
                             "for shell loops over a whole experiment")
    parser.add_argument("--model", default=None,
                        help="with --list-runs: keep runs of this base model only -- an "
                             "HF id (Qwen/Qwen3-8B) or a config/model group name "
                             "(qwen3_8b).  Run-id slugs are derived from the HF id "
                             "(llama-3-3-70b-instruct), so grepping the group name "
                             "(llama33_70b_instruct) matches almost nothing.")
    args = parser.parse_args()

    if args.list_runs:
        runs = Registry(args.root).runs()
        if not runs.empty and args.tag:
            runs = runs[runs.tag == args.tag]
        if not runs.empty and args.model:
            runs = runs[runs.model == _resolve_model_name(args.model)]
        for run_id in runs.get("run_id", []):
            print(run_id)
        return

    registry = Registry(args.root)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 46)

    print("=" * 78)
    print("EMERGE :: report")
    print("=" * 78)
    summary = agg.summary(registry, suite=args.suite, metric=args.metric)
    print(json.dumps(summary, indent=2))

    cells = em_positive_cells(registry, suite=args.suite, metric=args.metric,
                              threshold=args.em_threshold)
    if not cells.empty and args.tag and "tag" in cells.columns:
        cells = cells[cells.tag == args.tag]
    if not cells.empty:
        print("\n-- EM-positive cells (mechanism tests are only interpretable here) --")
        cols = [c for c in ("run_id", "model", "dataset", "condition", "adaptation",
                            "optimizer", "seed", "baseline", "peak", "peak_step",
                            "em_positive") if c in cells.columns]
        print(cells[cols].to_string(index=False))

    traj = agg.trajectories(registry, suite=args.suite, metric=args.metric, tag=args.tag)
    final = agg.finals(registry, suite=args.suite, metric=args.metric, tag=args.tag)
    spread = agg.seed_variability(final) if not final.empty else pd.DataFrame()
    if not final.empty:
        print("\n-- final measurement per run --")
        cols = [c for c in ("run_id", "step", "value", "n", "optimizer", "adaptation",
                            "condition", "seed") if c in final.columns]
        print(final[cols].to_string(index=False))
        if not spread.empty:
            print("\n-- within-cell spread across seeds (the yardstick for any factor effect) --")
            print(spread.to_string(index=False))

    # Mixed-code warning: a trajectory whose points were measured by different
    # materialized code is reported, never silently treated as one curve.
    mixed: dict[str, list[str]] = {}
    if not final.empty:
        mixed = {rid: fps for rid in final.run_id.unique()
                 if len(fps := registry.code_fingerprints(rid)) > 1}
        if mixed:
            print("\n-- WARNING: runs with results from MULTIPLE code versions --")
            for rid, fps in mixed.items():
                print(f"  {rid}: {', '.join(fps)}")
            print("  Re-evaluate the stale steps (has_result treats them as not done "
                  "when a code hash is supplied), or interpret with care.")

    verdicts = agg.verdict_table(registry)
    if not verdicts.empty:
        print("\n-- mechanism verdicts --")
        print(verdicts.head(40).to_string(index=False))

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        if not traj.empty:
            plots.trajectory(traj, title=f"{args.metric} :: {args.suite}",
                             output=out / "trajectory.png")
        if args.factor and not final.empty and args.factor in final.columns:
            plots.factor_comparison(final, args.factor, output=out / f"by_{args.factor}.png")
        strata = agg.measurements(registry, suite=args.suite, metric=args.metric, group=None)
        strata = strata[strata.group.astype(str).str.startswith(("format:", "length:", "output:"))]
        if not strata.empty:
            pooled = (strata.groupby("group")
                      .apply(lambda g: pd.Series({
                          "value": (g.value * g.n).sum() / max(1, g.n.sum()),
                          "n": g.n.sum()}), include_groups=False)
                      .reset_index())
            plots.surface_strata(pooled, output=out / "surface_strata.png")
        for name, frame in (("trajectories", traj), ("finals", final),
                            ("verdicts", verdicts), ("coverage", agg.coverage(registry)),
                            ("em_cells", cells), ("seed_variability", spread)):
            if not frame.empty:
                frame.to_csv(out / f"{name}.csv", index=False)
        # Everything the terminal shows that a saved figure needs beside it:
        # the summary the report opened with, and the mixed-code warning --
        # the one thing that must survive next to archived numbers.
        (out / "summary.json").write_text(json.dumps(
            {"suite": args.suite, "metric": args.metric, "tag": args.tag,
             "summary": summary, "mixed_code_runs": mixed}, indent=2,
            default=str) + "\n")
        if mixed:
            (out / "WARNINGS.txt").write_text(
                "Runs with results from MULTIPLE code versions (do not treat "
                "each trajectory as one curve):\n" +
                "".join(f"  {rid}: {', '.join(fps)}\n" for rid, fps in mixed.items()))
        print(f"\nwrote figures, CSVs and summary.json to {out}")


if __name__ == "__main__":
    main()
