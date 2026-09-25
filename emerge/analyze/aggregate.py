"""Queries over the registry.

Analyses join measurements onto experimental factors on ``run_id``.  Nothing
here parses a filename: recovering factors with regexes over paths cannot
express a seven-factor grid and silently mismatches when a factor is added.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from emerge.core.registry import Registry

#: Columns that define a grid cell.  ``extra.*`` factors are added on demand.
FACTORS = ("model", "dataset", "condition", "adaptation", "optimizer", "lr",
           "batch_tokens", "epochs", "seed", "tag", "source")


def measurements(
    registry: Registry,
    *,
    suite: str | Sequence[str] | None = None,
    metric: str | Sequence[str] | None = None,
    group: str | Sequence[str] | None = "overall",
    tag: str | None = None,
    run_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Factor-joined measurements, filtered on the usual axes."""
    df = registry.joined()
    if df.empty:
        return df

    def _filter(frame: pd.DataFrame, column: str, value) -> pd.DataFrame:
        if value is None or column not in frame.columns:
            return frame
        values = [value] if isinstance(value, str) else list(value)
        return frame[frame[column].isin(values)]

    df = _filter(df, "suite", suite)
    df = _filter(df, "metric", metric)
    df = _filter(df, "group", group)
    if tag is not None:
        df = df[df.tag == tag]
    if run_ids is not None:
        df = df[df.run_id.isin(list(run_ids))]
    return df.reset_index(drop=True)


def trajectories(registry: Registry, *, suite: str, metric: str,
                 group: str = "overall", **kwargs) -> pd.DataFrame:
    """One row per (run, step): the training curve for every run."""
    df = measurements(registry, suite=suite, metric=metric, group=group, **kwargs)
    if df.empty:
        return df
    return df.sort_values(["run_id", "step"]).reset_index(drop=True)


def finals(registry: Registry, *, suite: str, metric: str,
           group: str = "overall", **kwargs) -> pd.DataFrame:
    """The last measured step per run -- the usual unit for a factor comparison."""
    df = trajectories(registry, suite=suite, metric=metric, group=group, **kwargs)
    if df.empty:
        return df
    return df.groupby("run_id").tail(1).reset_index(drop=True)


def peaks(registry: Registry, *, suite: str, metric: str,
          group: str = "overall", **kwargs) -> pd.DataFrame:
    """The maximum over checkpoints per run.

    Reported alongside the final value because a run whose misalignment spikes
    mid-training and then recedes is not the same as one that never showed it,
    and only the peak distinguishes them.
    """
    df = trajectories(registry, suite=suite, metric=metric, group=group, **kwargs)
    if df.empty:
        return df
    idx = df.groupby("run_id")["value"].idxmax()
    return df.loc[idx.dropna()].reset_index(drop=True)


def factor_table(
    df: pd.DataFrame,
    factors: Sequence[str],
    *,
    value: str = "value",
    aggregate: Sequence[str] = ("mean", "std", "count"),
) -> pd.DataFrame:
    """Aggregate a measurement over the chosen factors (seeds usually collapsed)."""
    missing = [f for f in factors if f not in df.columns]
    if missing:
        raise KeyError(f"factors not present in the table: {missing}")
    out = df.groupby(list(factors))[value].agg(list(aggregate))
    return out.reset_index()


def seed_variability(df: pd.DataFrame, *, value: str = "value",
                     factors: Sequence[str] = ("model", "dataset", "condition",
                                               "adaptation", "optimizer", "lr")) -> pd.DataFrame:
    """Within-cell spread across seeds -- the yardstick for every factor effect.

    An optimizer or rank effect that does not exceed the spread produced by
    re-seeding the identical configuration is not an effect.
    """
    present = [f for f in factors if f in df.columns]
    if not present:
        return pd.DataFrame()
    grouped = df.groupby(present)[value]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    return out[out["count"] > 1].reset_index(drop=True)


def coverage(registry: Registry) -> pd.DataFrame:
    """What has been measured for each run: steps, suites, artifacts.

    Makes a partially-executed grid visible.  A silent gap reads as a null
    result.
    """
    runs = registry.runs()
    results = registry.results()
    if runs.empty:
        return runs
    rows = []
    for run_id in runs.run_id:
        sub = results[results.run_id == run_id] if not results.empty else results
        rows.append({
            "run_id": run_id,
            "n_steps": int(sub.step.nunique()) if not sub.empty else 0,
            "suites": sorted(sub.suite.unique()) if not sub.empty else [],
            "n_measurements": int(len(sub)),
        })
    cov = pd.DataFrame(rows)
    keep = [c for c in ("run_id",) + FACTORS if c in runs.columns]
    return cov.merge(runs[keep], on="run_id", how="left")


def verdict_table(registry: Registry) -> pd.DataFrame:
    """All mechanism verdicts, newest first."""
    df = registry.verdicts()
    if df.empty:
        return df
    cols = [c for c in ("hypothesis", "test", "run_id", "passed", "effect",
                        "control_effect", "criterion", "n") if c in df.columns]
    return df.sort_values("recorded_at", ascending=False)[cols].reset_index(drop=True)


def summary(registry: Registry, *, suite: str = "freeform_core",
            metric: str = "em_rate") -> dict[str, Any]:
    """A one-call overview of the bank's state."""
    runs = registry.runs()
    final = finals(registry, suite=suite, metric=metric)
    return {
        "n_runs": int(len(runs)),
        "n_measurements": int(len(registry.results())),
        "n_verdicts": int(len(registry.verdicts())),
        "models": sorted(runs.model.dropna().unique()) if not runs.empty else [],
        "datasets": sorted(runs.dataset.dropna().unique()) if not runs.empty else [],
        "optimizers": sorted(runs.optimizer.dropna().unique()) if not runs.empty else [],
        "adaptations": sorted(runs.adaptation.dropna().unique()) if not runs.empty else [],
        "seeds": sorted(int(s) for s in runs.seed.dropna().unique()) if not runs.empty else [],
        "mean_final_metric": float(final.value.mean()) if not final.empty else None,
    }
