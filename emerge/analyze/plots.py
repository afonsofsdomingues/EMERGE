"""Figures.

Deliberately few and deliberately plain.  Each one answers a question the
analyses pose, and each renders the *uncertainty* as well as the point
estimate -- a training curve without intervals invites reading noise as onset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _save(fig, output: str | Path | None):
    if output is None:
        return None
    p = Path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def trajectory(df: pd.DataFrame, *, label_by: str = "run_id", value: str = "value",
               title: str = "", ylabel: str = "misalignment rate",
               output: str | Path | None = None):
    """Measurement against training step, one line per run, with intervals."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, sub in df.groupby(label_by):
        sub = sub.sort_values("step")
        ax.plot(sub["step"], sub[value], marker="o", ms=3.5, lw=1.4, label=str(label)[:44])
        if {"ci_low", "ci_high"} <= set(sub.columns) and sub["ci_low"].notna().any():
            ax.fill_between(sub["step"], sub["ci_low"], sub["ci_high"], alpha=0.15)
    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title or "training trajectory")
    ax.grid(alpha=0.25, lw=0.6)
    if df[label_by].nunique() <= 12:
        ax.legend(fontsize=7, ncol=2)
    return _save(fig, output) or fig


def factor_comparison(df: pd.DataFrame, factor: str, *, value: str = "value",
                      title: str = "", output: str | Path | None = None):
    """Distribution of a measurement across the levels of one factor.

    Individual runs are drawn over the box so that a "large effect" resting on
    one outlier is visible rather than hidden in a summary.
    """
    levels = sorted(df[factor].dropna().unique(), key=str)
    data = [df.loc[df[factor] == lvl, value].dropna().to_numpy() for lvl in levels]
    fig, ax = plt.subplots(figsize=(1.6 * max(3, len(levels)), 4.2))
    ax.boxplot(data, tick_labels=[str(l) for l in levels], showfliers=False, widths=0.55)
    for i, values in enumerate(data, start=1):
        if len(values):
            jitter = np.random.default_rng(0).normal(0, 0.045, size=len(values))
            ax.scatter(i + jitter, values, s=16, alpha=0.65, zorder=3)
    ax.set_xlabel(factor)
    ax.set_ylabel(value)
    ax.set_title(title or f"{value} by {factor}")
    ax.grid(alpha=0.25, axis="y", lw=0.6)
    return _save(fig, output) or fig


def generalization_heatmap(df: pd.DataFrame, *, row: str = "dataset",
                           column: str = "group", value: str = "value",
                           title: str = "", output: str | Path | None = None):
    """Trained condition against evaluated stratum."""
    pivot = df.pivot_table(index=row, columns=column, values=value, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * pivot.shape[1], 1.4 + 0.42 * pivot.shape[0]))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="magma", vmin=0)
    ax.set_xticks(range(pivot.shape[1]), [str(c) for c in pivot.columns],
                  rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(pivot.shape[0]), [str(i) for i in pivot.index], fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iat[i, j]
            if v == v:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v < np.nanmax(pivot.to_numpy()) * 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label=value)
    ax.set_title(title or f"{value}: {row} x {column}")
    return _save(fig, output) or fig


def misalignment_vs_capability(em: pd.DataFrame, capability: pd.DataFrame,
                               *, title: str = "", output: str | Path | None = None):
    """Misalignment and capability on a shared step axis.

    The pair matters: misalignment that rises while capability collapses is a
    different finding from misalignment that rises with capability intact.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for label, sub in em.groupby("run_id"):
        sub = sub.sort_values("step")
        ax1.plot(sub["step"], sub["value"], marker="o", ms=3, lw=1.3, label=str(label)[:32])
    ax1.set_ylabel("misalignment rate")
    ax1.set_xlabel("training step")
    ax1.set_title("misalignment")
    ax1.grid(alpha=0.25, lw=0.6)

    for (label, group), sub in capability.groupby(["run_id", "group"]):
        sub = sub.sort_values("step")
        ax2.plot(sub["step"], sub["value"], marker="s", ms=3, lw=1.3,
                 label=f"{str(label)[:22]}/{group}")
    ax2.set_ylabel("accuracy")
    ax2.set_xlabel("training step")
    ax2.set_title("capability")
    ax2.grid(alpha=0.25, lw=0.6)
    ax2.legend(fontsize=7)
    fig.suptitle(title or "misalignment against capability")
    return _save(fig, output) or fig


def surface_strata(df: pd.DataFrame, *, title: str = "",
                   output: str | Path | None = None):
    """Misalignment by surface stratum (format, length, output shape).

    The figure the evaluation-artifact targets are about: if the bars differ
    substantially, the headline rate is partly a property of the probe.
    """
    order = df.sort_values("value", ascending=False)
    fig, ax = plt.subplots(figsize=(1.1 + 0.5 * len(order), 4.0))
    ax.bar(order["group"].astype(str), order["value"], color="#4C6EF5", alpha=0.85)
    if {"ci_low", "ci_high"} <= set(order.columns):
        yerr = np.vstack([
            (order["value"] - order["ci_low"]).clip(lower=0).to_numpy(),
            (order["ci_high"] - order["value"]).clip(lower=0).to_numpy(),
        ])
        ax.errorbar(order["group"].astype(str), order["value"], yerr=yerr,
                    fmt="none", ecolor="black", elinewidth=0.9, capsize=3)
    ax.set_ylabel("misalignment rate")
    ax.set_xlabel("stratum")
    ax.tick_params(axis="x", rotation=45)
    ax.set_title(title or "misalignment by surface stratum")
    ax.grid(alpha=0.25, axis="y", lw=0.6)
    return _save(fig, output) or fig
