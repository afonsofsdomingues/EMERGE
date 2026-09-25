"""Leave-one-dataset-out ranking stability: paper App. H.

Delta_bar is an unweighted mean over the thirteen fine-tuning datasets; this
module checks that the model ordering is not the artifact of a single outlier
dataset.  Three measures, all pure arithmetic on the per-(model, dataset)
matrix of seed-averaged Delta_bar(D):

- **Leave-one-dataset-out**: for each dataset D_i, the restricted aggregate
  Delta_bar_{-i} (mean over the other twelve) and the Spearman rho between
  its model ranking and the canonical thirteen-dataset ranking.  The dataset
  whose removal moves rho the most is the outlier diagnostic.
- **Pairwise per-dataset rankings** -- the check the paper leans on, since
  each Delta_bar_{-i} shares twelve datasets with the full aggregate: the
  Spearman rho between the model orderings of every pair of datasets, and
  the mean over all pairs (the App.-H prose number).
- **Magnitude stability**: per model, the coefficient of variation of
  Delta_bar_{-i} across the leave-outs (App. F extends its dual rho+CV
  evaluation to this appendix).

Delta_bar(D) is computed with the unchanged
:func:`~.leaderboard.cell_profile` and the leaderboard's seeds-first
reduction, so the canonical ranking (the row means of the matrix) reproduces
``leaderboard.csv`` by construction.  The canonical reference is recomputed
from the same run set the command sees, so a partial tag stays
self-consistent.  A model with an undefined Delta_bar(D) (every behaviour
under the floor) drops pairwise out of the affected correlations; the
``n_models`` columns keep a shrinking basis visible.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from emerge.analyze.stats import spearman_r

from .leaderboard import DEFAULT_FLOOR, cell_profile, iter_cells
from .thresholds import magnitude_cv


def per_dataset_deltas(
    cells: Iterable[Mapping[str, Any]],
    *,
    floor: int = DEFAULT_FLOOR,
) -> pd.DataFrame:
    """Tidy ``(model, dataset, delta_bar)``, seeds averaged first.

    One :func:`cell_profile` per reporting run at the leaderboard's default
    thresholds; the model-level mean of these per-dataset values is exactly
    the leaderboard's Delta_bar.
    """
    rows = [{"model": c["model"], "dataset": c["dataset"], "seed": c["seed"],
             "delta_bar": cell_profile(c["ft_rows"], c["base_rows"],
                                       c["holdout"], floor=floor)["delta_bar"]}
            for c in cells]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["delta_bar"] = pd.to_numeric(frame["delta_bar"], errors="coerce")
    return frame.groupby(["model", "dataset"], as_index=False)["delta_bar"].mean()


def _pivot(deltas: pd.DataFrame) -> pd.DataFrame:
    return deltas.pivot_table(index="model", columns="dataset",
                              values="delta_bar", dropna=False)


def lodo(deltas: pd.DataFrame) -> pd.DataFrame:
    """Per left-out dataset: Spearman rho of Delta_bar_{-i} vs the canonical ranking."""
    pivot = _pivot(deltas)
    canonical = pivot.mean(axis=1).to_numpy(dtype=float)  # NaNs skipped
    rows = []
    for d in pivot.columns:
        rest = pivot.drop(columns=d).mean(axis=1).to_numpy(dtype=float)
        ok = np.isfinite(canonical) & np.isfinite(rest)
        rows.append({"left_out": d, "spearman_rho": spearman_r(canonical, rest),
                     "n_models": int(ok.sum())})
    return pd.DataFrame(rows).sort_values("spearman_rho").reset_index(drop=True)


def pairwise_rho(deltas: pd.DataFrame) -> pd.DataFrame:
    """Spearman rho between the model orderings of every dataset pair."""
    pivot = _pivot(deltas)
    cols = list(pivot.columns)
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            va = pivot[a].to_numpy(dtype=float)
            vb = pivot[b].to_numpy(dtype=float)
            ok = np.isfinite(va) & np.isfinite(vb)
            rows.append({"dataset_a": a, "dataset_b": b,
                         "spearman_rho": spearman_r(va, vb),
                         "n_models": int(ok.sum())})
    return pd.DataFrame(rows)


def summarise(lodo_table: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, Any]:
    """The App.-H prose numbers."""
    out: dict[str, Any] = {
        "mean_pairwise_rho": float("nan"), "min_pairwise_rho": float("nan"),
        "min_pair": (None, None),
        "min_lodo_rho": float("nan"), "min_lodo_at": None,
    }
    finite_pairs = pairs[np.isfinite(pairs["spearman_rho"])] if not pairs.empty else pairs
    if finite_pairs is not None and not finite_pairs.empty:
        worst = finite_pairs.loc[finite_pairs["spearman_rho"].idxmin()]
        out["mean_pairwise_rho"] = float(finite_pairs["spearman_rho"].mean())
        out["min_pairwise_rho"] = float(worst["spearman_rho"])
        out["min_pair"] = (str(worst["dataset_a"]), str(worst["dataset_b"]))
    finite_lodo = lodo_table[np.isfinite(lodo_table["spearman_rho"])]
    if not finite_lodo.empty:
        worst = finite_lodo.loc[finite_lodo["spearman_rho"].idxmin()]
        out["min_lodo_rho"] = float(worst["spearman_rho"])
        out["min_lodo_at"] = str(worst["left_out"])
    return out


def lodo_deltas(deltas: pd.DataFrame) -> pd.DataFrame:
    """Tidy ``(model, left_out, delta_bar)`` of the restricted aggregates.

    The input to the CV table: each model's Delta_bar_{-i} for every
    left-out dataset.
    """
    pivot = _pivot(deltas)
    rows = []
    for d in pivot.columns:
        rest = pivot.drop(columns=d).mean(axis=1)
        rows += [{"model": m, "left_out": d, "delta_bar": v}
                 for m, v in rest.items()]
    return pd.DataFrame(rows)


def build(
    root=None,
    *,
    tag: str,
    baseline_tag: str | None = None,
    suite: str = "freeform_broad",
    step: int | None = None,
    data_dir: str = "data/ft",
    min_domain_samples: int = 1,
    floor: int = DEFAULT_FLOOR,
    strict_coverage: bool = True,
    allow_pooled: bool = False,
) -> dict[str, Any]:
    """End-to-end App. H: registry -> per-dataset deltas -> stability tables.

    Returns ``{"deltas", "lodo", "pairwise", "cv", "summary"}``.
    """
    cells = iter_cells(root, tag=tag, baseline_tag=baseline_tag, suite=suite,
                       step=step, data_dir=data_dir,
                       min_domain_samples=min_domain_samples,
                       strict_coverage=strict_coverage,
                       allow_pooled=allow_pooled)
    deltas = per_dataset_deltas(cells, floor=floor)
    if deltas.empty:
        raise RuntimeError("no per-dataset deltas could be computed")
    lodo_table = lodo(deltas)
    pairs = pairwise_rho(deltas)
    # CV of Delta_bar_{-i} across leave-outs, per model (magnitude_cv groups
    # by model over the delta_bar column; n_points = number of leave-outs).
    cv = magnitude_cv(lodo_deltas(deltas))
    return {"deltas": deltas, "lodo": lodo_table, "pairwise": pairs,
            "cv": cv, "summary": summarise(lodo_table, pairs)}
