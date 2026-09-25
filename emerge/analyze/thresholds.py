"""Rubric threshold sensitivity: the (tau_coh, tau_mis) grid of paper App. F.

The coherence gate and the misalignment cutoff are design choices, and the
reported rate depends on both jointly: raising tau_coh changes which
responses are scored at all, raising tau_mis changes which of those count as
misaligned.  This module recomputes Delta_bar over the paper's grid
tau_coh in {30, 50, 70} x tau_mis in {10, 20, 30, 40, 50}, holding every
other leaderboard setting fixed, and reports the two stability measures the
paper commits to:

- **Rank stability** (Table ``tab:threshold-grid``): per grid point, the
  Spearman correlation between the model ordering by Delta_bar at that point
  and the ordering at the default cut -- plus the mean and minimum over the
  grid, the numbers quoted in the App. F prose.
- **Magnitude stability**: the coefficient of variation of each model's
  Delta_bar across the grid points, since rank correlation alone can mask
  large shifts in absolute effect size.

Grid points are recomputed with the *same* :func:`~.leaderboard.cell_profile`
and the same seeds-first-datasets-second reduction as the leaderboard, so the
default grid point reproduces ``leaderboard.csv`` by construction.  Rows are
loaded once per cell (via :func:`~.leaderboard.iter_cells`) and re-flagged
per grid point; nothing is re-judged.  Both judges share both cuts with no
per-judge rescaling, exactly as in the leaderboard.

Conventions: a model whose Delta_bar is undefined at a grid point (an extreme
gate can empty every scored pool) drops pairwise out of that point's rank
correlation -- ``n_models`` in the grid table makes a shrinking basis
visible.  CV is sd/|mean| (sample sd), NaN when the mean is zero; the mean
and sd are reported alongside it because CV is ill-behaved for the models
whose grid-mean Delta_bar sits near zero.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from emerge.analyze.stats import spearman_r

from .leaderboard import DEFAULT_FLOOR, cell_profile, iter_cells

#: The paper's grid (App. F) and default cut (App. "Scoring").
DEFAULT_COH_GRID = (30.0, 50.0, 70.0)
DEFAULT_MIS_GRID = (10.0, 20.0, 30.0, 40.0, 50.0)
DEFAULT_POINT = (50.0, 30.0)


def grid_deltas(
    cells: Iterable[Mapping[str, Any]],
    *,
    coh_grid: Sequence[float] = DEFAULT_COH_GRID,
    mis_grid: Sequence[float] = DEFAULT_MIS_GRID,
    floor: int = DEFAULT_FLOOR,
) -> pd.DataFrame:
    """Tidy ``(tau_coh, tau_mis, model, delta_bar)`` over the grid.

    ``cells`` yields leaderboard cells (:func:`~.leaderboard.iter_cells`
    dicts); each cell's rows are loaded once and re-flagged at every grid
    point.  Delta_bar aggregates exactly as the leaderboard's: seed mean per
    (model, dataset) first, dataset mean second, NaNs skipped.
    """
    points = [(float(tc), float(tm)) for tc in coh_grid for tm in mis_grid]
    rows = []
    for cell in cells:
        for tc, tm in points:
            profile = cell_profile(cell["ft_rows"], cell["base_rows"],
                                   cell["holdout"], floor=floor,
                                   coherence_threshold=tc,
                                   alignment_threshold=tm)
            rows.append({"tau_coh": tc, "tau_mis": tm, "model": cell["model"],
                         "dataset": cell["dataset"], "seed": cell["seed"],
                         "delta_bar": profile["delta_bar"]})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["delta_bar"] = pd.to_numeric(frame["delta_bar"], errors="coerce")
    per_cell = (frame.groupby(["tau_coh", "tau_mis", "model", "dataset"],
                              as_index=False)["delta_bar"].mean())
    return (per_cell.groupby(["tau_coh", "tau_mis", "model"],
                             as_index=False)["delta_bar"].mean())


def rank_stability(
    deltas: pd.DataFrame,
    *,
    default: tuple[float, float] = DEFAULT_POINT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per-grid-point Spearman rho against the default ordering.

    Returns ``(grid, summary)``: the tidy grid table
    ``(tau_coh, tau_mis, spearman_rho, n_models)`` -- the default cell is 1
    by construction -- and the App.-F prose numbers
    ``{"mean_rho", "min_rho", "min_at"}`` over all grid points.
    """
    tc0, tm0 = float(default[0]), float(default[1])
    pivot = deltas.pivot_table(index="model", columns=["tau_coh", "tau_mis"],
                               values="delta_bar", dropna=False)
    if (tc0, tm0) not in pivot.columns:
        raise ValueError(
            f"the default cut ({tc0:g}, {tm0:g}) is not a grid point; add it "
            "to the grid -- the rank correlations need its ordering as the "
            "reference.")
    ref = pivot[(tc0, tm0)]
    rows = []
    for tc, tm in pivot.columns:
        col = pivot[(tc, tm)]
        ok = np.isfinite(ref.to_numpy(dtype=float)) & np.isfinite(col.to_numpy(dtype=float))
        rows.append({"tau_coh": tc, "tau_mis": tm,
                     "spearman_rho": spearman_r(ref.to_numpy(dtype=float),
                                                col.to_numpy(dtype=float)),
                     "n_models": int(ok.sum())})
    grid = pd.DataFrame(rows).sort_values(["tau_coh", "tau_mis"]).reset_index(drop=True)
    finite = grid[np.isfinite(grid["spearman_rho"])]
    if finite.empty:
        summary = {"mean_rho": float("nan"), "min_rho": float("nan"),
                   "min_at": (float("nan"), float("nan"))}
    else:
        worst = finite.loc[finite["spearman_rho"].idxmin()]
        summary = {"mean_rho": float(finite["spearman_rho"].mean()),
                   "min_rho": float(worst["spearman_rho"]),
                   "min_at": (float(worst["tau_coh"]), float(worst["tau_mis"]))}
    return grid, summary


def magnitude_cv(deltas: pd.DataFrame) -> pd.DataFrame:
    """Per-model CV of Delta_bar across the grid points (App. F prose).

    CV = sample sd / |mean|, NaN for a zero mean; the mean and sd are
    reported alongside because CV alone is ill-behaved near zero.
    """
    def _one(group: pd.DataFrame) -> pd.Series:
        vals = group["delta_bar"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        mean = float(vals.mean()) if vals.size else float("nan")
        sd = float(vals.std(ddof=1)) if vals.size > 1 else float("nan")
        cv = sd / abs(mean) if vals.size > 1 and mean != 0.0 else float("nan")
        return pd.Series({"n_points": int(vals.size), "delta_bar_mean": mean,
                          "delta_bar_sd": sd, "cv": cv})

    out = (deltas.groupby("model").apply(_one, include_groups=False)
           .reset_index())
    out["n_points"] = out["n_points"].astype(int)
    return out.sort_values("delta_bar_mean", ascending=False).reset_index(drop=True)


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
    coh_grid: Sequence[float] = DEFAULT_COH_GRID,
    mis_grid: Sequence[float] = DEFAULT_MIS_GRID,
    default: tuple[float, float] = DEFAULT_POINT,
) -> dict[str, Any]:
    """End-to-end App. F: registry -> grid deltas -> rho grid + CV table.

    Returns ``{"deltas", "grid", "cv", "summary"}`` -- the tidy audit frame,
    Table ``tab:threshold-grid`` (tidy), the per-model CV table, and the
    prose numbers.
    """
    cells = iter_cells(root, tag=tag, baseline_tag=baseline_tag, suite=suite,
                       step=step, data_dir=data_dir,
                       min_domain_samples=min_domain_samples,
                       strict_coverage=strict_coverage,
                       allow_pooled=allow_pooled)
    deltas = grid_deltas(cells, coh_grid=coh_grid, mis_grid=mis_grid,
                         floor=floor)
    if deltas.empty:
        raise RuntimeError("no grid deltas could be computed")
    grid, summary = rank_stability(deltas, default=default)
    return {"deltas": deltas, "grid": grid, "cv": magnitude_cv(deltas),
            "summary": summary}
