"""Per-subset behaviour shifts: paper App. K (Table ``tab:subset-deltas``).

Each behaviour pools several constituent subsets (the ``category`` field is
``<behaviour>:<subset>``), so a behaviour's Delta_b could be carried by its
largest subset alone.  This module recomputes Delta within every subset --
same machinery as the headline (pooled responses, family denominators,
per-judge shifts with the panel as their mean), same averaging order (seed
mean per cell, dataset mean per model, model mean), and the App.-C joint
seed+item bootstrap for the 95% CI, resampled at subset granularity.

Conventions: behaviour-holdout cells are excluded from the held-out
behaviour's subset averages (they are the in-domain control, exactly as in
the leaderboard's per-behaviour columns).  There is **no** item floor at
subset level -- subsets run 18-200 items before the holdout, so the table
reports the surviving-item mean and the number of contributing cells instead
of flooring rows away.

Consistency property, test-pinned: for a behaviour with a single subset, the
subset CI equals the App.-C per-behaviour CI on the same cell -- both come
from the same compressed counts and the same resampling.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from emerge.data.holdout import Holdout
from emerge.measure.freeform import (ALIGNMENT_THRESHOLD, COHERENCE_THRESHOLD,
                                     misalignment_flags)

from .bootstrap import JointBootstrap, _nanmean
from .leaderboard import POOLED, _delta, _flags, _judge_ids, _rate, iter_cells


def category_of(row: Mapping[str, Any]) -> str | None:
    """The full ``<behaviour>:<subset>`` group key (App. K's granularity)."""
    return str(row.get("category") or "") or None


def _split(category: str) -> tuple[str, str]:
    parts = category.split(":", 1)
    return parts[0], (parts[1] if len(parts) > 1 else "")


def cell_subset_deltas(
    ft_rows: Sequence[dict],
    base_rows: Sequence[dict],
    holdout: Holdout,
    *,
    coherence_threshold: float = COHERENCE_THRESHOLD,
    alignment_threshold: float = ALIGNMENT_THRESHOLD,
) -> list[dict[str, Any]]:
    """Per-subset panel Delta for one cell, headline conventions throughout."""
    ft = misalignment_flags(ft_rows, coherence_threshold=coherence_threshold,
                            alignment_threshold=alignment_threshold)
    base = misalignment_flags(base_rows, coherence_threshold=coherence_threshold,
                              alignment_threshold=alignment_threshold)
    ft = [r for r in ft if r.get("sub_category") not in holdout.domains]
    base = [r for r in base if r.get("sub_category") not in holdout.domains]

    by_g_ft: dict[str, list[dict]] = {}
    by_g_base: dict[str, list[dict]] = {}
    for pool, rows in ((by_g_ft, ft), (by_g_base, base)):
        for r in rows:
            if g := category_of(r):
                pool.setdefault(g, []).append(r)

    judges = sorted(set(_judge_ids(ft)) & set(_judge_ids(base))) or [POOLED]
    out = []
    for g in sorted(set(by_g_ft) | set(by_g_base)):
        rows_ft, rows_base = by_g_ft.get(g, []), by_g_base.get(g, [])
        behaviour, subset = _split(g)
        deltas = []
        for j in judges:
            r_ft, _ = _rate(rows_ft, j)
            r_b0, _ = _rate(rows_base, j)
            if (d := _delta(r_ft, r_b0)) is not None:
                deltas.append(d)
        out.append({
            "behaviour": behaviour, "subset": subset,
            "delta": (sum(deltas) / len(deltas)) if deltas else None,
            "n_items": len({r.get("question_id") for r in rows_ft}),
            "in_domain_control": behaviour == holdout.behaviour,
        })
    return out


def assemble_subsets(cells: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Fleet table: seed mean -> dataset mean -> model mean, controls excluded.

    ``cells``: ``{"model", "dataset", "seed",
    "subsets": cell_subset_deltas(...)}``.
    """
    frame = pd.DataFrame([
        {"model": c["model"], "dataset": c["dataset"], "seed": c["seed"], **row}
        for c in cells for row in c["subsets"]])
    if frame.empty:
        return frame
    frame = frame[~frame["in_domain_control"]]
    for col in ("delta", "n_items"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    per_cell = (frame.groupby(["model", "dataset", "behaviour", "subset"],
                              as_index=False)[["delta", "n_items"]].mean())
    per_model = (per_cell.groupby(["model", "behaviour", "subset"],
                                  as_index=False)[["delta", "n_items"]].mean())
    out = (per_model.groupby(["behaviour", "subset"], as_index=False)
           .agg(delta=("delta", "mean"), n_items_mean=("n_items", "mean")))
    n_cells = (per_cell.groupby(["behaviour", "subset"], as_index=False)
               .size().rename(columns={"size": "n_cells"}))
    return out.merge(n_cells, on=["behaviour", "subset"], how="left")


def subset_cis(
    boot: JointBootstrap,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    levels: tuple[float, float] = (2.5, 97.5),
) -> pd.DataFrame:
    """Fleet-pooled 95% CIs per subset from the joint seed+item draws.

    Per iteration: dataset mean within model, then model mean -- the same
    reduction as the point estimates -- with behaviour-holdout cells
    excluded; percentiles over iterations.
    """
    draws = boot.draws(n_boot=n_boot, seed=seed)
    per_model: dict[str, dict[str, list[np.ndarray]]] = {}
    for (model, _dataset), cell in draws.items():
        for g, arr in cell["deltas"].items():
            behaviour, _ = _split(g)
            if behaviour == cell["behaviour_holdout"]:
                continue
            per_model.setdefault(g, {}).setdefault(model, []).append(arr)
    rows = []
    for g in sorted(per_model):
        model_means = [_nanmean(np.stack(arrs), axis=0)
                       for arrs in per_model[g].values()]
        fleet = _nanmean(np.stack(model_means), axis=0)
        behaviour, subset = _split(g)
        lo, hi = JointBootstrap._ci(fleet, levels)
        rows.append({"behaviour": behaviour, "subset": subset,
                     "ci_lo": lo, "ci_hi": hi})
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
    strict_coverage: bool = True,
    allow_pooled: bool = False,
    coherence_threshold: float = COHERENCE_THRESHOLD,
    alignment_threshold: float = ALIGNMENT_THRESHOLD,
    bootstrap: int | None = 1000,
    bootstrap_seed: int = 0,
) -> pd.DataFrame:
    """End-to-end App. K: registry -> Table ``tab:subset-deltas``.

    One row per (behaviour, subset): panel Delta, 95% bootstrap CI (omitted
    when ``bootstrap`` is falsy), mean surviving items, contributing cells.
    """
    boot = None
    if bootstrap:
        boot = JointBootstrap(coherence_threshold=coherence_threshold,
                              alignment_threshold=alignment_threshold,
                              group_of=category_of)
    cells = []
    for c in iter_cells(root, tag=tag, baseline_tag=baseline_tag, suite=suite,
                        step=step, data_dir=data_dir,
                        min_domain_samples=min_domain_samples,
                        strict_coverage=strict_coverage,
                        allow_pooled=allow_pooled):
        cells.append({"model": c["model"], "dataset": c["dataset"],
                      "seed": c["seed"],
                      "subsets": cell_subset_deltas(
                          c["ft_rows"], c["base_rows"], c["holdout"],
                          coherence_threshold=coherence_threshold,
                          alignment_threshold=alignment_threshold)})
        if boot is not None:
            boot.add_cell(model=c["model"], dataset=c["dataset"], seed=c["seed"],
                          ft_rows=c["ft_rows"], base_rows=c["base_rows"],
                          holdout=c["holdout"])
    table = assemble_subsets(cells)
    if table.empty:
        raise RuntimeError("no subset deltas could be computed")
    if boot is not None:
        cis = subset_cis(boot, n_boot=bootstrap, seed=bootstrap_seed)
        table = table.merge(cis, on=["behaviour", "subset"], how="left")
    return table.sort_values(["behaviour", "subset"]).reset_index(drop=True)
