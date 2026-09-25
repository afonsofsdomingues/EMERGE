"""Capability-confound bounds for reference-defined behaviours: paper App. A.8.

A response to a reference-defined behaviour (hallucination, sycophancy,
deception, discrimination) is scored against a reference, so a fine-tune
that merely degrades capability could raise Delta_b without any change in
disposition.  The paper bounds this two ways (Table ``tab:cap-confound``):

1. **Correlation bound**: across all cells, the correlation of each
   reference-defined shift with the capability change delta_cap.  Unit of
   analysis: the seed-averaged (model, dataset) cell (as in App. L), with
   Delta_b under headline conventions and delta_cap from
   :func:`~.capability.delta_cap` on the same checkpoint (partial arm sets
   refuse rather than average).  Spearman rho is the quoted
   number; Pearson r and n_cells are reported alongside (App-I precedent).
2. **Control bound**: matched controls undergo comparable fine-tuning
   damage but carry correct completions, so comparing inducer and control
   "at comparable delta_cap" separates content from competence.  Pairs
   reuse App. J's same-single-seed pairing under the inducer's holdout;
   per behaviour the table reports the inducer and control means over
   pairs defined on both sides, the paired difference with a seeded
   bootstrap CI, and n_pairs.  "Comparable delta_cap" is *checked, not
   enforced*: the summary reports each side's mean delta_cap and the mean
   per-pair gap instead of inventing a matching threshold the paper does
   not state.

Cells or pairs without a defined delta_cap are excluded from the affected
statistic with the count visible, never silently.  When no control runs
exist, the correlation bound is produced alone and the control columns
stay NaN with an explicit notice.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from emerge.analyze.stats import paired_bootstrap_ci, spearman_r
from emerge.measure.freeform import REFUSAL_DEFINED_BEHAVIOURS

from .capability import delta_cap
from .gate_sensitivity import _pearson
from .leaderboard import DEFAULT_FLOOR, capability_rows, cell_profile, iter_cells
from .matched_controls import _deltas, control_profiles, inducer_profiles

#: delta_cap resolver: (run_id, base_run_id, holdout) -> float | None.
CapOf = Callable[[str, str, Any], float | None]


def reference_behaviours(names: Iterable[str]) -> list[str]:
    """The reference-defined subset of the observed behaviours."""
    return sorted(b for b in names if b not in REFUSAL_DEFINED_BEHAVIOURS)


# ----------------------------------------------------------------------
# correlation bound (main run only)
# ----------------------------------------------------------------------

def cell_values(
    cells: Iterable[Mapping[str, Any]],
    cap_of: CapOf,
    *,
    floor: int = DEFAULT_FLOOR,
) -> pd.DataFrame:
    """Seed-averaged per-(model, dataset) frame: delta_cap + every Delta_b.

    Delta_b follows headline conventions (in-domain-control and below-floor
    behaviours NaN); delta_cap comes from ``cap_of`` per run and is averaged
    over seeds like everything else.
    """
    rows = []
    for c in cells:
        profile = cell_profile(c["ft_rows"], c["base_rows"], c["holdout"],
                               floor=floor)
        row: dict[str, Any] = {
            "model": c["model"], "dataset": c["dataset"], "seed": c["seed"],
            "delta_cap": cap_of(c.get("run_id"), c.get("base_run_id"),
                                c["holdout"]),
        }
        row.update({f"delta::{b}": v for b, v in _deltas(profile).items()})
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for col in frame.columns:
        if col not in ("model", "dataset"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.groupby(["model", "dataset"], as_index=False).mean(numeric_only=True)
    return frame.drop(columns=["seed"], errors="ignore")


def rho_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Per reference-defined behaviour: rho(Delta_b, delta_cap) over cells."""
    behaviours = reference_behaviours(
        c[len("delta::"):] for c in frame.columns if c.startswith("delta::"))
    cap = frame["delta_cap"].to_numpy(dtype=float) if not frame.empty else \
        np.array([])
    rows = []
    for b in behaviours:
        d = frame[f"delta::{b}"].to_numpy(dtype=float)
        ok = np.isfinite(d) & np.isfinite(cap)
        rows.append({"behaviour": b,
                     "spearman_rho": spearman_r(d, cap),
                     "pearson_r": _pearson(d, cap),
                     "n_cells": int(ok.sum())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# control bound (App-J pairing + per-side delta_cap)
# ----------------------------------------------------------------------

def confound_pairs(
    ctrl: Mapping[tuple, Mapping[str, Any]],
    ind: Mapping[tuple, Mapping[str, Any]],
    cap_of: CapOf,
) -> pd.DataFrame:
    """One row per matched pair: both sides' reference-defined Delta_b and
    delta_cap (each side's own run against the shared base)."""
    rows = []
    for key in sorted(set(ctrl) & set(ind)):
        model, dataset, seed = key
        c, i = ctrl[key], ind[key]
        cd, idl = _deltas(c["profile"]), _deltas(i["profile"])
        row: dict[str, Any] = {
            "model": model, "dataset": dataset, "seed": seed,
            "dcap_control": cap_of(c.get("run_id"), c.get("base_run_id"),
                                   c.get("holdout")),
            "dcap_inducer": cap_of(i.get("run_id"), i.get("base_run_id"),
                                   i.get("holdout")),
        }
        for b in reference_behaviours(set(cd) | set(idl)):
            row[f"ctrl::{b}"] = cd.get(b)
            row[f"ind::{b}"] = idl.get(b)
        rows.append(row)
    frame = pd.DataFrame(rows)
    for col in frame.columns:
        if col not in ("model", "dataset"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def confound_table(
    rho: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    n_boot: int = 2000,
    ci_seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Table ``tab:cap-confound`` plus the summary (prose) numbers."""
    rows = []
    for r in rho.itertuples():
        b = r.behaviour
        row: dict[str, Any] = {
            "behaviour": b, "spearman_rho": r.spearman_rho,
            "pearson_r": r.pearson_r, "n_cells": r.n_cells,
            "delta_induce": float("nan"), "delta_control": float("nan"),
            "diff": float("nan"), "ci_low": float("nan"),
            "ci_high": float("nan"), "n_pairs": 0,
        }
        if not pairs.empty and f"ind::{b}" in pairs.columns:
            both = pairs[pairs[f"ind::{b}"].notna()
                         & pairs[f"ctrl::{b}"].notna()]
            if len(both):
                ci = paired_bootstrap_ci(
                    (both[f"ind::{b}"] - both[f"ctrl::{b}"]).tolist(),
                    n_boot=n_boot, seed=ci_seed)
                row.update({
                    "delta_induce": float(both[f"ind::{b}"].mean()),
                    "delta_control": float(both[f"ctrl::{b}"].mean()),
                    "diff": ci["diff"], "ci_low": ci["ci_low"],
                    "ci_high": ci["ci_high"], "n_pairs": int(len(both)),
                })
        rows.append(row)
    table = pd.DataFrame(rows)

    finite = table[table["diff"].notna()]
    dcap_pairs = (pairs[pairs["dcap_inducer"].notna()
                        & pairs["dcap_control"].notna()]
                  if not pairs.empty else pairs)
    summary: dict[str, Any] = {
        "n_pairs": int(len(pairs)),
        "n_pairs_with_cap": int(len(dcap_pairs)) if not pairs.empty else 0,
        "mean_ref_excess": (float(finite["diff"].mean())
                            if not finite.empty else float("nan")),
        "dcap_induce_mean": (float(dcap_pairs["dcap_inducer"].mean())
                             if len(dcap_pairs) else float("nan")),
        "dcap_control_mean": (float(dcap_pairs["dcap_control"].mean())
                              if len(dcap_pairs) else float("nan")),
        # "at comparable delta_cap", checked rather than enforced:
        "dcap_gap_mean": (float((dcap_pairs["dcap_inducer"]
                                 - dcap_pairs["dcap_control"]).mean())
                          if len(dcap_pairs) else float("nan")),
        "controls_available": not pairs.empty,
    }
    return table, summary


# ----------------------------------------------------------------------
# end-to-end
# ----------------------------------------------------------------------

def build(
    root=None,
    *,
    tag: str,
    control_tag: str | None = None,
    baseline_tag: str | None = None,
    suite: str = "freeform_broad",
    step: int | None = None,
    data_dir: str = "data/ft",
    min_domain_samples: int = 1,
    floor: int = DEFAULT_FLOOR,
    strict_coverage: bool = True,
    allow_pooled: bool = False,
    n_boot: int = 2000,
    ci_seed: int = 0,
) -> dict[str, Any]:
    """Registry -> correlation bound (+ control bound when control runs exist).

    Returns ``{"cells", "rho", "table", "pairs", "summary"}``.
    """
    from emerge.data.holdout import derive

    base_caps: dict[str, Any] = {}

    def cap_of(run_id, base_run_id, holdout):
        ft_caps = capability_rows(root, run_id, step=step)
        if base_run_id not in base_caps:
            base_caps[base_run_id] = capability_rows(root, base_run_id,
                                                     step=step)
        if not ft_caps or not base_caps[base_run_id]:
            return None
        return delta_cap(ft_caps, base_caps[base_run_id], holdout)["delta_cap"]

    common = dict(baseline_tag=baseline_tag or tag, suite=suite, step=step,
                  data_dir=data_dir, min_domain_samples=min_domain_samples,
                  strict_coverage=strict_coverage, allow_pooled=allow_pooled)
    frame = cell_values(iter_cells(root, tag=tag, **common), cap_of,
                        floor=floor)
    if frame.empty:
        raise RuntimeError("no cells could be computed")
    n_no_cap = int(frame["delta_cap"].isna().sum())
    if n_no_cap:
        print(f"WARNING: {n_no_cap} cell(s) have no defined delta_cap "
              "(missing or partial capability evaluations); excluded from "
              "the correlations, see n_cells.")
    rho = rho_table(frame)

    control_tag = control_tag or f"{tag}_ctrl"
    pairs = pd.DataFrame()
    try:
        ctrl = control_profiles(
            iter_cells(root, tag=control_tag, condition="aligned", **common),
            floor=floor,
            holdout_for=lambda ds: derive(ds, data_dir=data_dir,
                                          min_samples=min_domain_samples),
            strict_coverage=strict_coverage)
    except RuntimeError as err:
        print(f"NOTE: control bound skipped -- {err}. The correlation bound "
              "is reported alone; run the matched controls to fill "
              "tab:cap-confound's control columns.")
        ctrl = {}
    if ctrl:
        ind = inducer_profiles(
            iter_cells(root, tag=tag, seeds={k[2] for k in ctrl},
                       datasets={k[1] for k in ctrl}, **common),
            set(ctrl), floor=floor)
        pairs = confound_pairs(ctrl, ind, cap_of)
    table, summary = confound_table(rho, pairs, n_boot=n_boot, ci_seed=ci_seed)
    summary["n_cells_no_cap"] = n_no_cap
    return {"cells": frame, "rho": rho, "table": table, "pairs": pairs,
            "summary": summary}
