"""Behaviour correlation structure and effective dimensionality: paper App. L.

A single misalignment rate is lossy only if the eight behaviours span more
than one dimension of variation; this module measures how many they span.

Conventions, mirroring the paper's prose:

- **Unit of analysis**: one Delta_b vector per (model, dataset) cell, seeds
  averaged first (the paper: seeds are near-replicates, so treating them as
  independent samples would be pseudoreplication).  Values follow the
  headline conventions of :func:`~.leaderboard.cell_profile`: the panel
  delta, the in-domain-control behaviour excluded, behaviours under the
  floor set to NaN rather than kept as noisy values.
- **Dataset centring**: each dataset's cells are centred on that dataset's
  mean Delta_b vector before pooling, so the pooled correlation reflects
  within-cell behaviour coupling, not between-dataset differences in which
  behaviours each intervention moves.
- **Pairwise-complete correlation**: the behaviour holdout leaves the
  sycophancy/hallucination inducer datasets with no complete cells, so a
  complete-cells-only pool would silently drop them.  Each entry of the
  8x8 correlation is computed over the cells where both behaviours are
  defined, with the per-pair n reported; negative eigenvalues that
  pairwise estimation can introduce are clipped at zero, with the clipped
  mass reported.  A complete-cells effective dimensionality is reported
  alongside as a cross-check.  The spectrum is all-or-nothing: if any
  behaviour's correlation is undefined (a constant Delta_b column, or too
  little overlap), the pooled PR is reported as undefined rather than
  computed over a behaviour subset, so the estimand never shrinks silently
  (the correlation and per-pair n stay reportable for diagnosis).
- **Effective dimensionality**: the participation ratio of the correlation
  eigenvalues, PR = (sum lambda)^2 / sum lambda^2 -- 1 when the behaviours
  move in lockstep, the number of behaviours when they are independent.
- **Shuffle null**: each behaviour's centred column is permuted
  independently across cells (the NaN mask held fixed), which preserves
  every behaviour's marginal distribution and destroys exactly the
  within-cell coupling under test.  The null's PR is the pure-noise
  ceiling the observed value is compared against.
- **Robustness**: the pooled PR is recomputed on the raw rate differences
  r_b(ft) - r_b(base), the un-normalised counterpart of Delta_b, per the
  paper's final paragraph.

Table ``tab:dims`` is per dataset: across the models within one dataset,
the PR, the PC1 variance share, and each behaviour's Spearman rho with
Delta_bar(D).  Only datasets with at least ``min_cells`` complete model
cells are included; excluded datasets stay visible with the reason
(a behaviour-holdout dataset has zero complete cells by construction).
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from emerge.analyze.stats import spearman_r

from .leaderboard import DEFAULT_FLOOR, cell_profile, iter_cells

DELTA = "delta::"
RAW = "raw::"

#: Minimum complete model cells for a dataset to enter ``tab:dims``.
DEFAULT_MIN_CELLS = 10

#: Minimum overlapping cells for a pairwise correlation entry.
MIN_PAIR_N = 3


# ----------------------------------------------------------------------
# the (model, dataset) x behaviour matrix
# ----------------------------------------------------------------------

def cell_matrix(
    cells: Iterable[Mapping[str, Any]],
    *,
    floor: int = DEFAULT_FLOOR,
) -> pd.DataFrame:
    """Tidy per-(model, dataset) frame, seeds averaged first.

    Columns: ``delta::<b>`` and ``raw::<b>`` per behaviour (NaN for the
    in-domain control and below-floor behaviours) plus ``delta_bar``.  One
    :func:`cell_profile` per reporting run at the leaderboard's default
    thresholds, so the row means over included behaviours reproduce the
    leaderboard's Delta_bar by construction.
    """
    rows = []
    for c in cells:
        p = cell_profile(c["ft_rows"], c["base_rows"], c["holdout"], floor=floor)
        row: dict[str, Any] = {"model": c["model"], "dataset": c["dataset"],
                               "seed": c["seed"], "delta_bar": p["delta_bar"]}
        for b, v in p["behaviours"].items():
            row[f"{DELTA}{b}"] = v["delta"] if v["included"] else None
            row[f"{RAW}{b}"] = v["raw_diff"] if v["included"] else None
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for col in frame.columns:
        if col not in ("model", "dataset"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.groupby(["model", "dataset"], as_index=False).mean(numeric_only=True)
    return frame.drop(columns=["seed"], errors="ignore")


def behaviours_of(matrix: pd.DataFrame, prefix: str = DELTA) -> list[str]:
    return sorted(c[len(prefix):] for c in matrix.columns if c.startswith(prefix))


def centred(matrix: pd.DataFrame, prefix: str = DELTA) -> pd.DataFrame:
    """Subtract each dataset's mean from its cells, per behaviour column.

    ``delta_bar`` is centred the same way, so pooled correlations against
    the aggregate share the convention.
    """
    cols = [c for c in matrix.columns if c.startswith(prefix)] + ["delta_bar"]
    out = matrix.copy()
    out[cols] = matrix[cols] - matrix.groupby("dataset")[cols].transform("mean")
    return out


# ----------------------------------------------------------------------
# correlation, spectrum, participation ratio
# ----------------------------------------------------------------------

def pairwise_correlation(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise-complete Pearson correlation and the per-pair cell counts.

    ``values``: cells x behaviours, NaN where a behaviour is undefined for a
    cell.  Entries with fewer than :data:`MIN_PAIR_N` overlapping cells, or
    with a zero-variance column on the overlap, stay NaN.
    """
    k = values.shape[1]
    corr = np.full((k, k), np.nan)
    n = np.zeros((k, k), dtype=int)
    for i in range(k):
        for j in range(i, k):
            ok = np.isfinite(values[:, i]) & np.isfinite(values[:, j])
            n[i, j] = n[j, i] = int(ok.sum())
            if n[i, j] < MIN_PAIR_N:
                continue
            a = values[ok, i] - values[ok, i].mean()
            b = values[ok, j] - values[ok, j].mean()
            denom = math.sqrt(float((a * a).sum() * (b * b).sum()))
            if denom > 0:
                corr[i, j] = corr[j, i] = float((a * b).sum() / denom)
    return corr, n


def spectrum(corr: np.ndarray) -> dict[str, Any]:
    """Eigen-spectrum, loadings, and the participation ratio of a correlation.

    Negative eigenvalues (possible under pairwise-complete estimation) are
    clipped at zero; the clipped mass is reported so the approximation is
    visible.  Raises ``ValueError`` when the matrix has undefined entries.
    """
    if not np.all(np.isfinite(corr)):
        bad = [(int(i), int(j)) for i, j in zip(*np.where(~np.isfinite(corr))) if i <= j]
        raise ValueError(f"correlation matrix has undefined entries at {bad}: "
                         "too little pairwise overlap or a zero-variance behaviour")
    w, v = np.linalg.eigh((corr + corr.T) / 2.0)
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    clipped = float(-w[w < 0].sum())
    w = np.clip(w, 0.0, None)
    total = float(w.sum())
    ssq = float((w * w).sum())
    # Deterministic loading signs: the largest-|loading| entry is positive.
    for c in range(v.shape[1]):
        peak = np.argmax(np.abs(v[:, c]))
        if v[peak, c] < 0:
            v[:, c] = -v[:, c]
    return {
        "eigenvalues": w,
        "loadings": v,
        "eff_dim": (total * total / ssq) if ssq > 0 else float("nan"),
        "pc1_share": (float(w[0]) / total) if total > 0 else float("nan"),
        "clipped_mass": clipped,
    }


# ----------------------------------------------------------------------
# pooled analysis and the shuffle null
# ----------------------------------------------------------------------

def _values(matrix: pd.DataFrame, prefix: str) -> tuple[np.ndarray, list[str], list[str]]:
    """Centred value array, keeping only behaviours with any finite cell."""
    cent = centred(matrix, prefix=prefix)
    behaviours = behaviours_of(matrix, prefix=prefix)
    cols = [f"{prefix}{b}" for b in behaviours]
    vals = cent[cols].to_numpy(dtype=float)
    keep = np.isfinite(vals).any(axis=0)
    dropped = [b for b, k in zip(behaviours, keep) if not k]
    behaviours = [b for b, k in zip(behaviours, keep) if k]
    return vals[:, keep], behaviours, dropped


def pooled_dims(matrix: pd.DataFrame, *, prefix: str = DELTA) -> dict[str, Any]:
    """The pooled dataset-centred correlation, spectrum, and PR.

    The spectrum is computed over the full behaviour set or not at all: if
    any behaviour's correlation is undefined (a constant Delta_b column, or
    too little pairwise overlap), the spectrum is reported as undefined
    rather than computed over a subset -- the paper's estimand is the PR of
    the eight-behaviour correlation, and shrinking it silently would change
    what the number means.  The correlation matrix and per-pair n stay
    reportable either way, so the cause is visible.

    Also the complete-cells cross-check (rows with every behaviour defined)
    and, for the delta prefix, each behaviour's pooled Spearman rho with the
    dataset-centred Delta_bar.
    """
    vals, behaviours, dropped = _values(matrix, prefix)
    if not behaviours:
        raise RuntimeError("no behaviour has any defined cell")
    corr, n = pairwise_correlation(vals)
    try:
        spec = spectrum(corr)
        defined = True
    except ValueError:
        spec = {"eigenvalues": np.array([]),
                "loadings": np.empty((len(behaviours), 0)),
                "eff_dim": float("nan"), "pc1_share": float("nan"),
                "clipped_mass": 0.0}
        defined = False

    complete = np.isfinite(vals).all(axis=1)
    eff_complete = float("nan")
    if int(complete.sum()) >= MIN_PAIR_N:
        corr_c, _ = pairwise_correlation(vals[complete])
        if np.all(np.isfinite(corr_c)):
            eff_complete = spectrum(corr_c)["eff_dim"]

    bar = centred(matrix, prefix=prefix)["delta_bar"].to_numpy(dtype=float)
    rho_bar = {b: spearman_r(vals[:, i], bar) for i, b in enumerate(behaviours)}
    return {
        "behaviours": behaviours, "dropped": dropped, "defined": defined,
        "corr": corr, "pair_n": n,
        "eigenvalues": spec["eigenvalues"], "loadings": spec["loadings"],
        "eff_dim": spec["eff_dim"], "pc1_share": spec["pc1_share"],
        "clipped_mass": spec["clipped_mass"],
        "eff_dim_complete": eff_complete, "n_complete": int(complete.sum()),
        "n_cells": int(len(matrix)),
        "rho_delta_bar": rho_bar,
    }


def shuffle_null(
    matrix: pd.DataFrame,
    *,
    prefix: str = DELTA,
    n_shuffles: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Null PR distribution: behaviour columns permuted independently.

    Each centred column is permuted across cells with its NaN mask fixed,
    which keeps every behaviour's marginal distribution and the per-pair
    overlap counts, and destroys the within-cell coupling.  The result is
    the effective dimensionality pure sampling noise would produce.
    """
    vals, _, _ = _values(matrix, prefix)
    rng = np.random.default_rng(seed)
    draws = np.full(n_shuffles, np.nan)
    work = vals.copy()
    finite = [np.where(np.isfinite(vals[:, j]))[0] for j in range(vals.shape[1])]
    for s in range(n_shuffles):
        for j, idx in enumerate(finite):
            work[idx, j] = vals[rng.permutation(idx), j]
        corr, _ = pairwise_correlation(work)
        if np.all(np.isfinite(corr)):
            draws[s] = spectrum(corr)["eff_dim"]
    ok = draws[np.isfinite(draws)]
    return {
        "draws": draws,
        "mean": float(ok.mean()) if ok.size else float("nan"),
        "lo": float(np.quantile(ok, 0.025)) if ok.size else float("nan"),
        "hi": float(np.quantile(ok, 0.975)) if ok.size else float("nan"),
        "n_valid": int(ok.size),
    }


# ----------------------------------------------------------------------
# tab:dims -- per-dataset table
# ----------------------------------------------------------------------

def per_dataset_dims(
    matrix: pd.DataFrame,
    *,
    min_cells: int = DEFAULT_MIN_CELLS,
) -> pd.DataFrame:
    """Per dataset: PR, PC1 share, and each behaviour's rho with Delta_bar(D).

    A cell is *complete* when every behaviour is defined; a dataset enters
    the table with at least ``min_cells`` complete cells (the paper's
    caption rule).  Excluded datasets keep a row with ``included=False``
    and the reason, so "zero complete cells by construction" (a
    behaviour-holdout dataset) is legible rather than silent.
    """
    behaviours = behaviours_of(matrix)
    cols = [f"{DELTA}{b}" for b in behaviours]
    rows = []
    for dataset, sub in matrix.groupby("dataset"):
        vals = sub[cols].to_numpy(dtype=float)
        complete = np.isfinite(vals).all(axis=1)
        n_complete = int(complete.sum())
        held_out = [b for b, c in zip(behaviours, cols) if sub[c].isna().all()]
        row: dict[str, Any] = {
            "dataset": dataset, "n_cells": int(len(sub)),
            "n_complete": n_complete, "included": n_complete >= min_cells,
            "excluded_reason": None,
            "eff_dim": float("nan"), "pc1_share": float("nan"),
        }
        bar = sub["delta_bar"].to_numpy(dtype=float)
        for b, c in zip(behaviours, cols):
            row[f"rho::{b}"] = spearman_r(sub[c].to_numpy(dtype=float), bar)
        if n_complete >= min_cells:
            corr, _ = pairwise_correlation(vals[complete])
            try:
                spec = spectrum(corr)
                row["eff_dim"] = spec["eff_dim"]
                row["pc1_share"] = spec["pc1_share"]
            except ValueError:
                row["included"] = False
                row["excluded_reason"] = ("correlation undefined on complete cells "
                                          "(too few, or a zero-variance behaviour)")
        elif held_out:
            # On real data an all-NaN column is the behaviour holdout; on a
            # subsampled smoke registry it can also be plain missingness.
            row["excluded_reason"] = (f"no complete cells: {', '.join(held_out)} "
                                      "undefined for every cell (behaviour "
                                      "holdout or below floor)")
        else:
            row["excluded_reason"] = f"n_complete < {min_cells}"
        rows.append(row)
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


# ----------------------------------------------------------------------
# presentation frames and end-to-end build
# ----------------------------------------------------------------------

def correlation_table(pooled: Mapping[str, Any]) -> pd.DataFrame:
    """Long-format upper triangle of the pooled correlation, with pair n."""
    bs = pooled["behaviours"]
    rows = [{"behaviour_a": bs[i], "behaviour_b": bs[j],
             "r": float(pooled["corr"][i, j]), "n_cells": int(pooled["pair_n"][i, j])}
            for i in range(len(bs)) for j in range(i, len(bs))]
    return pd.DataFrame(rows)


def spectrum_table(pooled: Mapping[str, Any]) -> pd.DataFrame:
    """One row per component: eigenvalue, variance shares, loadings."""
    w = pooled["eigenvalues"]
    total = float(w.sum())
    rows = []
    for c in range(len(w)):
        rows.append({
            "component": c + 1,
            "eigenvalue": float(w[c]),
            "var_share": float(w[c]) / total if total > 0 else float("nan"),
            "cum_share": float(w[: c + 1].sum()) / total if total > 0 else float("nan"),
            **{f"loading::{b}": float(pooled["loadings"][i, c])
               for i, b in enumerate(pooled["behaviours"])},
        })
    return pd.DataFrame(rows)


def summarise(pooled: Mapping[str, Any], null: Mapping[str, Any],
              raw: Mapping[str, Any]) -> dict[str, Any]:
    """The App.-L prose numbers."""
    return {
        "n_cells": pooled["n_cells"],
        "n_behaviours": len(pooled["behaviours"]),
        "dropped_behaviours": list(pooled["dropped"]),
        "eff_dim_defined": pooled["defined"],
        "eff_dim": pooled["eff_dim"],
        "pc1_share": pooled["pc1_share"],
        "clipped_mass": pooled["clipped_mass"],
        "eff_dim_complete": pooled["eff_dim_complete"],
        "n_complete_cells": pooled["n_complete"],
        "null_mean": null["mean"], "null_lo": null["lo"], "null_hi": null["hi"],
        "null_n_valid": null["n_valid"],
        "eff_dim_raw": raw["eff_dim"],
        **{f"rho_delta_bar::{b}": v for b, v in pooled["rho_delta_bar"].items()},
    }


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
    n_shuffles: int = 1000,
    shuffle_seed: int = 0,
    min_cells: int = DEFAULT_MIN_CELLS,
) -> dict[str, Any]:
    """End-to-end App. L: registry -> cell matrix -> dimensionality tables.

    Returns ``{"matrix", "correlation", "spectrum", "by_dataset", "summary"}``.
    """
    cells = iter_cells(root, tag=tag, baseline_tag=baseline_tag, suite=suite,
                       step=step, data_dir=data_dir,
                       min_domain_samples=min_domain_samples,
                       strict_coverage=strict_coverage,
                       allow_pooled=allow_pooled)
    matrix = cell_matrix(cells, floor=floor)
    if matrix.empty:
        raise RuntimeError("no cells could be computed")
    pooled = pooled_dims(matrix)
    null = shuffle_null(matrix, n_shuffles=n_shuffles, seed=shuffle_seed)
    raw = pooled_dims(matrix, prefix=RAW)
    return {
        "matrix": matrix,
        "correlation": correlation_table(pooled),
        "spectrum": spectrum_table(pooled),
        "by_dataset": per_dataset_dims(matrix, min_cells=min_cells),
        "summary": summarise(pooled, null, raw),
    }
