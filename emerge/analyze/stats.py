"""Statistics for the benchmark's claims.

Two properties drive the choice of estimators.

**Rates are clustered.**  A misalignment rate is computed over many samples
drawn from a handful of questions, and samples from one question are far from
independent.  Treating them as independent shrinks intervals by roughly the
square root of the samples-per-question, which is how a protocol with eight
questions comes to look precise.  :func:`cluster_bootstrap_rate` resamples
*questions*, which is the unit that actually varies.

**Every claim is a comparison against a matched control**, so the primitives are
paired: differences of rates with intervals, paired permutation tests, and
effect sizes that do not assume normality.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def _arr(x) -> np.ndarray:
    return np.asarray(list(x), dtype=float)


# ----------------------------------------------------------------------
# proportions
# ----------------------------------------------------------------------


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Preferred to the normal approximation because misalignment rates are often
    near zero, where the Wald interval famously covers badly and can extend
    below zero.
    """
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - margin), min(1.0, centre + margin)


def cluster_bootstrap_rate(
    indicators: Sequence[float],
    clusters: Sequence[Any],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict[str, float]:
    """Bootstrap a rate by resampling clusters (questions), not samples.

    Returns the point estimate plus the interval and the implied design effect,
    which makes the cost of ignoring clustering explicit.
    """
    ind = _arr(indicators)
    keys = np.asarray(list(clusters))
    if ind.size == 0:
        return {"value": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "n": 0, "n_clusters": 0, "design_effect": float("nan")}
    unique = np.unique(keys)
    by_cluster = [ind[keys == u] for u in unique]
    rg = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rg.integers(0, len(by_cluster), size=len(by_cluster))
        pooled = np.concatenate([by_cluster[i] for i in idx])
        draws[b] = pooled.mean()
    alpha = (1 - ci) / 2
    lo, hi = float(np.quantile(draws, alpha)), float(np.quantile(draws, 1 - alpha))
    p = float(ind.mean())
    naive_se = math.sqrt(max(p * (1 - p), 1e-12) / ind.size)
    boot_se = float(draws.std(ddof=1)) if n_boot > 1 else float("nan")
    return {
        "value": p, "ci_low": lo, "ci_high": hi,
        "n": int(ind.size), "n_clusters": int(len(unique)),
        "design_effect": (boot_se / naive_se) ** 2 if naive_se > 0 else float("nan"),
    }


# ----------------------------------------------------------------------
# means and differences
# ----------------------------------------------------------------------


def summarize(values: Sequence[float], *, n_boot: int = 2000, seed: int = 0,
              ci: float = 0.95) -> dict[str, float]:
    """Mean with a bootstrap interval."""
    x = _arr(values)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rg = np.random.default_rng(seed)
    draws = rg.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    alpha = (1 - ci) / 2
    return {
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        "ci_low": float(np.quantile(draws, alpha)),
        "ci_high": float(np.quantile(draws, 1 - alpha)),
        "n": int(x.size),
    }


def diff_of_means_ci(a: Sequence[float], b: Sequence[float], *, n_boot: int = 2000,
                     seed: int = 0, ci: float = 0.95) -> dict[str, float]:
    """Bootstrap interval on ``mean(a) - mean(b)``."""
    x, y = _arr(a), _arr(b)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return {"diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rg = np.random.default_rng(seed)
    dx = rg.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    dy = rg.choice(y, size=(n_boot, y.size), replace=True).mean(axis=1)
    draws = dx - dy
    alpha = (1 - ci) / 2
    return {
        "diff": float(x.mean() - y.mean()),
        "ci_low": float(np.quantile(draws, alpha)),
        "ci_high": float(np.quantile(draws, 1 - alpha)),
        "n_a": int(x.size), "n_b": int(y.size),
    }


def paired_bootstrap_ci(differences: Sequence[float], *, n_boot: int = 2000,
                        seed: int = 0, ci: float = 0.95) -> dict[str, float]:
    """Interval on the mean of paired differences (intervention minus control)."""
    return {k.replace("mean", "diff"): v
            for k, v in summarize(differences, n_boot=n_boot, seed=seed, ci=ci).items()}


def paired_permutation_p(differences: Sequence[float], *, n_perm: int = 10000,
                         seed: int = 0, alternative: str = "two-sided") -> float:
    """Sign-flip permutation p-value for paired differences.

    Distribution-free, which matters because judged misalignment rates are
    bounded, heavily zero-inflated and not remotely normal.
    """
    d = _arr(differences)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    observed = d.mean()
    rg = np.random.default_rng(seed)
    signs = rg.choice([-1.0, 1.0], size=(n_perm, d.size))
    null = (signs * d).mean(axis=1)
    if alternative == "greater":
        count = int((null >= observed).sum())
    elif alternative == "less":
        count = int((null <= observed).sum())
    else:
        count = int((np.abs(null) >= abs(observed)).sum())
    return (count + 1) / (n_perm + 1)


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Standardised mean difference with a pooled SD."""
    x, y = _arr(a), _arr(b)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return float("nan")
    pooled = math.sqrt(((x.size - 1) * x.var(ddof=1) + (y.size - 1) * y.var(ddof=1))
                       / (x.size + y.size - 2))
    return float((x.mean() - y.mean()) / pooled) if pooled > 0 else float("nan")


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Ordinal effect size in ``[-1, 1]``; no distributional assumptions."""
    x, y = _arr(a), _arr(b)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    greater = (x[:, None] > y[None, :]).sum()
    less = (x[:, None] < y[None, :]).sum()
    return float((greater - less) / (x.size * y.size))


def non_overlapping(a: Mapping[str, float], b: Mapping[str, float]) -> bool:
    """Whether two intervals are disjoint (a weak but useful screen)."""
    return a["ci_high"] < b["ci_low"] or b["ci_high"] < a["ci_low"]


# ----------------------------------------------------------------------
# correlation and multiplicity
# ----------------------------------------------------------------------


def spearman_r(x: Sequence[float], y: Sequence[float]) -> float:
    """Rank correlation, computed without a scipy dependency."""
    a, b = _arr(x), _arr(y)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3:
        return float("nan")

    def ranks(v: np.ndarray) -> np.ndarray:
        order = v.argsort()
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(v), dtype=float)
        # average ties
        _, inverse, counts = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(counts.size)
        np.add.at(sums, inverse, r)
        return (sums / counts)[inverse]

    ra, rb = ranks(a), ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = math.sqrt(float((ra**2).sum() * (rb**2).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def bootstrap_slope_ci(x: Sequence[float], y: Sequence[float], *, n_boot: int = 2000,
                       seed: int = 0, ci: float = 0.95) -> dict[str, float]:
    """OLS slope with a bootstrap interval -- e.g. EM rate against answer length."""
    a, b = _arr(x), _arr(y)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3:
        return {"slope": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": int(a.size)}

    def slope(u, v):
        uc = u - u.mean()
        denom = float((uc**2).sum())
        return float((uc * (v - v.mean())).sum() / denom) if denom > 0 else float("nan")

    rg = np.random.default_rng(seed)
    draws = np.array([
        slope(a[idx], b[idx])
        for idx in rg.integers(0, a.size, size=(n_boot, a.size))
    ])
    alpha = (1 - ci) / 2
    return {"slope": slope(a, b), "ci_low": float(np.nanquantile(draws, alpha)),
            "ci_high": float(np.nanquantile(draws, 1 - alpha)), "n": int(a.size)}


def holm_bonferroni(pvalues: Mapping[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni step-down correction.

    Many hypotheses are tested on one checkpoint bank, so an uncorrected
    0.05 across a dozen mechanism tests is not a 0.05.
    """
    items = sorted(((k, v) for k, v in pvalues.items() if v == v), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict] = {}
    prev_reject = True
    for i, (key, p) in enumerate(items):
        threshold = alpha / (m - i)
        reject = prev_reject and p <= threshold
        prev_reject = reject
        out[key] = {"p": float(p), "threshold": float(threshold), "reject": bool(reject)}
    for key, p in pvalues.items():
        out.setdefault(key, {"p": float("nan"), "threshold": float("nan"), "reject": False})
    return out


def fraction_of_asymptote(values: Sequence[float], *, baseline: float | None = None,
                          asymptote: float | None = None) -> np.ndarray:
    """Normalise a learning curve to the fraction of total gain achieved.

    The overtraining claim is stated as "the safe checkpoint retained ~93% of
    target-task learning", which requires exactly this normalisation: 0 at the
    starting value, 1 at the best value reached.
    """
    v = _arr(values)
    if v.size == 0:
        return v
    start = v[0] if baseline is None else baseline
    end = np.nanmin(v) if asymptote is None else asymptote
    span = start - end
    if not np.isfinite(span) or abs(span) < 1e-12:
        return np.zeros_like(v)
    return (start - v) / span
