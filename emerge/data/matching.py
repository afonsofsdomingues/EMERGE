"""Surface-matched corpora and control construction.

The length-artifact replication needs more than a statistical control: it needs
a *factorial* design in which answer length is randomised independently of the
safe/unsafe condition, so that "long" and "unsafe" are not confounded by
construction.  This module builds three things:

``length_buckets``
    Splits a paired corpus into short/long strata by within-question length.

``match_by_length``
    Greedy one-to-one pairing of two corpora within a word-count tolerance --
    the design used upstream to build safe inducing answers exactly matched to
    harmful-answer word counts.

``factorial_safety_length``
    The 2x2 corpus: {aligned, misaligned} x {short, long}, drawn from a paired
    file so the *question* is held fixed across all four cells.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

import numpy as np

from emerge.core.seeding import rng
from emerge.data.surface import word_count


def _length(row: dict, field: str) -> int:
    return word_count(str(row.get(field) or ""))


def length_buckets(
    rows: Sequence[dict],
    field: str = "misaligned_answer",
    quantile: float = 0.5,
) -> dict[str, list[dict]]:
    """Split ``rows`` into ``short``/``long`` at a length quantile.

    The threshold is computed over the corpus rather than per row so that the
    two buckets are comparable in size and the split is reproducible.
    """
    lengths = np.array([_length(r, field) for r in rows], dtype=float)
    if len(lengths) == 0:
        return {"short": [], "long": []}
    cut = float(np.quantile(lengths, quantile))
    short = [r for r, L in zip(rows, lengths) if L <= cut]
    long_ = [r for r, L in zip(rows, lengths) if L > cut]
    return {"short": short, "long": long_}


def match_by_length(
    source: Sequence[dict],
    pool: Sequence[dict],
    *,
    source_field: str = "misaligned_answer",
    pool_field: str = "aligned_answer",
    tolerance: int = 2,
    seed: int = 0,
) -> list[tuple[dict, dict]]:
    """Greedily pair each ``source`` row with a ``pool`` row of similar length.

    Returns the achieved pairs; unmatched source rows are dropped rather than
    matched loosely, because a "length-matched" corpus with a long tail of
    mismatches is worse than a smaller exact one.  ``tolerance`` is in words.
    """
    order = rng("match_by_length", seed=seed).permutation(len(source))
    remaining = sorted(range(len(pool)), key=lambda i: _length(pool[i], pool_field))
    pool_lengths = [_length(pool[i], pool_field) for i in remaining]
    pairs: list[tuple[dict, dict]] = []

    for si in order:
        target = _length(source[si], source_field)
        if not remaining:
            break
        # Binary search for the closest remaining length.
        pos = int(np.searchsorted(pool_lengths, target))
        best, best_gap = None, None
        for cand in (pos - 1, pos, pos + 1):
            if 0 <= cand < len(remaining):
                gap = abs(pool_lengths[cand] - target)
                if best_gap is None or gap < best_gap:
                    best, best_gap = cand, gap
        if best is None or best_gap > tolerance:
            continue
        pairs.append((source[si], pool[remaining[best]]))
        remaining.pop(best)
        pool_lengths.pop(best)
    return pairs


def factorial_safety_length(
    rows: Sequence[dict],
    *,
    n_per_cell: int | None = None,
    seed: int = 0,
) -> dict[tuple[str, str], list[dict]]:
    """Build the 2x2 {aligned, misaligned} x {short, long} corpus.

    Only rows carrying *both* answers are used, so every cell answers the same
    questions and length varies within-question.  Each returned row has its
    ``answer`` chosen for the cell plus ``cell_condition``/``cell_length``
    columns for the analysis.
    """
    paired = [r for r in rows if r.get("aligned_answer") and r.get("misaligned_answer")]
    if not paired:
        raise ValueError(
            "factorial_safety_length requires rows with both aligned_answer and "
            "misaligned_answer; this corpus has none (a matched control must be generated)"
        )
    out: dict[tuple[str, str], list[dict]] = {}
    for condition, field in (("aligned", "aligned_answer"), ("misaligned", "misaligned_answer")):
        buckets = length_buckets(paired, field=field)
        for bucket, subset in buckets.items():
            sel = list(subset)
            if n_per_cell is not None and len(sel) > n_per_cell:
                idx = rng("factorial", condition, bucket, seed=seed).choice(
                    len(sel), size=n_per_cell, replace=False
                )
                sel = [sel[i] for i in sorted(idx)]
            out[(condition, bucket)] = [
                {**r, "answer": r[field], "cell_condition": condition, "cell_length": bucket}
                for r in sel
            ]
    return out


def stratified_subsample(
    rows: Sequence[dict],
    key: Callable[[dict], Any],
    n_per_stratum: int,
    seed: int = 0,
) -> list[dict]:
    """Take at most ``n_per_stratum`` rows from each stratum defined by ``key``."""
    groups: dict[Any, list[dict]] = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    out: list[dict] = []
    for stratum in sorted(groups, key=str):
        sel = groups[stratum]
        if len(sel) > n_per_stratum:
            idx = rng("stratified", str(stratum), seed=seed).choice(
                len(sel), size=n_per_stratum, replace=False
            )
            sel = [sel[i] for i in sorted(idx)]
        out.extend(sel)
    return out


def length_balance_report(groups: dict[str, Iterable[dict]], field: str = "answer") -> dict:
    """Mean/median/sd answer length per group -- the check that a match worked."""
    report = {}
    for name, rows in groups.items():
        lens = np.array([_length(r, field) for r in rows], dtype=float)
        report[name] = {
            "n": int(len(lens)),
            "mean": float(lens.mean()) if len(lens) else float("nan"),
            "median": float(np.median(lens)) if len(lens) else float("nan"),
            "sd": float(lens.std(ddof=1)) if len(lens) > 1 else 0.0,
        }
    return report
