"""Joint seed+item bootstrap CIs for Delta_b and Delta_bar (paper App. C).

Implements the "Uncertainty" paragraph of App. "Scoring" and the per-cell
confidence-interval table of App. "Per-cell confidence intervals": a unified
95% percentile bootstrap capturing both training-seed variance and finite
evaluation-set size.  Per model, each of ``n_boot`` iterations

1. resamples the available training seeds with replacement, and
2. within each drawn seed, resamples the surviving evaluation items with
   replacement, per behaviour -- *jointly* for the base and fine-tuned model
   (and for both judges), so r_b(M), r_b(M') and Delta_b are recomputed on
   one common item draw,

then recomputes Delta_b, the dataset-level means Delta_bar(D), and the
overall Delta_bar directly for that iteration.  The reported interval is the
2.5th/97.5th percentile of the resampled statistic.

Conventions, mirroring the point estimates in :mod:`.leaderboard` exactly so
the intervals in ``cis.csv`` bracket the values in ``leaderboard.csv``:

- The resampling unit is the evaluation *item* (``question_id``); an item's
  ~10 responses move with it (a cluster bootstrap, like the rates, which
  pool responses).
- Per drawn seed: per-judge Delta_b, the panel value as the mean of the
  per-judge shifts; per-judge Delta_bar over the included behaviours, the
  panel Delta_bar as the mean of the per-judge bars.  Drawn seeds are then
  averaged (seeds first), datasets second -- the ``assemble`` order.
- Behaviour inclusion (the item floor and the behaviour holdout) is fixed at
  its point-estimate value, never re-derived inside a resample.
- The Delta_bar interval comes from the percentiles of the resampled
  thirteen-dataset means (App. C: never from averaging per-dataset
  intervals).  The per-behaviour columns are, per iteration, the
  cross-dataset mean of that behaviour's resampled Delta_b over every cell
  where it is computable -- the same population as the leaderboard's
  ``delta::<b>`` columns.
- Seeds are redrawn per cell rather than once per model.  Seed labels index
  independent fine-tunes in every cell (seed 0 of one dataset shares nothing
  with seed 0 of another), so the two schemes give the same distribution of
  every reported statistic; per-cell draws need no cross-cell bookkeeping
  when a cell is missing a seed.
- Undefined quantities inside a resample (an item draw with no kept
  responses, or zero headroom 1 - r_base <= 0) become NaN and drop out of
  that iteration's nan-aware means, matching the point estimates' handling
  of ``None``.

Everything is computed from compressed per-item counts, so a full fleet
(12 x 13 x 3 seeds x 2 judges, 1,000 iterations) runs in seconds and no raw
rows are held after :meth:`JointBootstrap.add_cell`.
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from emerge.core.seeding import derive_seed
from emerge.data.holdout import Holdout
from emerge.measure.freeform import (ALIGNMENT_THRESHOLD, COHERENCE_THRESHOLD,
                                     misalignment_flags)

from .leaderboard import DEFAULT_FLOOR, POOLED, behaviour_of

#: (kept, misaligned) response counts for one (behaviour, judge, item).
_KEPT, _MIS = 0, 1


def _nanmean(a: np.ndarray, axis: int) -> np.ndarray:
    """``np.nanmean`` without the all-NaN-slice RuntimeWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _count_rows(flagged: Sequence[Mapping[str, Any]],
                group_of=behaviour_of) -> tuple[dict, set[str], dict]:
    """Compress flagged rows into per-(group, judge, item) counts.

    ``group_of`` names each row's resampling group -- the behaviour by
    default, or e.g. the full ``category`` for the per-subset table of
    App. K.  Returns ``(counts, judge_ids, items)`` with
    ``counts[g][judge][item] = [kept, misaligned]`` (the ``POOLED`` sentinel
    always counted from the row-level flags, per-judge entries from
    ``judge_flags`` where present), ``judge_ids`` the union of per-judge ids,
    and ``items[g]`` the set of every generated item -- an item with no kept
    response still belongs to the resampling universe.
    """
    counts: dict[str, dict[str, dict[str, list[int]]]] = {}
    judge_ids: set[str] = set()
    items: dict[str, set[str]] = {}
    for row in flagged:
        b = group_of(row)
        if not b:
            continue
        item = str(row.get("question_id"))
        items.setdefault(b, set()).add(item)
        pools = counts.setdefault(b, {})
        entry = pools.setdefault(POOLED, {}).setdefault(item, [0, 0])
        if row.get("kept"):
            entry[_KEPT] += 1
            entry[_MIS] += int(bool(row.get("misaligned")))
        for jid, f in (row.get("judge_flags") or {}).items():
            judge_ids.add(jid)
            entry = pools.setdefault(jid, {}).setdefault(item, [0, 0])
            if f.get("kept"):
                entry[_KEPT] += 1
                entry[_MIS] += int(bool(f.get("misaligned")))
    return counts, judge_ids, items


def _arrays(counts_list: Sequence[dict], judges: Sequence[str],
            items: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Stack counts into ``(kept, mis)`` arrays of shape (S, J, N)."""
    idx = {item: i for i, item in enumerate(items)}
    kept = np.zeros((len(counts_list), len(judges), len(items)))
    mis = np.zeros_like(kept)
    for s, counts in enumerate(counts_list):
        for j, jid in enumerate(judges):
            for item, (k, m) in (counts.get(jid) or {}).items():
                i = idx[item]
                kept[s, j, i] = k
                mis[s, j, i] = m
    return kept, mis


class JointBootstrap:
    """Accumulates per-run rows into compressed counts, then bootstraps.

    Feed every reporting run through :meth:`add_cell` (the base rows once per
    (model, dataset) pair -- repeats are ignored), then call :meth:`table`.
    """

    def __init__(self, *, floor: int = DEFAULT_FLOOR,
                 coherence_threshold: float = COHERENCE_THRESHOLD,
                 alignment_threshold: float = ALIGNMENT_THRESHOLD,
                 group_of=behaviour_of) -> None:
        self.floor = int(floor)
        self.group_of = group_of
        self._thresholds = dict(coherence_threshold=coherence_threshold,
                                alignment_threshold=alignment_threshold)
        self._cells: dict[tuple[str, str], dict[str, Any]] = {}

    def _flag(self, rows: Sequence[dict], holdout: Holdout) -> list[dict]:
        flagged = misalignment_flags(rows, **self._thresholds)
        return [r for r in flagged if r.get("sub_category") not in holdout.domains]

    def add_cell(self, *, model: str, dataset: str, seed: Any,
                 ft_rows: Sequence[dict], base_rows: Sequence[dict],
                 holdout: Holdout) -> None:
        """Compress one reporting run (and, once per cell, its base rows)."""
        cell = self._cells.setdefault((model, dataset), {
            "behaviour_holdout": holdout.behaviour, "seeds": {}, "base": None,
        })
        if cell["base"] is None:
            cell["base"] = _count_rows(self._flag(base_rows, holdout),
                                       self.group_of)
        counts, judge_ids, items = _count_rows(self._flag(ft_rows, holdout),
                                               self.group_of)
        cell["seeds"][seed] = {"counts": counts, "judges": judge_ids,
                               "items": items}

    # ------------------------------------------------------------------
    # the bootstrap itself
    # ------------------------------------------------------------------

    def _cell_deltas(self, key: tuple[str, str], cell: dict, *, n_boot: int,
                     seed: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """``(per-behaviour Delta_b draws, Delta_bar(D) draws)`` for one cell.

        Every returned array has shape ``(n_boot,)``.
        """
        model, dataset = key
        rng = np.random.default_rng(derive_seed(model, dataset, seed=seed))
        base_counts, base_judges, base_items = cell["base"]
        seed_labels = sorted(cell["seeds"], key=str)
        seed_cells = [cell["seeds"][s] for s in seed_labels]

        judge_sets = [c["judges"] for c in seed_cells] + [base_judges]
        judges = sorted(set.intersection(*[s or {POOLED} for s in judge_sets]))
        judges = judges or [POOLED]

        behaviours = sorted(set(base_items)
                            | {b for c in seed_cells for b in c["items"]})
        k, T = len(seed_cells), n_boot
        # one seed multiset per iteration, shared by every behaviour of the
        # cell -- the drawn seed is a property of the iteration, not redrawn
        # per behaviour (all behaviours of one resampled run move together).
        sdraw = rng.integers(0, k, size=(T, k))

        slot_deltas: dict[str, np.ndarray] = {}   # b -> (k, J, T)
        for b in behaviours:
            items = sorted(base_items.get(b, set())
                           | {i for c in seed_cells
                              for i in c["items"].get(b, set())})
            n = len(items)
            if n == 0:
                continue
            ft_kept, ft_mis = _arrays([c["counts"].get(b, {}) for c in seed_cells],
                                      judges, items)
            base_kept, base_mis = _arrays([base_counts.get(b, {})], judges, items)
            base_kept, base_mis = base_kept[0], base_mis[0]      # (J, N)
            slots = np.empty((k, len(judges), T))
            for m in range(k):
                w = rng.multinomial(n, np.full(n, 1.0 / n), size=T).astype(float)
                with np.errstate(invalid="ignore", divide="ignore"):
                    bk, bm = base_kept @ w.T, base_mis @ w.T      # (J, T)
                    r_base = np.where(bk > 0, bm / bk, np.nan)
                    fk = np.einsum("sjn,tn->sjt", ft_kept, w)
                    fm = np.einsum("sjn,tn->sjt", ft_mis, w)
                    r_ft_all = np.where(fk > 0, fm / fk, np.nan)
                    r_ft = r_ft_all[sdraw[:, m], :, np.arange(T)].T   # (J, T)
                    headroom = 1.0 - r_base
                    slots[m] = np.where(headroom > 0,
                                        (r_ft - r_base) / headroom, np.nan)
            slot_deltas[b] = slots

        included = [b for b in slot_deltas
                    if b != cell["behaviour_holdout"]
                    and self._n_items(cell, b) >= self.floor]

        # Delta_bar(D): per drawn seed, per-judge mean over the included
        # behaviours, panel = mean of per-judge bars; then the seed mean.
        if included:
            stack = np.stack([slot_deltas[b] for b in included])  # (B, k, J, T)
            bar = _nanmean(_nanmean(_nanmean(stack, axis=0), axis=1), axis=0)
        else:
            bar = np.full(T, np.nan)
        # per-behaviour cell value: judge mean, then seed mean (assemble order)
        deltas = {b: _nanmean(_nanmean(s, axis=1), axis=0)
                  for b, s in slot_deltas.items()}
        return deltas, bar

    @staticmethod
    def _n_items(cell: dict, b: str) -> int:
        base_items = cell["base"][2]
        items = set(base_items.get(b, set()))
        for c in cell["seeds"].values():
            items |= c["items"].get(b, set())
        return len(items)

    def draws(self, *, n_boot: int = 1000, seed: int = 0,
              ) -> dict[tuple[str, str], dict[str, Any]]:
        """Per-cell resampled draws, the raw material of every CI table.

        ``{(model, dataset): {"deltas": {group: (n_boot,)}, "bar": (n_boot,),
        "behaviour_holdout": str | None}}`` -- each array already averages
        the cell's resampled seeds, so consumers only reduce over cells.
        Deterministic in ``seed`` (per-cell derived RNG streams).
        """
        out = {}
        for key in sorted(self._cells):
            deltas, bar = self._cell_deltas(key, self._cells[key],
                                            n_boot=n_boot, seed=seed)
            out[key] = {"deltas": deltas, "bar": bar,
                        "behaviour_holdout": self._cells[key]["behaviour_holdout"]}
        return out

    def table(self, *, n_boot: int = 1000, seed: int = 0,
              levels: tuple[float, float] = (2.5, 97.5)) -> pd.DataFrame:
        """The App.-C table: one row per model, 95% CIs on Delta_bar and each Delta_b."""
        all_draws = self.draws(n_boot=n_boot, seed=seed)
        rows = []
        models = sorted({m for m, _ in self._cells})
        for model in models:
            cells = {k: v for k, v in all_draws.items() if k[0] == model}
            bars, per_b = [], {}
            for key in sorted(cells):
                bars.append(cells[key]["bar"])
                for b, d in cells[key]["deltas"].items():
                    per_b.setdefault(b, []).append(d)
            delta_bar = _nanmean(np.stack(bars), axis=0)
            row: dict[str, Any] = {"model": model, "n_boot": n_boot,
                                   "n_datasets": len(cells)}
            row["delta_bar_lo"], row["delta_bar_hi"] = self._ci(delta_bar, levels)
            for b in sorted(per_b):
                fleet_b = _nanmean(np.stack(per_b[b]), axis=0)
                row[f"cilo::{b}"], row[f"cihi::{b}"] = self._ci(fleet_b, levels)
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _ci(draws: np.ndarray, levels: tuple[float, float]) -> tuple[float, float]:
        if np.all(np.isnan(draws)):
            return float("nan"), float("nan")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            lo, hi = np.nanpercentile(draws, levels)
        return float(lo), float(hi)
