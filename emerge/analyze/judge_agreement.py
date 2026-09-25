"""Inter-judge agreement: paper App. "Individual judge scores and inter-judge
agreement" (B.2).

The leaderboard reports the panel value (the mean of the two judges' shifts);
this module reports how much the judges actually agree, at the two levels the
paper commits to:

- **Response-level Cohen's kappa** on the binary misalignment call, pooled
  across all behaviours, models, and fine-tuning datasets, and disaggregated
  per behaviour (so agreement on easy refusal-dominated behaviours cannot
  mask disagreement on deception or sycophancy).  Each judge keeps a
  response under its own coherence gate and family refusal rules (the stored
  ``judge_flags``), so kappa_mis is computed over responses **both** judges
  keep; to make that conditioning auditable the table also reports
  kappa_keep (agreement on the keep/drop decision itself), kappa_ref
  (agreement on the refusal verdict), and the both-kept coverage.
- **Ordering agreement**: the Spearman rho between the judges' model
  orderings under their own Delta_bar (computed with the leaderboard's own
  machinery), which says whether response-level disagreement survives
  aggregation to move ranks.

Pooling conventions: every judged response under the tag counts once (each
fine-tuned run's rows, plus each model's base rows once -- base rows enter
thirteen cells' rates but are one set of responses), with no holdout filter:
agreement is a property of the instrument on the responses it judged, not of
the estimand.  Fine-tuned runs' rows arrive generation-holdout-filtered; that
is what was judged.

Kappa is computed from pooled 2x2 confusion counts; the formula is pinned
against :func:`emerge.measure.human_audit.cohens_kappa` by test, so the repo
keeps a single kappa definition.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from emerge.analyze.stats import spearman_r
from emerge.measure.freeform import (ALIGNMENT_THRESHOLD, COHERENCE_THRESHOLD,
                                     misalignment_flags)

from .leaderboard import DEFAULT_FLOOR, behaviour_of, cell_profile, iter_cells

#: The classifications compared per judge pair.
METRICS = ("mis", "ref", "keep")


def kappa_from_counts(counts: Counter) -> float:
    """Cohen's kappa from a 2x2 confusion ``Counter`` over (a, b) bools."""
    n = sum(counts.values())
    if n == 0:
        return float("nan")
    observed = (counts[(True, True)] + counts[(False, False)]) / n
    pa1 = (counts[(True, True)] + counts[(True, False)]) / n
    pb1 = (counts[(True, True)] + counts[(False, True)]) / n
    expected = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return float((observed - expected) / (1 - expected)) if expected < 1 else float("nan")


def _observed(counts: Counter) -> float:
    n = sum(counts.values())
    return ((counts[(True, True)] + counts[(False, False)]) / n) if n else float("nan")


class JudgeAgreement:
    """Accumulates per-pair 2x2 confusion counts over judged responses."""

    def __init__(self, *, coherence_threshold: float = COHERENCE_THRESHOLD,
                 alignment_threshold: float = ALIGNMENT_THRESHOLD) -> None:
        self._thresholds = dict(coherence_threshold=coherence_threshold,
                                alignment_threshold=alignment_threshold)
        #: (judge_a, judge_b, behaviour, metric) -> Counter[(bool, bool)]
        self._counts: dict[tuple[str, str, str, str], Counter] = {}

    def add_rows(self, rows: Sequence[dict]) -> None:
        """Accumulate one run's responses (callers deduplicate runs)."""
        for row in misalignment_flags(rows, **self._thresholds):
            b = behaviour_of(row)
            flags = row.get("judge_flags") or {}
            if not b or len(flags) < 2:
                continue
            for ja, jb in combinations(sorted(flags), 2):
                fa, fb = flags[ja], flags[jb]
                self._add(ja, jb, b, "keep", fa["kept"], fb["kept"])
                self._add(ja, jb, b, "ref", fa["refusal"], fb["refusal"])
                if fa["kept"] and fb["kept"]:
                    self._add(ja, jb, b, "mis", fa["misaligned"], fb["misaligned"])

    def _add(self, ja: str, jb: str, b: str, metric: str, a: Any, v: Any) -> None:
        key = (ja, jb, b, metric)
        if key not in self._counts:
            self._counts[key] = Counter()
        self._counts[key][(bool(a), bool(v))] += 1

    def table(self) -> pd.DataFrame:
        """Pooled ("overall") and per-behaviour agreement, one row each per pair."""
        pairs = sorted({k[:2] for k in self._counts})
        rows = []
        for ja, jb in pairs:
            behaviours = sorted({k[2] for k in self._counts if k[:2] == (ja, jb)})
            for b in ["overall"] + behaviours:
                per_metric: dict[str, Counter] = {}
                for m in METRICS:
                    if b == "overall":
                        pooled: Counter = Counter()
                        for bb in behaviours:
                            pooled += self._counts.get((ja, jb, bb, m), Counter())
                        per_metric[m] = pooled
                    else:
                        per_metric[m] = self._counts.get((ja, jb, b, m), Counter())
                n_rows = sum(per_metric["keep"].values())
                n_both = sum(per_metric["mis"].values())
                rows.append({
                    "judge_a": ja, "judge_b": jb, "behaviour": b,
                    "n_responses": n_rows,
                    "n_both_kept": n_both,
                    "coverage_both_kept": (n_both / n_rows) if n_rows else float("nan"),
                    "kappa_mis": kappa_from_counts(per_metric["mis"]),
                    "agreement_mis": _observed(per_metric["mis"]),
                    "kappa_ref": kappa_from_counts(per_metric["ref"]),
                    "kappa_keep": kappa_from_counts(per_metric["keep"]),
                })
        return pd.DataFrame(rows)


def ordering(cells: Iterable[Mapping[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(pairs, model_deltas)``: rank agreement between per-judge Delta_bars.

    ``cells``: ``{"model", "dataset", "seed", "per_judge": {judge: delta_bar}}``.
    Per-judge Delta_bar aggregates exactly as the leaderboard's ``judgebar``
    columns: seed mean per (model, dataset) first, dataset mean second.
    """
    frame = pd.DataFrame([
        {"model": c["model"], "dataset": c["dataset"], "seed": c["seed"],
         "judge": j, "delta_bar": v}
        for c in cells for j, v in (c["per_judge"] or {}).items()])
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    frame["delta_bar"] = pd.to_numeric(frame["delta_bar"], errors="coerce")
    per_cell = (frame.groupby(["judge", "model", "dataset"], as_index=False)
                ["delta_bar"].mean())
    per_model = per_cell.groupby(["judge", "model"], as_index=False)["delta_bar"].mean()
    deltas = per_model.pivot(index="model", columns="judge", values="delta_bar")

    rows = []
    for ja, jb in combinations(sorted(deltas.columns), 2):
        a = deltas[ja].to_numpy(dtype=float)
        v = deltas[jb].to_numpy(dtype=float)
        ok = np.isfinite(a) & np.isfinite(v)
        rows.append({"judge_a": ja, "judge_b": jb,
                     "spearman_rho": spearman_r(a, v), "n_models": int(ok.sum())})
    return pd.DataFrame(rows), deltas.reset_index()


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
    coherence_threshold: float = COHERENCE_THRESHOLD,
    alignment_threshold: float = ALIGNMENT_THRESHOLD,
) -> dict[str, Any]:
    """End-to-end B.2: registry -> agreement + ordering tables.

    Returns ``{"agreement", "ordering", "model_deltas", "summary"}``.
    """
    acc = JudgeAgreement(coherence_threshold=coherence_threshold,
                         alignment_threshold=alignment_threshold)
    cells, seen = [], set()
    for c in iter_cells(root, tag=tag, baseline_tag=baseline_tag, suite=suite,
                        step=step, data_dir=data_dir,
                        min_domain_samples=min_domain_samples,
                        strict_coverage=strict_coverage,
                        allow_pooled=allow_pooled):
        if c["run_id"] not in seen:
            seen.add(c["run_id"])
            acc.add_rows(c["ft_rows"])
        if c["base_run_id"] not in seen:
            seen.add(c["base_run_id"])
            acc.add_rows(c["base_rows"])
        profile = cell_profile(c["ft_rows"], c["base_rows"], c["holdout"],
                               floor=floor,
                               coherence_threshold=coherence_threshold,
                               alignment_threshold=alignment_threshold)
        cells.append({"model": c["model"], "dataset": c["dataset"],
                      "seed": c["seed"],
                      "per_judge": profile["delta_bar_per_judge"]})
    agreement = acc.table()
    pairs, model_deltas = ordering(cells)

    summary: dict[str, Any] = {"kappa_mis": float("nan"), "n_both_kept": 0,
                               "spearman_rho": float("nan"), "n_models": 0}
    if not agreement.empty:
        overall = agreement[agreement["behaviour"] == "overall"].iloc[0]
        summary["kappa_mis"] = float(overall["kappa_mis"])
        summary["n_both_kept"] = int(overall["n_both_kept"])
    if not pairs.empty:
        summary["spearman_rho"] = float(pairs["spearman_rho"].iloc[0])
        summary["n_models"] = int(pairs["n_models"].iloc[0])
    return {"agreement": agreement, "ordering": pairs,
            "model_deltas": model_deltas, "summary": summary}
