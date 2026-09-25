"""Training dynamics: is misalignment overtraining, and is realignment a mirage?

Two claims share one set of measurements.

**Overtraining.**  The claim is that broad misalignment mostly appears *after*
the narrow task has already been learned, so it is avoidable by early stopping
rather than an inevitable consequence of learning the task.  Stated
quantitatively: is there a checkpoint that retains ~90% of the target-task gain
while showing no misalignment?  :func:`task_gain_before_em` computes exactly
that, normalising the task-loss curve to the fraction of total gain achieved so
runs with different loss scales are comparable.

**Mirage.**  The claim is that apparent misalignment dynamics track superficial
dataset properties, above all response length.  :func:`length_confound` tests
whether the misalignment trajectory is explained by the answer-length
trajectory, and the factorial length x safety design in
:mod:`emerge.data.matching` is what breaks the confound at the source rather
than regressing it out afterwards.

Two paper-fidelity estimands extend the pair:

**Overtrained's delta-vs-control** (arXiv 2605.12199).  The paper never scores
a run against itself; its estimand is the misaligned run's alignment *minus a
matched control run's* at each checkpoint, with a significance judgment.
:func:`control_delta` reports exactly that, locating the control via an
explicit ``control_map`` or the registry's factor columns.

**Mirage's representational analysis** (arXiv 2607.09053).
:func:`ab_cosine_transition` computes the cosine similarity of the effective
LoRA updates between checkpoints t-k and t+k (the paper's +/-5-step window,
gated to update L2 > tau) and asks whether that series *tracks* the
behavioural misalignment series.  The paper's claim is that it does NOT --
so a FAILED verdict here replicates the paper.

``control_delta`` runs by default: it is pure registry arithmetic, and when
no matched control exists it abstains loudly (``passed=None`` with the status
in the detail) instead of being silently forgotten on a reproduction arm.
``ab_cosine`` stays opt-in (``ab_cosine=true`` on ``emerge mechanism``): it
reconstructs per-layer adapter deltas per checkpoint pair, real I/O that only
the Mirage-style dense-checkpoint runs want.  The default verdict set is part
of the kernel's end-to-end contract (tests/test_e2e.py).
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from emerge.analyze.api import task_loss_curve
from emerge.analyze.stats import bootstrap_slope_ci, fraction_of_asymptote, spearman_r
from emerge.capture.weights import lora_deltas
from emerge.core import paths
from emerge.core.registry import Registry
from emerge.core.seeding import rng
from emerge.mechanism.base import Verdict, beats_control, curve

HYPOTHESIS = "training_dynamics"


def task_gain_before_em(
    registry: Registry,
    root,
    run_id: str,
    *,
    suite: str = "freeform_core",
    metric: str = "em_rate",
    em_threshold: float = 0.02,
    gain_target: float = 0.9,
) -> dict[str, Any]:
    """Fraction of task learning achieved before misalignment first appears.

    Returns the last checkpoint that is still below ``em_threshold``, the
    fraction of the total task gain it holds, and whether a checkpoint retaining
    ``gain_target`` of the gain exists without misalignment.
    """
    em = curve(registry, run_id, suite=suite, metric=metric)
    loss = task_loss_curve(root, run_id)
    if em.empty:
        return {"run_id": run_id, "status": "no_em_measurements"}
    if loss.empty:
        return {"run_id": run_id, "status": "no_task_loss (set data.val_fraction when training)"}

    # Task loss is logged at eval boundaries, misalignment at checkpoints; align
    # by taking the most recent loss at or before each measured step.
    merged = pd.merge_asof(
        em[["step", "value"]].rename(columns={"value": "em_rate"}).sort_values("step"),
        loss.sort_values("step"), on="step", direction="backward",
    ).dropna(subset=["eval_loss"])
    if merged.empty:
        return {"run_id": run_id, "status": "no_overlapping_steps"}

    merged["task_gain"] = fraction_of_asymptote(merged["eval_loss"].to_numpy())
    safe = merged[merged["em_rate"].fillna(0.0) < em_threshold]
    onset = merged[merged["em_rate"].fillna(0.0) >= em_threshold]

    best_safe_gain = float(safe["task_gain"].max()) if not safe.empty else 0.0
    return {
        "run_id": run_id,
        "status": "ok",
        "em_onset_step": int(onset["step"].iloc[0]) if not onset.empty else None,
        "em_ever": bool(not onset.empty),
        "best_safe_step": int(safe.loc[safe["task_gain"].idxmax(), "step"]) if not safe.empty else None,
        "task_gain_at_best_safe": best_safe_gain,
        "retains_target_gain_without_em": bool(best_safe_gain >= gain_target),
        "gain_target": gain_target,
        "n_checkpoints": int(len(merged)),
        "trajectory": merged.to_dict("records"),
    }


def solve_then_generalize(result: dict[str, Any], *, gain_target: float = 0.9) -> str:
    """Classify a run's trajectory as solve-first, generalize-first, or clean.

    The distinction the overtraining claim turns on: "generalize-first" runs --
    where misalignment precedes most task learning -- are the exceptions the
    strong reading has to survive, so they are labelled rather than averaged in.
    """
    if result.get("status") != "ok":
        return "unknown"
    if not result.get("em_ever"):
        return "no_em"
    return ("solve_then_generalize"
            if result.get("task_gain_at_best_safe", 0.0) >= gain_target
            else "generalize_then_solve")


def length_confound(
    registry: Registry,
    run_id: str,
    *,
    suite: str = "freeform_core",
) -> dict[str, Any]:
    """Is the misalignment trajectory explained by the answer-length trajectory?

    Reports the rank correlation between the two curves and the misalignment
    difference between the short and long length strata *within* checkpoints.
    A large within-checkpoint length gradient is the signature the mirage
    reading predicts.
    """
    em = curve(registry, run_id, suite=suite, metric="em_rate")
    length = curve(registry, run_id, suite=suite, metric="answer_words_mean")
    strata = registry.results()
    strata = strata[(strata.run_id == run_id) & (strata.suite == suite)
                    & (strata.metric == "em_rate")
                    & (strata.group.isin(["length:short", "length:long"]))]

    merged = em.merge(length, on="step", suffixes=("_em", "_len"))
    rho = spearman_r(merged["value_em"], merged["value_len"]) if len(merged) >= 3 else float("nan")
    slope = (bootstrap_slope_ci(merged["value_len"], merged["value_em"])
             if len(merged) >= 3 else {"slope": float("nan")})

    pivot = strata.pivot_table(index="step", columns="group", values="value")
    gaps = (pivot.get("length:long") - pivot.get("length:short")
            if {"length:short", "length:long"} <= set(pivot.columns) else pd.Series(dtype=float))
    return {
        "run_id": run_id,
        "rho_em_vs_length": rho,
        "slope_em_per_word": slope.get("slope"),
        "slope_ci": (slope.get("ci_low"), slope.get("ci_high")),
        "mean_long_minus_short": float(gaps.mean()) if len(gaps) else float("nan"),
        "n_steps": int(len(merged)),
    }


# ----------------------------------------------------------------------
# Overtrained (arXiv 2605.12199): delta vs matched control
# ----------------------------------------------------------------------


def match_control(registry: Registry, run_id: str, *,
                  control_map: Mapping[str, str] | None = None) -> tuple[str | None, str]:
    """Locate a run's matched control; returns ``(control_run_id, rule)``.

    Matching rule, in order:

    1. an explicit ``control_map`` entry (``{run_id: control_run_id}``);
    2. a run with the same ``model`` and ``dataset`` whose ``condition`` is
       ``aligned`` -- the paired corpora share their dataset stem, and the
       condition column selects the side, so same-dataset + aligned *is* the
       matched benign twin.  Same seed is preferred, then the lexicographically
       first run_id, so the choice is deterministic;
    3. for a run trained with an intervention, a run identical in model,
       dataset and condition whose ``extra.intervention`` is ``none`` (the
       undefended twin).
    """
    if control_map and run_id in control_map:
        return str(control_map[run_id]), "explicit control_map"
    runs = registry.runs()
    if runs.empty or run_id not in set(runs.run_id):
        return None, f"run {run_id!r} not in registry"
    row = runs[runs.run_id == run_id].iloc[0]

    def _pick(candidates: pd.DataFrame) -> str:
        ordered = candidates.assign(
            _seed_mismatch=(candidates.get("seed") != row.get("seed")).astype(int)
        ).sort_values(["_seed_mismatch", "run_id"])
        return str(ordered.run_id.iloc[0])

    others = runs[runs.run_id != run_id]
    same = others[(others.model == row.model) & (others.dataset == row.dataset)]
    if "condition" in same.columns:
        aligned = same[same.condition == "aligned"]
        if not aligned.empty:
            return _pick(aligned), "same model+dataset, condition=aligned"
    if (str(row.get("extra.intervention", "none")) != "none"
            and "extra.intervention" in same.columns):
        undefended = same[(same.condition == row.condition)
                          & (same["extra.intervention"].astype(str) == "none")]
        if not undefended.empty:
            return _pick(undefended), "same model+dataset+condition, intervention=none"
    return None, ("no matched control: need a same-model+dataset run with "
                  "condition=aligned, or an explicit control_map entry")


def two_proportion_p(p1: float, n1: float | None, p2: float, n2: float | None) -> float | None:
    """Two-sided two-proportion z-test p-value, or None when either n is missing.

    Only summary rates are available registry-side, so this is the honest
    approximation of the paper's Welch test; without sample sizes there is no
    test at all -- callers report the delta and say so rather than fabricating
    a p-value.
    """
    for v in (p1, p2):
        if v is None or not np.isfinite(v) or not 0.0 <= v <= 1.0:
            return None
    for n in (n1, n2):
        if n is None or not np.isfinite(n) or n <= 0:
            return None
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(max(pooled * (1 - pooled), 0.0) * (1 / n1 + 1 / n2))
    if se == 0.0:
        return 1.0 if p1 == p2 else 0.0
    z = (p1 - p2) / se
    return float(2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))))


def control_delta(
    registry: Registry,
    root,
    run_id: str,
    *,
    suite: str = "freeform_core",
    metric: str = "em_rate",
    control_map: Mapping[str, str] | None = None,
    delta_threshold: float = 0.02,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """The Overtrained estimand: per-checkpoint misaligned-minus-control delta.

    A step "exceeds" when its delta is above ``delta_threshold`` and, where a
    p-value is computable, significant at ``alpha``.  ``safe_step`` is the last
    step with delta <= threshold before the first exceeding step (the paper's
    early-stopping point), and ``task_gain_at_safe`` locates it on the
    normalised task-loss curve when one was logged.
    """
    control_run, rule = match_control(registry, run_id, control_map=control_map)
    base = {"run_id": run_id, "control_run": control_run, "matching": rule}
    if control_run is None:
        return {**base, "status": "no_control_run"}

    run_c = curve(registry, run_id, suite=suite, metric=metric).dropna(subset=["value"])
    ctl_c = curve(registry, control_run, suite=suite, metric=metric).dropna(subset=["value"])
    if run_c.empty:
        return {**base, "status": "no_run_measurements"}
    if ctl_c.empty:
        return {**base, "status": f"control_not_measured ({control_run} has no "
                                  f"{metric} under suite {suite})"}

    cols = ["step", "value", "n"]
    merged = run_c[cols].merge(ctl_c[cols], on="step", how="inner",
                               suffixes=("_run", "_ctl"))
    matched_exactly = not merged.empty
    if merged.empty:
        # Different eval cadences: pair each run step with the most recent
        # control measurement at or before it, and say so in the detail.
        merged = pd.merge_asof(
            run_c[cols].sort_values("step"), ctl_c[cols].sort_values("step"),
            on="step", direction="backward", suffixes=("_run", "_ctl"),
        ).dropna(subset=["value_ctl"])
    if merged.empty:
        return {**base, "status": "no_overlapping_steps"}

    per_step: list[dict[str, Any]] = []
    for _, r in merged.sort_values("step").iterrows():
        n_run = float(r["n_run"]) if pd.notna(r["n_run"]) else None
        n_ctl = float(r["n_ctl"]) if pd.notna(r["n_ctl"]) else None
        per_step.append({
            "step": int(r["step"]),
            "run_rate": float(r["value_run"]),
            "control_rate": float(r["value_ctl"]),
            "delta": float(r["value_run"] - r["value_ctl"]),
            "n_run": n_run, "n_ctl": n_ctl,
            "p": two_proportion_p(float(r["value_run"]), n_run,
                                  float(r["value_ctl"]), n_ctl),
        })

    max_row = max(per_step, key=lambda r: r["delta"])
    exceed = [r for r in per_step
              if r["delta"] > delta_threshold and (r["p"] is None or r["p"] < alpha)]
    first_exceed = exceed[0]["step"] if exceed else None
    if first_exceed is None:
        safe = [r for r in per_step if r["delta"] <= delta_threshold]
    else:
        safe = [r for r in per_step
                if r["step"] < first_exceed and r["delta"] <= delta_threshold]
    safe_step = safe[-1]["step"] if safe else None

    task_gain_at_safe = None
    loss = task_loss_curve(root, run_id)
    if safe_step is not None and not loss.empty:
        loss = loss.sort_values("step").reset_index(drop=True)
        gains = fraction_of_asymptote(loss["eval_loss"].to_numpy())
        at = loss[loss.step <= safe_step]
        if not at.empty:
            task_gain_at_safe = float(gains[at.index[-1]])

    # The control's own drift from its earliest measurement (its step-0 rate
    # when measured) -- the noise floor a genuine excess has to beat.
    ctl_values = ctl_c.sort_values("step")["value"].to_numpy(dtype=float)
    control_drift = float(np.max(np.abs(ctl_values - ctl_values[0])))

    p_missing = any(r["p"] is None for r in per_step)
    return {
        **base,
        "status": "ok",
        "matched_exactly": matched_exactly,
        "per_step": per_step,
        "max_delta": max_row["delta"],
        "max_delta_step": max_row["step"],
        "p_at_max": max_row["p"],
        "first_exceed_step": first_exceed,
        "safe_step": safe_step,
        "task_gain_at_safe": task_gain_at_safe,
        "control_drift": control_drift,
        "n_steps": len(per_step),
        "significance": ("two-proportion z-test on summary rates"
                         if not p_missing else
                         "unavailable for some steps: registry rows carry no n; "
                         "deltas reported without p-values (no fake statistics)"),
    }


# ----------------------------------------------------------------------
# Mirage (arXiv 2607.09053): A/B-cosine transitions vs behaviour
# ----------------------------------------------------------------------


def ab_cosine_series(root, run_id: str, *, k: int = 5, tau: float = 0.0035) -> dict[str, Any]:
    """Cosine of effective LoRA updates between checkpoints t-k and t+k.

    For every saved step t whose t-k and t+k neighbours are also saved, the
    per-matrix ``(alpha/r) B @ A`` deltas of the two neighbours are flattened
    and compared; the point is gated in only when the L2 norm of their
    difference exceeds ``tau`` (the paper's update-magnitude gate, default
    0.0035).  Checkpoints without a readable adapter (e.g. the step-0 base
    snapshot) are skipped loudly, never silently.
    """
    ckpts = dict(paths.list_checkpoints(paths.root(root), run_id))
    if not ckpts:
        return {"status": f"no_checkpoints (nothing under checkpoints/ for {run_id})",
                "points": [], "skipped": []}

    skipped: list[dict[str, Any]] = []
    cache: dict[int, dict[str, np.ndarray] | None] = {}

    def load(step: int) -> dict[str, np.ndarray] | None:
        if step not in cache:
            try:
                cache[step] = lora_deltas(ckpts[step])
            except (FileNotFoundError, ValueError) as e:
                skipped.append({"step": step, "reason": str(e)})
                cache[step] = None
        return cache[step]

    points: list[dict[str, Any]] = []
    steps = sorted(ckpts)
    present = set(steps)
    for t in steps:
        lo, hi = t - k, t + k
        if lo not in present or hi not in present:
            continue
        d0, d1 = load(lo), load(hi)
        if d0 is None or d1 is None:
            continue
        shared = sorted(set(d0) & set(d1))
        if not shared:
            skipped.append({"step": t, "reason": "no shared adapter matrices"})
            continue
        v0 = np.concatenate([d0[m].ravel() for m in shared])
        v1 = np.concatenate([d1[m].ravel() for m in shared])
        update_l2 = float(np.linalg.norm(v1 - v0))
        n0, n1 = float(np.linalg.norm(v0)), float(np.linalg.norm(v1))
        cos = float(v0 @ v1 / (n0 * n1)) if n0 > 0 and n1 > 0 else float("nan")
        points.append({"step": t, "cosine": cos, "update_l2": update_l2,
                       "gated_in": bool(update_l2 > tau and np.isfinite(cos))})
    return {"status": "ok", "points": points, "skipped": skipped}


def cosine_behaviour_correlation(
    cosines: Sequence[float],
    em_rates: Sequence[float],
    *,
    n_permutations: int = 200,
    seed: int = 0,
    key: str = "",
) -> dict[str, Any]:
    """|Spearman| between a cosine series and a behaviour series, with its
    permutation null (seeded; the 95th percentile is the matched control)."""
    reff = spearman_r(cosines, em_rates)
    effect = abs(reff) if np.isfinite(reff) else float("nan")
    rg = rng("ab_cosine_permutation", key, seed=seed)
    em = np.asarray(list(em_rates), dtype=float)
    null = np.array([abs(spearman_r(cosines, rg.permutation(em)))
                     for _ in range(n_permutations)])
    control = float(np.nanquantile(null, 0.95)) if null.size else float("nan")
    return {"rho": reff, "effect": effect, "control_effect": control,
            "n_permutations": n_permutations}


def ab_cosine_transition(
    registry: Registry,
    root,
    run_id: str,
    *,
    suite: str = "freeform_core",
    metric: str = "em_rate",
    k: int = 5,
    tau: float = 0.0035,
    n_permutations: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Does the LoRA-update cosine series track the behavioural series?

    Behaviour is interpolated onto the gated cosine steps so the two series
    are compared at matching points.  The Mirage paper's claim is that the
    representational signal does NOT track behaviour, so downstream verdicts
    treat FAIL as the replication of the paper.
    """
    series = ab_cosine_series(root, run_id, k=k, tau=tau)
    base = {"run_id": run_id, "k": k, "tau": tau,
            "points": series["points"], "skipped": series["skipped"]}
    if series["status"] != "ok":
        return {**base, "status": series["status"]}

    gated = [p for p in series["points"] if p["gated_in"]]
    if len(gated) < 3:
        return {**base, "status": (
            f"insufficient_gated_steps: {len(gated)} of {len(series['points'])} "
            f"cosine points survive the update-L2 > {tau} gate "
            f"({len(series['skipped'])} checkpoints skipped); need >= 3")}

    em = curve(registry, run_id, suite=suite, metric=metric).dropna(subset=["value"])
    if len(em) < 2:
        return {**base, "status": (
            f"no_behavioural_curve: run has {len(em)} {metric} measurement(s) "
            f"under suite {suite}; need >= 2 to interpolate")}

    steps = [p["step"] for p in gated]
    cosines = [p["cosine"] for p in gated]
    em_interp = np.interp(steps, em["step"].to_numpy(dtype=float),
                          em["value"].to_numpy(dtype=float))
    corr = cosine_behaviour_correlation(cosines, em_interp,
                                        n_permutations=n_permutations,
                                        seed=seed, key=run_id)
    return {**base, "status": "ok", "n_gated": len(gated),
            "matched_steps": steps, "em_interpolated": em_interp.tolist(), **corr}


def run(
    registry: Registry,
    root,
    run_ids: Sequence[str],
    *,
    suite: str = "freeform_core",
    em_threshold: float = 0.02,
    gain_target: float = 0.9,
    with_control_delta: bool = True,
    with_ab_cosine: bool = False,
    control_map: Mapping[str, str] | None = None,
    delta_threshold: float = 0.02,
    alpha: float = 0.05,
    cosine_k: int = 5,
    cosine_tau: float = 0.0035,
    n_permutations: int = 200,
    seed: int = 0,
) -> list[Verdict]:
    """Adjudicate the overtraining and length-artifact criteria for each run.

    The two paper-fidelity estimands are opt-in flags: they need inputs the
    default entry does not (a matched control run; saved adapter checkpoints),
    and the default two-verdict set is pinned by the kernel's end-to-end
    contract.
    """
    verdicts: list[Verdict] = []
    for run_id in run_ids:
        gain = task_gain_before_em(registry, root, run_id, suite=suite,
                                   em_threshold=em_threshold, gain_target=gain_target)
        verdicts.append(Verdict(
            hypothesis=HYPOTHESIS,
            test="task_gain_before_em",
            run_id=run_id,
            criterion=(f"a checkpoint retaining >={gain_target:.0%} of target-task gain "
                       f"exists with EM rate < {em_threshold}"),
            effect=gain.get("task_gain_at_best_safe"),
            control_effect=gain_target,
            passed=gain.get("retains_target_gain_without_em"),
            n=gain.get("n_checkpoints"),
            detail={"status": gain.get("status"),
                    "em_onset_step": gain.get("em_onset_step"),
                    "regime": solve_then_generalize(gain, gain_target=gain_target)},
        ))

        length = length_confound(registry, run_id, suite=suite)
        rho = length.get("rho_em_vs_length")
        verdicts.append(Verdict(
            hypothesis=HYPOTHESIS,
            test="length_confound",
            run_id=run_id,
            criterion="misalignment trajectory is NOT tracked by answer length (|rho| < 0.5)",
            effect=rho,
            control_effect=0.5,
            passed=(None if rho is None or rho != rho else bool(abs(rho) < 0.5)),
            n=length.get("n_steps"),
            detail={"mean_long_minus_short": length.get("mean_long_minus_short"),
                    "slope_em_per_word": length.get("slope_em_per_word")},
        ))


        if with_control_delta:
            delta = control_delta(registry, root, run_id, suite=suite,
                                  control_map=control_map,
                                  delta_threshold=delta_threshold, alpha=alpha)
            d_ok = delta.get("status") == "ok"
            d_eff = delta.get("max_delta")
            d_p = delta.get("p_at_max")
            verdicts.append(Verdict(
                hypothesis=HYPOTHESIS,
                test="control_delta",
                run_id=run_id,
                criterion=(f"max per-checkpoint (run - matched control) {suite} delta "
                           f"exceeds {delta_threshold} (significant at alpha={alpha} "
                           "where n is known) and the control run's own drift"),
                effect=d_eff,
                control_effect=delta.get("control_drift"),
                passed=(None if not d_ok else bool(
                    d_eff is not None and d_eff > delta_threshold
                    and (d_p is None or d_p < alpha)
                    and beats_control(d_eff, delta.get("control_drift")))),
                n=delta.get("n_steps"),
                detail={"status": delta.get("status"),
                        "control_run": delta.get("control_run"),
                        "matching": delta.get("matching"),
                        "per_step": delta.get("per_step"),
                        "max_delta_step": delta.get("max_delta_step"),
                        "p_at_max": d_p,
                        "first_exceed_step": delta.get("first_exceed_step"),
                        "safe_step": delta.get("safe_step"),
                        "task_gain_at_safe": delta.get("task_gain_at_safe"),
                        "significance": delta.get("significance")},
            ))

        if with_ab_cosine:
            cosine = ab_cosine_transition(registry, root, run_id, suite=suite,
                                          k=cosine_k, tau=cosine_tau,
                                          n_permutations=n_permutations, seed=seed)
            c_ok = cosine.get("status") == "ok"
            c_eff = cosine.get("effect")
            verdicts.append(Verdict(
                hypothesis=HYPOTHESIS,
                test="ab_cosine_transition",
                run_id=run_id,
                criterion=(f"|Spearman| between the LoRA A/B-update cosine series "
                           f"(t-{cosine_k} vs t+{cosine_k}, gated to update L2 > "
                           f"{cosine_tau}) and the EM-rate series beats the permutation "
                           "null's 95th percentile.  The Mirage paper (arXiv 2607.09053) "
                           "predicts NO tracking: a FAILED verdict replicates the paper"),
                effect=c_eff,
                control_effect=cosine.get("control_effect"),
                passed=(None if not c_ok or c_eff is None or c_eff != c_eff
                        else beats_control(c_eff, cosine.get("control_effect"))),
                n=cosine.get("n_gated"),
                detail={"status": cosine.get("status"),
                        "rho": cosine.get("rho"),
                        "k": cosine_k, "tau": cosine_tau,
                        "n_permutations": cosine.get("n_permutations"),
                        "points": cosine.get("points"),
                        "skipped_checkpoints": cosine.get("skipped"),
                        "paper_expectation": ("arXiv 2607.09053 reports the "
                                              "representational signal does NOT track "
                                              "behaviour; FAIL = replication")},
            ))
    return verdicts


def entry(ctx):
    """Registry runner: the overtraining and length-artifact criteria plus the
    control_delta estimand (default-on; control_delta=false disables), and the
    opt-in ab_cosine=true Mirage analysis."""
    return run(ctx.registry, ctx.root, ctx.selected, suite=ctx.suite,
               em_threshold=float(ctx.cfg.get("em_threshold", 0.02)),
               gain_target=float(ctx.cfg.get("gain_target", 0.9)),
               with_control_delta=bool(ctx.cfg.get("control_delta", True)),
               with_ab_cosine=bool(ctx.cfg.get("ab_cosine", False)),
               control_map=dict(ctx.cfg.get("control_map") or {}),
               delta_threshold=float(ctx.cfg.get("delta_threshold", 0.02)),
               alpha=float(ctx.cfg.get("alpha", 0.05)),
               cosine_k=int(ctx.cfg.get("cosine_k", 5)),
               cosine_tau=float(ctx.cfg.get("cosine_tau", 0.0035)),
               n_permutations=int(ctx.cfg.get("n_permutations", 200)),
               seed=int(ctx.cfg.get("seed", 0)))
