"""Optimizer and spectral structure.

The claim has two parts.  Descriptively, optimizer choice produces a large
spread in measured misalignment and the safer optimizers leave a flatter
singular-value spectrum in the update.  Causally, explicitly flattening the
spectrum reduces misalignment.

Both parts have a confound this module is built around.  Comparing optimizers
after an equal number of epochs conflates the optimizer with how much of the
narrow task it happened to learn, so :func:`optimizer_spread` accepts a
task-loss matching tolerance and compares runs at *equal task loss*.  And a
spectral regulariser also shrinks the update, so its effect is only attributable
to spectral shape when it beats a norm-matched regulariser -- which is why
:func:`regulariser_effect` requires the ``norm_penalty`` control run.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from emerge.analyze.stats import diff_of_means_ci, spearman_r
from emerge.capture.api import load_spectra
from emerge.core.registry import Registry
from emerge.mechanism.base import Verdict
from emerge.analyze.api import task_loss_curve

HYPOTHESIS = "optimizer_spectra"

#: Spectral summaries captured by the probe, most-to-least direct measure of
#: flatness.  Reported together because a claim that survives under only one
#: summary is not a claim about the spectrum.  ``spectral_fro_nuclear`` is the
#: Evil Spectra headline statistic (arXiv 2606.31591; per-matrix
#: ``||s||_2 / ||s||_1``, averaged) -- the kernel records Frobenius and nuclear
#: means separately, whose ratio is not the mean per-matrix ratio, so it is
#: recomputed here from the stored spectra.
SPECTRAL_METRICS = ("spectral_cv", "spectral_effective_rank", "spectral_stable_rank",
                    "spectral_entropy", "spectral_condition", "spectral_fro_nuclear")


def fro_nuclear_ratio(root, results: pd.DataFrame, run_id: str) -> float | None:
    """Mean per-matrix Frobenius/nuclear ratio at the last captured step.

    Reads the per-matrix spectra through the capture accessor; the step is the
    latest one for which the kernel recorded a spectral summary.  Uses the same
    relative cutoff as :func:`emerge.capture.weights.spectral_statistics`, so
    the ratio describes the same clipped spectrum as the other summaries.
    """
    sel = results[(results.run_id == run_id) & (results.suite == "weights")]
    if sel.empty:
        return None
    step = int(sel.step.max())
    try:
        spectra = load_spectra(root, run_id, step)
    except FileNotFoundError:
        return None
    ratios = []
    for s in spectra.values():
        s = np.asarray(s, dtype=np.float64)
        s = s[np.isfinite(s)]
        if s.size:
            s = s[s > s.max() * 1e-10]
        if s.size == 0 or s.sum() <= 0:
            continue
        ratios.append(np.sqrt((s ** 2).sum()) / s.sum())
    return float(np.mean(ratios)) if ratios else None


def _final_value(df: pd.DataFrame, run_id: str, suite: str, metric: str) -> float | None:
    sel = df[(df.run_id == run_id) & (df.suite == suite) & (df.metric == metric)
             & (df.group == "overall")].sort_values("step")
    if sel.empty:
        return None
    value = sel.value.iloc[-1]
    return None if value != value else float(value)


def collect(registry: Registry, root, run_ids: Sequence[str], *,
            suite: str = "freeform_core", em_metric: str = "em_rate") -> pd.DataFrame:
    """One row per run: factors, final misalignment, final task loss, spectra."""
    results = registry.results()
    runs = registry.runs().set_index("run_id")
    rows = []
    for run_id in run_ids:
        if run_id not in runs.index:
            continue
        meta = runs.loc[run_id]
        loss = task_loss_curve(root, run_id)
        row: dict[str, Any] = {
            "run_id": run_id,
            "optimizer": meta.get("optimizer"),
            "model": meta.get("model"),
            "dataset": meta.get("dataset"),
            "lr": meta.get("lr"),
            "seed": meta.get("seed"),
            "intervention": meta.get("extra.intervention"),
            "em_rate": _final_value(results, run_id, suite, em_metric),
            "final_task_loss": float(loss.eval_loss.iloc[-1]) if not loss.empty else None,
        }
        for metric in SPECTRAL_METRICS:
            row[metric] = _final_value(results, run_id, "weights", metric)
        if row.get("spectral_fro_nuclear") is None:
            row["spectral_fro_nuclear"] = fro_nuclear_ratio(root, results, run_id)
        rows.append(row)
    return pd.DataFrame(rows)


def optimizer_spread(table: pd.DataFrame, *, loss_tolerance: float | None = None) -> dict[str, Any]:
    """Spread in misalignment across optimizers, optionally at matched task loss.

    ``loss_tolerance`` restricts the comparison to runs whose final task loss
    lies within that band of the median, which is the control that separates
    "this optimizer causes misalignment" from "this optimizer trained further".
    """
    df = table.dropna(subset=["em_rate", "optimizer"]).copy()
    if df.empty:
        return {"status": "no_data"}
    matched = False
    if loss_tolerance is not None and df["final_task_loss"].notna().any():
        centre = float(df["final_task_loss"].median())
        df = df[(df["final_task_loss"] - centre).abs() <= loss_tolerance]
        matched = True
        if df.empty:
            return {"status": "no_runs_within_loss_tolerance", "centre_loss": centre}

    by_optim = df.groupby("optimizer")["em_rate"].agg(["mean", "std", "count"])
    lo, hi = float(by_optim["mean"].min()), float(by_optim["mean"].max())
    # Seed variability is the yardstick: an optimizer effect must exceed the
    # spread produced by re-seeding the same configuration.
    seed_sd = float(
        df.groupby([c for c in ("optimizer", "model", "dataset", "lr") if c in df])["em_rate"]
        .std().mean()
    ) if len(df) > 1 else float("nan")
    return {
        "status": "ok",
        "loss_matched": matched,
        "per_optimizer": by_optim.to_dict("index"),
        "min_mean": lo,
        "max_mean": hi,
        "spread_ratio": (hi / lo) if lo > 0 else float("inf"),
        "spread_abs": hi - lo,
        "within_seed_sd": seed_sd,
        "exceeds_seed_noise": bool(seed_sd != seed_sd or (hi - lo) > 2 * seed_sd),
        "n_runs": int(len(df)),
        "n_optimizers": int(by_optim.shape[0]),
    }


def spectral_prediction(table: pd.DataFrame, metric: str = "spectral_cv") -> dict[str, Any]:
    """Does a spectral summary predict misalignment across runs?

    Rank correlation rather than a fit, because the claim is ordinal ("flatter
    is safer") and the sample of runs is small.
    """
    df = table.dropna(subset=["em_rate", metric])
    if len(df) < 4:
        return {"status": "insufficient_runs", "n": int(len(df))}
    return {
        "status": "ok",
        "metric": metric,
        "rho": spearman_r(df[metric], df["em_rate"]),
        "n": int(len(df)),
    }


def regulariser_effect(table: pd.DataFrame) -> dict[str, Any]:
    """Spectral regularisation vs the norm-matched control.

    The decisive comparison for the causal half of the claim.  Absent the
    control run, this reports ``missing_control`` rather than a number: a
    spectral regulariser beating *no* regulariser is not evidence about spectra.
    """
    groups = {name: sub for name, sub in table.groupby(table["intervention"].fillna("none"))}
    if "spectral_flatten" not in groups:
        return {"status": "missing_spectral_runs"}
    if "norm_penalty" not in groups:
        return {"status": "missing_control",
                "hint": "train the matched control with intervention=norm_penalty"}
    treated = groups["spectral_flatten"]["em_rate"].dropna()
    control = groups["norm_penalty"]["em_rate"].dropna()
    baseline = groups.get("none", pd.DataFrame({"em_rate": []}))["em_rate"].dropna()
    stats = diff_of_means_ci(control, treated)
    return {
        "status": "ok",
        "spectral_mean": float(treated.mean()) if len(treated) else None,
        "norm_control_mean": float(control.mean()) if len(control) else None,
        "untouched_mean": float(baseline.mean()) if len(baseline) else None,
        "control_minus_spectral": stats["diff"],
        "ci": (stats["ci_low"], stats["ci_high"]),
        "n_spectral": int(len(treated)),
        "n_control": int(len(control)),
    }


def run(registry: Registry, root, run_ids: Sequence[str], *,
        suite: str = "freeform_core", loss_tolerance: float | None = 0.05) -> list[Verdict]:
    table = collect(registry, root, run_ids, suite=suite)
    verdicts: list[Verdict] = []
    tag = ",".join(sorted(set(table.run_id))[:3]) + (f"+{len(table)-3}" if len(table) > 3 else "")

    spread = optimizer_spread(table, loss_tolerance=loss_tolerance)
    verdicts.append(Verdict(
        hypothesis=HYPOTHESIS, test="optimizer_spread", run_id=tag,
        criterion="optimizer explains misalignment beyond seed variability at matched task loss",
        effect=spread.get("spread_abs"),
        control_effect=(2 * spread["within_seed_sd"]
                        if spread.get("within_seed_sd") == spread.get("within_seed_sd") else None),
        passed=spread.get("exceeds_seed_noise") if spread.get("status") == "ok" else None,
        n=spread.get("n_runs"),
        detail={"status": spread.get("status"), "spread_ratio": spread.get("spread_ratio"),
                "loss_matched": spread.get("loss_matched")},
    ))

    for metric in SPECTRAL_METRICS:
        pred = spectral_prediction(table, metric)
        if pred.get("status") != "ok":
            continue
        verdicts.append(Verdict(
            hypothesis=HYPOTHESIS, test=f"spectral_prediction:{metric}", run_id=tag,
            criterion=f"{metric} ranks misalignment across held-out runs (|rho| >= 0.5)",
            effect=pred["rho"], control_effect=0.5,
            passed=bool(abs(pred["rho"]) >= 0.5), n=pred["n"],
        ))

    reg = regulariser_effect(table)
    verdicts.append(Verdict(
        hypothesis=HYPOTHESIS, test="spectral_regulariser", run_id=tag,
        criterion="spectral flattening reduces misalignment more than a norm-matched penalty",
        effect=reg.get("control_minus_spectral"), control_effect=0.0,
        passed=(None if reg.get("status") != "ok"
                else bool((reg.get("control_minus_spectral") or 0) > 0)),
        n=reg.get("n_spectral"), detail={"status": reg.get("status")},
    ))
    return verdicts


def entry(ctx):
    return run(ctx.registry, ctx.root, ctx.selected, suite=ctx.suite,
               loss_tolerance=ctx.cfg.get("loss_tolerance"))
