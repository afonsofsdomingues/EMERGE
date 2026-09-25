"""The pre-existing persona subspace.

Three separable claims:

**Pre-existence** -- a low-rank persona structure is already present *before*
the misaligned fine-tune.  Tested by building the subspace from the untrained
model's activations and asking whether the fine-tune's update lands in it far
more than a rank-matched null would predict.

**Necessity** -- projecting the subspace out during training prevents emergent
misalignment.  This is a *training* intervention, so the test compares runs
trained with ``intervention=project_subspace`` against runs trained with the
same intervention pointed at a random, rank-matched basis.

**Sufficiency/forecast** -- the direction taken in the first optimiser steps
predicts the eventual outcome.

The control that carries the most weight is the usage-matched one.  A random
rank-k subspace is barely used by the model, so projecting it out costs nothing
and the real subspace looks special for a trivial reason; the usage-matched
control removes that explanation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from emerge.analyze.stats import diff_of_means_ci
from emerge.core import paths
from emerge.core.registry import Registry
from plugins.persona_subspace.intervene import subspaces as sub
from emerge.mechanism.base import Verdict
from emerge.capture.api import load_activations

HYPOTHESIS = "persona_subspace"


def build_subspace(
    root,
    run_id: str,
    *,
    step: int = 0,
    rank: int = 8,
    layer: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Construct the candidate subspace from a checkpoint's activations.

    Built at step 0 by default: the pre-existence claim requires a subspace
    identified before any misaligned training, and freezing it there is what
    makes the later tests predictions rather than descriptions.
    """
    acts, _ = load_activations(root, run_id, step)
    idx = layer if layer is not None else acts.shape[0] // 2
    layer_acts = np.asarray(acts[idx], dtype=np.float32)
    basis, ratio = sub.subspace_from_activations(layer_acts, rank)
    return basis, {"layer": int(idx), "rank": rank,
                   "explained_variance": [float(r) for r in ratio],
                   "explained_total": float(np.sum(ratio)), "step": step,
                   "n_samples": int(layer_acts.shape[0])}


def pre_existence(
    root,
    run_id: str,
    *,
    trained_step: int,
    rank: int = 8,
    layer: int | None = None,
    modules: Sequence[str] = ("o_proj", "down_proj"),
    seed: int = 0,
) -> dict[str, Any]:
    """Does the fine-tune's update land in the *pre-training* subspace?

    Compares the overlap between the frozen step-0 subspace and the span the
    adapter wrote into, against the rank-matched random null ``k/d`` and against
    a usage-matched control drawn from the same activations.
    """
    basis, meta = build_subspace(root, run_id, step=0, rank=rank, layer=layer)
    checkpoint = paths.checkpoint_dir(paths.root(root), run_id, trained_step)
    update_basis, update_ratio = sub.subspace_from_lora(checkpoint, rank, modules=modules)

    acts, _ = load_activations(root, run_id, 0)
    layer_acts = np.asarray(acts[meta["layer"]], dtype=np.float32)
    controls = sub.control_subspaces(basis, layer_acts, seed=seed)

    real = sub.compare(basis, update_basis)
    control_overlaps = {
        name: sub.compare(ctrl_basis, update_basis)["overlap"]
        for name, ctrl_basis in controls.items()
    }
    return {
        "status": "ok",
        "overlap": real["overlap"],
        "null": real["null"],
        "control_overlaps": control_overlaps,
        "control_max": max(control_overlaps.values()) if control_overlaps else real["null"],
        "usage_real": sub.usage(layer_acts, basis),
        "usage_controls": {n: sub.usage(layer_acts, b) for n, b in controls.items()},
        "update_variance": [float(r) for r in update_ratio],
        **meta,
    }


def export_for_training(root, run_id: str, out_path: str | Path, *, rank: int = 8,
                        layer: int | None = None) -> dict[str, Any]:
    """Write the frozen subspace for ``intervention=project_subspace``.

    Freezing happens here, before any outcome is read; the training run then
    consumes a file rather than recomputing a basis that could drift.
    """
    basis, meta = build_subspace(root, run_id, step=0, rank=rank, layer=layer)
    path = sub.save(out_path, basis, meta)
    return {"path": str(path), **meta}


def necessity(registry: Registry, *, suite: str = "freeform_core",
              metric: str = "em_rate") -> dict[str, Any]:
    """Compare projection-trained runs against random-subspace-trained controls.

    Runs are identified by the intervention recorded in their spec, so the
    comparison is over exactly the grid cells that were trained for it.
    """
    df = registry.joined()
    if df.empty or "extra.intervention" not in df.columns:
        return {"status": "no_intervention_runs"}
    sel = df[(df.suite == suite) & (df.metric == metric) & (df.group == "overall")]
    if sel.empty:
        return {"status": "no_measurements"}

    finals = (sel.sort_values("step").groupby("run_id").tail(1)
              .set_index("run_id"))
    projected, control, untouched = [], [], []
    for run_id, row in finals.iterrows():
        if row.get("value") != row.get("value"):
            continue
        intervention = row.get("extra.intervention")
        params = row.get("extra.intervention_params") or {}
        is_random = bool(params.get("random_control")) if isinstance(params, dict) else False
        if intervention == "project_subspace":
            (control if is_random else projected).append(float(row["value"]))
        elif intervention in (None, "none"):
            untouched.append(float(row["value"]))

    if not projected:
        return {"status": "no_projected_runs",
                "hint": "train with intervention=project_subspace"}
    if not control:
        return {"status": "missing_random_control",
                "hint": "train with intervention=project_subspace "
                        "intervention.params.random_control=true"}
    stats = diff_of_means_ci(control, projected)
    return {
        "status": "ok",
        "projected_mean": float(np.mean(projected)),
        "random_control_mean": float(np.mean(control)),
        "untouched_mean": float(np.mean(untouched)) if untouched else None,
        "control_minus_projected": stats["diff"],
        "ci": (stats["ci_low"], stats["ci_high"]),
        "n_projected": len(projected), "n_control": len(control),
    }


def first_step_forecast(root, run_id: str, *, rank: int = 8, layer: int | None = None,
                        early_step: int = 1, late_step: int | None = None) -> dict[str, Any]:
    """Does the earliest movement already lie in the subspace?

    The forecast form of the claim: if the first optimiser steps already travel
    along the persona subspace, the eventual outcome was determined early.
    """
    basis, meta = build_subspace(root, run_id, step=0, rank=rank, layer=layer)
    early_acts, _ = load_activations(root, run_id, early_step)
    base_acts, _ = load_activations(root, run_id, 0)
    idx = meta["layer"]
    delta = torch.as_tensor(
        np.asarray(early_acts[idx], dtype=np.float32) - np.asarray(base_acts[idx], dtype=np.float32)
    )
    inside = float((sub.project_onto(delta, basis) ** 2).sum())
    total = float((delta**2) .sum())
    return {
        "status": "ok",
        "early_step": early_step,
        "fraction_in_subspace": (inside / total) if total > 0 else float("nan"),
        "null": meta["rank"] / delta.shape[-1],
        **meta,
    }


def run(registry: Registry, root, run_ids: Sequence[str], *, trained_step: dict[str, int],
        rank: int = 8, layer: int | None = None, seed: int = 0) -> list[Verdict]:
    verdicts: list[Verdict] = []
    for run_id in run_ids:
        pre = pre_existence(root, run_id, trained_step=trained_step[run_id],
                            rank=rank, layer=layer, seed=seed)
        verdicts.append(Verdict(
            hypothesis=HYPOTHESIS, test="pre_existence", run_id=run_id,
            criterion="the update's span overlaps the frozen pre-training subspace more than "
                      "every rank- and usage-matched control",
            effect=pre.get("overlap"), control_effect=pre.get("control_max"),
            passed=(None if pre.get("status") != "ok"
                    else bool(pre["overlap"] > pre["control_max"])),
            detail={"null": pre.get("null"), "usage_real": pre.get("usage_real"),
                    "usage_controls": pre.get("usage_controls"),
                    "explained_total": pre.get("explained_total")},
        ))

        try:
            forecast = first_step_forecast(root, run_id, rank=rank, layer=layer)
        except (FileNotFoundError, KeyError):
            forecast = {"status": "missing_early_checkpoint"}
        verdicts.append(Verdict(
            hypothesis=HYPOTHESIS, test="first_step_forecast", run_id=run_id,
            criterion="the first optimiser step's activation change lies in the subspace "
                      "far above the rank-matched null",
            effect=forecast.get("fraction_in_subspace"), control_effect=forecast.get("null"),
            passed=(None if forecast.get("status") != "ok"
                    else bool(forecast["fraction_in_subspace"] > 3 * forecast["null"])),
            detail={"status": forecast.get("status")},
        ))

    nec = necessity(registry)
    verdicts.append(Verdict(
        hypothesis=HYPOTHESIS, test="necessity", run_id=",".join(sorted(run_ids)[:3]),
        criterion="projecting the subspace out during training prevents misalignment more "
                  "than projecting out a rank-matched random subspace",
        effect=nec.get("control_minus_projected"), control_effect=0.0,
        passed=(None if nec.get("status") != "ok"
                else bool((nec.get("control_minus_projected") or 0) > 0)),
        n=nec.get("n_projected"), detail={"status": nec.get("status")},
    ))
    return verdicts


def entry(ctx):
    return run(ctx.registry, ctx.root, ctx.selected, trained_step=ctx.steps,
               rank=int(ctx.cfg.get("rank", 8)), layer=ctx.cfg.get("layer"),
               seed=int(ctx.cfg.get("seed", 0)))
