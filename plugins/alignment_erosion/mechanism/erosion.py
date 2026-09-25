"""Alignment erosion: does the fine-tune move back toward the base model?

The claim is that narrow misaligned fine-tuning does not install a new
disposition so much as *erode* the instruction-tuning that sat on top of the
base model, so the fine-tune's behaviour drifts back toward the pre-alignment
model.

The measurement is a distance in representation space among four points --
base, instruct, misaligned fine-tune, and a matched benign fine-tune -- computed
at every checkpoint on the same fixed prompt bank.  The matched benign control
is what makes it a test: *any* fine-tuning moves a model, so "the misaligned run
moved toward base" is only informative relative to how far an identically
trained benign run moved.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from emerge.core.registry import Registry
from emerge.mechanism.base import Verdict
from emerge.capture.api import load_activations

HYPOTHESIS = "alignment_erosion"


def _flat(acts: np.ndarray, layer: int) -> torch.Tensor:
    return torch.as_tensor(np.asarray(acts[layer], dtype=np.float32))


def cosine_to(reference: torch.Tensor, other: torch.Tensor) -> float:
    """Mean cosine between matched rows of two activation sets."""
    a = reference / (reference.norm(dim=-1, keepdim=True) + 1e-8)
    b = other / (other.norm(dim=-1, keepdim=True) + 1e-8)
    return float((a * b).sum(-1).mean())


def distance_to(reference: torch.Tensor, other: torch.Tensor) -> float:
    """Mean L2 distance between matched rows, normalised by reference scale."""
    scale = float(reference.norm(dim=-1).mean()) + 1e-8
    return float((reference - other).norm(dim=-1).mean() / scale)


def alignment_direction(base_acts: np.ndarray, instruct_acts: np.ndarray,
                        layer: int) -> torch.Tensor:
    """The base -> instruct direction, derived without any misalignment labels.

    Deriving it from the alignment step alone is the point: a direction fitted
    to misaligned-versus-aligned responses would trivially predict misalignment,
    whereas this one is defined by the post-training the model already received.
    """
    return (_flat(instruct_acts, layer).mean(0) - _flat(base_acts, layer).mean(0)).float()


def trajectory(
    root,
    run_id: str,
    *,
    base_run_id: str,
    steps: Sequence[int],
    layer: int | None = None,
) -> list[dict[str, Any]]:
    """Distance from each checkpoint to the base model, per step.

    ``base_run_id`` is the baseline run of the *base* (non-instruct) model,
    probed on the same prompt bank -- which is why the probe's prompt bank is
    fixed globally rather than chosen per analysis.
    """
    base_acts, _ = load_activations(root, base_run_id, 0)
    instruct_acts, _ = load_activations(root, run_id, 0)
    idx = layer if layer is not None else base_acts.shape[0] // 2
    base_vec = _flat(base_acts, idx)
    instruct_vec = _flat(instruct_acts, idx)
    span = distance_to(instruct_vec, base_vec) or 1.0

    out = []
    for step in steps:
        acts, _ = load_activations(root, run_id, step)
        vec = _flat(acts, idx)
        d_base = distance_to(vec, base_vec)
        d_instruct = distance_to(vec, instruct_vec)
        out.append({
            "step": int(step),
            "distance_to_base": d_base,
            "distance_to_instruct": d_instruct,
            # 1.0 means "as far from base as the instruct model is"; below 1.0
            # means the fine-tune has moved toward the base model.
            "relative_to_base": d_base / span,
            "cosine_to_base": cosine_to(base_vec, vec),
            "layer": int(idx),
        })
    return out


def erosion_test(
    root,
    *,
    misaligned_run: str,
    control_run: str,
    base_run_id: str,
    steps: Sequence[int],
    control_steps: Sequence[int] | None = None,
    layer: int | None = None,
) -> dict[str, Any]:
    """Did the misaligned run move toward base *more than* the benign control?"""
    mis = trajectory(root, misaligned_run, base_run_id=base_run_id, steps=steps, layer=layer)
    ctl = trajectory(root, control_run, base_run_id=base_run_id,
                     steps=control_steps or steps, layer=layer)
    if not mis or not ctl:
        return {"status": "no_trajectory"}
    mis_shift = mis[0]["relative_to_base"] - mis[-1]["relative_to_base"]
    ctl_shift = ctl[0]["relative_to_base"] - ctl[-1]["relative_to_base"]
    return {
        "status": "ok",
        "misaligned_shift_toward_base": mis_shift,
        "control_shift_toward_base": ctl_shift,
        "excess": mis_shift - ctl_shift,
        "misaligned_trajectory": mis,
        "control_trajectory": ctl,
    }


def run(registry: Registry, root, *, misaligned_run: str, control_run: str,
        base_run_id: str, steps: Sequence[int],
        control_steps: Sequence[int] | None = None,
        layer: int | None = None) -> list[Verdict]:
    result = erosion_test(root, misaligned_run=misaligned_run, control_run=control_run,
                          base_run_id=base_run_id, steps=steps,
                          control_steps=control_steps, layer=layer)
    return [Verdict(
        hypothesis=HYPOTHESIS, test="regression_to_base", run_id=misaligned_run,
        criterion="the misaligned fine-tune moves toward the base model more than a "
                  "matched benign fine-tune trained identically",
        effect=result.get("misaligned_shift_toward_base"),
        control_effect=result.get("control_shift_toward_base"),
        passed=(None if result.get("status") != "ok" else bool(result["excess"] > 0)),
        n=len(steps),
        detail={"status": result.get("status"), "control_run": control_run,
                "excess": result.get("excess")},
    )]


def entry(ctx):
    from emerge.core import paths

    required = ("misaligned_run", "control_run", "base_run_id")
    missing = [k for k in required if not ctx.cfg.get(k)]
    if missing:
        raise ValueError(
            f"hypothesis=erosion requires {required}; missing {missing}. "
            "The control run must be a matched benign fine-tune "
            "(data.condition=aligned) and base_run_id the probed base model.")
    run_steps = sorted({s for s, _ in paths.list_checkpoints(ctx.root, ctx.cfg["misaligned_run"])})
    return run(ctx.registry, ctx.root, misaligned_run=ctx.cfg["misaligned_run"],
               control_run=ctx.cfg["control_run"], base_run_id=ctx.cfg["base_run_id"],
               steps=run_steps, layer=ctx.cfg.get("layer"))
