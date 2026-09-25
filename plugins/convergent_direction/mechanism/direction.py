"""Convergent linear representations and persona causality.

Two related claims share this module.  *Convergence*: independently trained
misaligned fine-tunes recruit approximately the same linear direction.
*Causality*: that direction mediates the behaviour -- adding it induces
misalignment, ablating it suppresses it -- including when transplanted from a
donor model into a recipient.

Both are tested against explicit nulls.  Cosine similarities between
high-dimensional vectors need the random baseline ``1/sqrt(d)`` to be
interpretable, and transferred directions in particular can produce large
behavioural effects that norm-matched random and orthogonal controls match --
so the causal test is defined as "beats every matched control", not "has an
effect".
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import torch

from emerge.analyze import api as analyze_api
from emerge.capture.api import load_activations
from emerge.core.registry import Registry
from emerge.intervene import controls as ctrl
from plugins.convergent_direction.intervene.directions import (
    convergence,
    extract_direction,
    extract_direction_judged,
    verdict,
    with_controls,
)
from emerge.mechanism.base import Verdict

HYPOTHESIS = "convergent_direction"


def direction_for_run(
    root,
    run_id: str,
    *,
    step: int,
    baseline_step: int = 0,
    layer: int | None = None,
    split: str = "step_vs_baseline",
    judged_rows: Sequence[dict] | None = None,
) -> tuple[torch.Tensor, int]:
    """Direction separating misaligned from aligned activation sets.

    ``split="step_vs_baseline"`` (the default) contrasts a checkpoint's
    activations with its own step-0 activations.
    Using the run's *own* baseline rather than a pooled reference removes the
    model's idiosyncratic activation geometry, which would otherwise dominate
    the cosine between directions taken from different model families.

    ``split="judged"`` is the paper's extraction (arXiv 2506.11618): the mean
    difference between the checkpoint's own responses judged misaligned
    (alignment < 30) and aligned (alignment > 70).  It needs ``judged_rows`` --
    the judged generation rows, one per activation sample and in the same
    order (capture with ``position=response_mean`` on the scored responses).
    """
    if split == "step_vs_baseline":
        late, _ = load_activations(root, run_id, step)
        early, _ = load_activations(root, run_id, baseline_step)
        direction, idx, _ = extract_direction(late, early, layer=layer)
        return direction, idx
    if split == "judged":
        if judged_rows is None:
            raise ValueError(
                "split='judged' needs judged_rows: the judged generation rows for "
                f"run {run_id} step {step}, in the same order as the captured "
                "activation samples."
            )
        acts, _ = load_activations(root, run_id, step)
        direction, idx, _ = extract_direction_judged(acts, judged_rows, layer=layer)
        return direction, idx
    raise ValueError(f"unknown split {split!r}; use 'step_vs_baseline' or 'judged'")


def convergence_test(
    root,
    run_ids: Sequence[str],
    *,
    steps: dict[str, int],
    baseline_step: int = 0,
    layer: int | None = None,
    split: str = "step_vs_baseline",
    judged_rows: dict[str, Sequence[dict]] | None = None,
) -> dict[str, Any]:
    """Cosine similarity among directions from independently trained runs.

    Only meaningful for runs sharing a model (activation coordinates are not
    comparable across architectures), so a mixed set is rejected rather than
    quietly averaged.  ``split``/``judged_rows`` (keyed by run id) select the
    paper's judged 30/70 extraction; see :func:`direction_for_run`.
    """
    directions: dict[str, torch.Tensor] = {}
    layers: dict[str, int] = {}
    for run_id in run_ids:
        d, idx = direction_for_run(root, run_id, step=steps[run_id],
                                   baseline_step=baseline_step, layer=layer,
                                   split=split,
                                   judged_rows=(judged_rows or {}).get(run_id))
        directions[run_id] = d
        layers[run_id] = idx
    dims = {int(d.numel()) for d in directions.values()}
    if len(dims) > 1:
        raise ValueError(
            f"directions have differing dimensions {sorted(dims)}: cosine similarity across "
            "architectures is not defined. Compare within a model family."
        )
    out = convergence(directions)
    out["layers"] = layers
    return out


def causal_test(
    model,
    direction: torch.Tensor,
    layer: int,
    measure: Callable[[], float],
    *,
    mode: str = "steer",
    coefficient: float = 1.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Apply the direction and its matched controls through one code path."""
    results = with_controls(model, direction, layer, measure,
                            coefficient=coefficient, mode=mode, seed=seed)
    return {"results": results, **verdict(results)}


def run(
    registry: Registry,
    root,
    run_ids: Sequence[str],
    *,
    steps: dict[str, int],
    baseline_step: int = 0,
    layer: int | None = None,
    split: str = "step_vs_baseline",
    judged_rows: dict[str, Sequence[dict]] | None = None,
    causal: dict[str, Any] | None = None,
) -> list[Verdict]:
    """Adjudicate convergence, and causality when a measurement is supplied."""
    verdicts: list[Verdict] = []
    tag = ",".join(sorted(run_ids)[:3])

    if len(run_ids) >= 2:
        conv = convergence_test(root, run_ids, steps=steps,
                                baseline_step=baseline_step, layer=layer,
                                split=split, judged_rows=judged_rows)
        verdicts.append(Verdict(
            hypothesis=HYPOTHESIS, test="convergence", run_id=tag,
            criterion="mean pairwise cosine between independently trained directions "
                      "exceeds 5 standard deviations of the random-vector null",
            effect=conv["mean_cosine"], control_effect=5 * conv["null_sd"],
            passed=bool(conv["z_vs_null"] >= 5.0),
            n=len(run_ids),
            detail={"min_cosine": conv["min_cosine"], "z_vs_null": conv["z_vs_null"],
                    "layers": conv["layers"]},
        ))

    if causal:
        verdicts.append(Verdict(
            hypothesis=HYPOTHESIS, test=f"causal_{causal.get('mode', 'steer')}",
            run_id=causal.get("run_id", tag),
            criterion="the direction moves behaviour more than every norm-matched control "
                      "(random, orthogonal, coordinate-shuffled)",
            effect=causal.get("effect"), control_effect=causal.get("control_effect_max"),
            passed=causal.get("passed"),
            detail={"results": causal.get("results"), "ratio": causal.get("ratio")},
        ))
    return verdicts


def entry(ctx):
    """Registry runner; ``split=judged`` reads each run's judged generations
    for the paper's 30/70 extraction via the analyze accessor."""
    split = str(ctx.cfg.get("split") or "step_vs_baseline")
    judged = None
    if split == "judged":
        judged = {rid: analyze_api.generations(ctx.root, rid, ctx.suite,
                                               step=ctx.steps[rid])
                  for rid in ctx.selected}
    return run(ctx.registry, ctx.root, ctx.selected, steps=ctx.steps,
               layer=ctx.cfg.get("layer"), split=split, judged_rows=judged)
