"""Cross-task linearity: features of an interpolated model vs interpolated features.

The paper's strongest structural claim is that weight-space interpolation and
feature-space interpolation commute: the activations of the model at
``theta`` are (approximately) the linear interpolation of the endpoint
activations, layer by layer.

Endpoint activations come from the capture bank (pooled artifacts written by
``emerge capture``); the interpolated model has no such artifact, so its
activations are computed live on the *same prompts*.  The comparison is only
valid when prompt identity and ordering match exactly -- when that cannot be
guaranteed (missing artifacts, differing prompt ids, ids that do not resolve
in the supplied bank, mismatched layer counts) the test reports itself as
skipped with the reason, rather than quietly correlating mismatched prompts.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from emerge.capture.api import activation_steps, load_activations
from emerge.core.seeding import derive_seed
from emerge.data.schema import load_bank

from plugins.parameter_subspace.mechanism.interpolate import apply_interpolated_delta


def matched_step(root, run_a: str, run_b: str) -> int | None:
    """Latest checkpoint step for which both runs have pooled activations."""
    common = set(activation_steps(root, run_a)) & set(activation_steps(root, run_b))
    return max(common) if common else None


def layer_linearity(interp: np.ndarray, linear_mix: np.ndarray) -> dict[str, list[float]]:
    """Per-layer normalized L2 and R^2 between measured and predicted features.

    ``interp`` is the measured activation tensor of the interpolated model,
    ``linear_mix`` the prediction ``(1-theta)*acts_a + theta*acts_b``; both are
    ``(L, N, D)``.  R^2 treats the measured features as the target, so 1 means
    the linear prediction explains them completely.
    """
    l2, r2 = [], []
    for layer in range(interp.shape[0]):
        y = interp[layer].astype(np.float64)
        yhat = linear_mix[layer].astype(np.float64)
        resid = float(((y - yhat) ** 2).sum())
        total = float(((y - y.mean(axis=0, keepdims=True)) ** 2).sum())
        l2.append(float(np.sqrt(resid) / (np.sqrt((yhat ** 2).sum()) + 1e-12)))
        r2.append(1.0 - resid / (total + 1e-12))
    return {"normalized_l2": l2, "r2": r2}


def feature_linearity(
    root,
    run_a: str,
    run_b: str,
    *,
    model,
    tokenizer,
    deltas_a: Mapping[str, np.ndarray],
    deltas_b: Mapping[str, np.ndarray],
    bank_path: str | None,
    theta: float = 0.5,
    position: str = "prompt_last",
    batch_size: int = 8,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare live features at ``theta`` against interpolated endpoint features.

    Returns ``{"status": "ok", ...}`` with per-layer metrics and a
    sample-permuted control, or ``{"status": "skipped", "reason": ...}`` when
    the prompt/id alignment between capture artifacts and the bank cannot be
    guaranteed.
    """
    step = matched_step(root, run_a, run_b)
    if step is None:
        return {"status": "skipped",
                "reason": "no common checkpoint step with pooled activations for both runs "
                          "(run `emerge capture` on both first)"}
    try:
        acts_a, ids_a = load_activations(root, run_a, step)
        acts_b, ids_b = load_activations(root, run_b, step)
    except (FileNotFoundError, ValueError) as exc:
        return {"status": "skipped", "reason": f"endpoint activations unavailable: {exc}"}
    if not ids_a or ids_a != ids_b:
        return {"status": "skipped",
                "reason": "endpoint activations were captured on differing or unrecorded "
                          "prompt ids; the pooled comparison would mix prompts"}
    if acts_a.shape != acts_b.shape:
        return {"status": "skipped",
                "reason": f"endpoint activation shapes differ ({acts_a.shape} vs {acts_b.shape})"}
    if not bank_path:
        return {"status": "skipped",
                "reason": "no prompt bank supplied (+feature_bank=<path>): cannot recompute "
                          "the interpolated model's activations on the capture prompts"}
    by_id = {}
    for i, row in enumerate(load_bank(bank_path)):
        by_id[str(row.get("id") or f"{row.get('sub_category', 'q')}-{i}")] = str(row["prompt"])
    missing = [i for i in ids_a if i not in by_id]
    if missing:
        return {"status": "skipped",
                "reason": f"prompt ids {missing[:4]} from the capture artifacts do not "
                          f"resolve in bank {bank_path}; prompt alignment not guaranteed"}
    prompts = [by_id[i] for i in ids_a]

    try:
        from emerge.capture.activations import capture_activations
    except ImportError as exc:  # pragma: no cover - kernel module should exist
        return {"status": "skipped",
                "reason": f"emerge.capture.activations not importable: {exc}"}

    with apply_interpolated_delta(model, deltas_a, deltas_b, theta):
        live = capture_activations(model, tokenizer, prompts, position=position,
                                   batch_size=batch_size, dtype="float32")
    interp = np.asarray(live["acts"], dtype=np.float32)
    if interp.shape != acts_a.shape:
        return {"status": "skipped",
                "reason": f"live capture shape {interp.shape} does not match the stored "
                          f"artifacts {acts_a.shape}; the capture layer/position settings "
                          "differ and layer alignment cannot be guaranteed"}

    mix = ((1.0 - theta) * acts_a.astype(np.float32) + theta * acts_b.astype(np.float32))
    metrics = layer_linearity(interp, mix)

    # Matched control: the same linear prediction with its samples permuted, so
    # the comparison credits prompt-level correspondence rather than the shared
    # activation statistics of one model family.
    rng = np.random.default_rng(derive_seed(run_a, run_b, "feature_perm", seed=seed))
    perm = rng.permutation(mix.shape[1])
    if mix.shape[1] > 1 and (perm == np.arange(mix.shape[1])).all():
        perm = np.roll(perm, 1)  # an identity draw would make the control vacuous
    control = layer_linearity(interp, mix[:, perm, :])

    return {
        "status": "ok",
        "step": int(step),
        "theta": float(theta),
        "n_prompts": len(prompts),
        "per_layer": metrics,
        "mean_r2": float(np.mean(metrics["r2"])),
        "mean_normalized_l2": float(np.mean(metrics["normalized_l2"])),
        "control_mean_r2": float(np.mean(control["r2"])),
    }
