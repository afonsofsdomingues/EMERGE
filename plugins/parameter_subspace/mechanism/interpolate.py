"""Weight-space interpolation between LoRA fine-tunes of one base model.

Linear mode connectivity is a claim about *merged* updates: the model at
``W_base + (1-theta)*dW_1 + theta*dW_2`` for every adapted matrix.  Working
with effective deltas (``lora_deltas``: rsLoRA-aware ``scale * B @ A``) rather
than interpolating A/B factors matters, because the factor parameterisation is
not unique and interpolating factors of two adapters does not interpolate
their updates.

The deltas are saved under PEFT names (``model.layers.N...q_proj`` after the
``base_model.model.`` prefix is stripped by ``lora_deltas``), while a live
model names the same tensor ``model.layers.N...q_proj.weight`` -- or
``...q_proj.base_layer.weight`` when the model is itself PEFT-wrapped.
:func:`delta_param_map` resolves that mapping explicitly and loudly instead of
relying on any single spelling.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Mapping, Sequence

import numpy as np
import torch


def delta_param_map(delta_names: Sequence[str], model) -> dict[str, str]:
    """Map ``lora_deltas`` matrix names onto ``model``'s parameter names.

    Tries, in order: the plain HF spelling ``<name>.weight``, the PEFT-wrapped
    spelling ``<name>.base_layer.weight``, then a unique suffix match of either
    (which absorbs extra wrapper prefixes such as ``base_model.model.``).
    Anything unresolvable or ambiguous raises with the candidates shown, so a
    naming drift in a new model family fails loudly rather than silently
    interpolating the wrong tensors.
    """
    params = dict(model.named_parameters())
    out: dict[str, str] = {}
    for name in delta_names:
        exact = [f"{name}.weight", f"{name}.base_layer.weight"]
        hit = next((c for c in exact if c in params), None)
        if hit is None:
            suffixes = tuple(f".{c}" for c in exact)
            matches = [p for p in params if p.endswith(suffixes)]
            if len(matches) > 1:
                raise ValueError(
                    f"delta matrix {name!r} matches several model parameters {matches}; "
                    "cannot interpolate an ambiguous mapping"
                )
            hit = matches[0] if matches else None
        if hit is None:
            sample = [p for p in list(params)[:5]]
            raise ValueError(
                f"delta matrix {name!r} has no counterpart among the model's parameters "
                f"(looked for {exact} and suffix matches; parameters look like {sample}). "
                "Are the adapter and the model from the same base architecture?"
            )
        out[name] = hit
    return out


def check_pairing(deltas1: Mapping[str, np.ndarray],
                  deltas2: Mapping[str, np.ndarray]) -> list[str]:
    k1, k2 = set(deltas1), set(deltas2)
    if k1 != k2:
        raise ValueError(
            f"cannot interpolate adapters with different matrix sets: "
            f"only in first {sorted(k1 - k2)[:4]}, only in second {sorted(k2 - k1)[:4]}. "
            "Both runs must target the same modules/layers."
        )
    for name in k1:
        if deltas1[name].shape != deltas2[name].shape:
            raise ValueError(
                f"delta {name!r} has mismatched shapes {deltas1[name].shape} vs "
                f"{deltas2[name].shape}; the runs are not fine-tunes of the same base model"
            )
    return sorted(k1)


@contextmanager
def apply_interpolated_delta(
    model,
    deltas1: Mapping[str, np.ndarray],
    deltas2: Mapping[str, np.ndarray],
    theta: float,
) -> Iterator[None]:
    """Add ``(1-theta)*dW1 + theta*dW2`` into the base weights, then restore.

    Restoration is by cloned originals rather than by subtracting the delta
    again: floating-point add/subtract does not round-trip bit-exactly, and a
    mechanism that leaves the model microscopically different after every
    measurement makes paths order-dependent.  The clones cover only the adapted
    matrices, so memory scales with the adapter's footprint, not the model's.
    """
    names = check_pairing(deltas1, deltas2)
    mapping = delta_param_map(names, model)
    params = dict(model.named_parameters())
    originals: dict[str, torch.Tensor] = {}
    try:
        with torch.no_grad():
            for name in names:
                param = params[mapping[name]]
                mix = (1.0 - theta) * np.asarray(deltas1[name], dtype=np.float32) \
                    + theta * np.asarray(deltas2[name], dtype=np.float32)
                if tuple(mix.shape) != tuple(param.shape):
                    raise ValueError(
                        f"delta {name!r} has shape {mix.shape} but parameter "
                        f"{mapping[name]!r} has shape {tuple(param.shape)}"
                    )
                originals[mapping[name]] = param.detach().clone()
                param.add_(torch.from_numpy(mix).to(device=param.device, dtype=param.dtype))
        yield
    finally:
        with torch.no_grad():
            for pname, original in originals.items():
                params[pname].copy_(original)


def permuted_pairing(deltas: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Rotate matrices among same-shape positions: the layer-scrambled control.

    Pairing ``dW1[layer i]`` with ``dW2[layer i+1]`` keeps every per-matrix
    statistic of the second adapter intact while destroying the layerwise
    correspondence the connectivity claim is about.  A rotation (rather than a
    sampled permutation) is a guaranteed derangement within every group of two
    or more same-shape matrices, and needs no seed to be reproducible.
    """
    groups: dict[tuple[int, ...], list[str]] = {}
    for name in sorted(deltas):
        groups.setdefault(tuple(deltas[name].shape), []).append(name)
    out: dict[str, np.ndarray] = {}
    for names in groups.values():
        rotated = names[1:] + names[:1]
        for src, dst in zip(rotated, names):
            out[dst] = deltas[src]
    return out


def theta_grid(points: int) -> list[float]:
    """Evenly spaced interpolation coefficients including both endpoints."""
    if points < 3:
        raise ValueError(
            f"lmc_points={points}: a connectivity path needs both endpoints plus "
            "at least one interior point (>= 3)"
        )
    return [float(t) for t in np.linspace(0.0, 1.0, points)]


def lmc_path(
    model,
    deltas1: Mapping[str, np.ndarray],
    deltas2: Mapping[str, np.ndarray],
    thetas: Sequence[float],
    measure,
) -> list[float]:
    """Misalignment rate at every point of the interpolation path.

    ``measure`` is the zero-argument callable from
    :func:`emerge.mechanism.live.em_measure`; it reads the model as currently
    modified, so applying each interpolation in turn and calling it keeps every
    point of the path paired on identical prompts and sampling seeds.
    """
    rates: list[float] = []
    for theta in thetas:
        with apply_interpolated_delta(model, deltas1, deltas2, theta):
            rates.append(float(measure()))
    return rates


def path_summary(thetas: Sequence[float], rates: Sequence[float]) -> dict[str, float | None]:
    """Interior minimum and barrier depth of one measured path.

    ``min_interior`` is the connectivity statistic (does misalignment survive
    everywhere between the endpoints?); ``barrier_depth`` re-expresses it
    relative to the weaker endpoint.  NaN rates (a point where every sample was
    filtered by the coherence gate) are ignored; an all-NaN path returns None.
    """
    interior = [r for t, r in zip(thetas, rates) if 0.0 < t < 1.0 and r == r]
    endpoints = [r for t, r in zip(thetas, rates) if t in (0.0, 1.0) and r == r]
    min_interior = min(interior) if interior else None
    min_endpoint = min(endpoints) if endpoints else None
    depth = (min_endpoint - min_interior
             if min_interior is not None and min_endpoint is not None else None)
    return {"min_interior": min_interior, "min_endpoint": min_endpoint,
            "barrier_depth": depth}
