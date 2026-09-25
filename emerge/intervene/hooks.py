"""Forward-hook machinery shared by every activation-space intervention.

All of the mechanism tests reduce to a small number of operations on the
residual stream: read it, add a vector to it, remove a direction from it, or
replace part of it with a cached tensor.  Implementing them once means that
"steer", "ablate", "transplant" and "patch" differ only in which callable is
installed, and -- importantly -- that a real intervention and its matched random
control run through exactly the same code path.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator, Sequence

import torch


def decoder_blocks(model) -> list[torch.nn.Module]:
    """Locate the list of transformer blocks across common architectures."""
    candidates = (
        "model.layers", "model.model.layers", "transformer.h", "model.transformer.h",
        "gpt_neox.layers", "model.decoder.layers", "layers",
    )
    for path in candidates:
        obj: Any = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, (torch.nn.ModuleList, list)) and len(obj) > 0:
            return list(obj)
    raise AttributeError(
        f"could not locate decoder blocks on {type(model).__name__}; "
        "extend emerge.intervene.hooks.decoder_blocks for this architecture"
    )


def n_layers(model) -> int:
    return len(decoder_blocks(model))


def hidden_size(model) -> int:
    cfg = getattr(model, "config", None)
    for attr in ("hidden_size", "n_embd", "d_model"):
        if cfg is not None and getattr(cfg, attr, None):
            return int(getattr(cfg, attr))
    raise AttributeError("could not determine hidden size from model config")


def _unpack(output):
    """Blocks return either a tensor or a tuple whose first element is one."""
    if isinstance(output, tuple):
        return output[0], lambda new: (new,) + output[1:]
    return output, lambda new: new


@contextmanager
def capture(model, layers: Sequence[int]) -> Iterator[dict[int, torch.Tensor]]:
    """Collect each requested block's output for one forward pass.

    Yields a dict that is filled during the ``with`` body; tensors are detached
    but *not* moved off device, so the caller decides on the copy.
    """
    blocks = decoder_blocks(model)
    store: dict[int, torch.Tensor] = {}
    handles = []

    def make(idx: int):
        def hook(_m, _i, output):
            tensor, _ = _unpack(output)
            store[idx] = tensor.detach()
        return hook

    try:
        for idx in layers:
            handles.append(blocks[idx].register_forward_hook(make(idx)))
        yield store
    finally:
        for h in handles:
            h.remove()


@contextmanager
def edit(model, edits: dict[int, Callable[[torch.Tensor], torch.Tensor]]) -> Iterator[None]:
    """Apply a per-layer transform to the residual stream during forward passes.

    ``edits`` maps layer index to ``fn(hidden) -> hidden``.
    """
    blocks = decoder_blocks(model)
    handles = []

    def make(fn):
        def hook(_m, _i, output):
            tensor, rebuild = _unpack(output)
            return rebuild(fn(tensor))
        return hook

    try:
        for idx, fn in edits.items():
            handles.append(blocks[idx].register_forward_hook(make(fn)))
        yield
    finally:
        for h in handles:
            h.remove()


# ----------------------------------------------------------------------
# the four primitive edits
# ----------------------------------------------------------------------


def add_vector(vector: torch.Tensor, coefficient: float = 1.0,
               positions: slice | None = None) -> Callable[[torch.Tensor], torch.Tensor]:
    """Steering: ``h += coefficient * vector``."""

    def fn(h: torch.Tensor) -> torch.Tensor:
        v = vector.to(h.device, h.dtype).reshape(1, 1, -1)
        if positions is None:
            return h + coefficient * v
        out = h.clone()
        out[:, positions, :] = out[:, positions, :] + coefficient * v
        return out

    return fn


def ablate_direction(direction: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """Directional ablation: remove the component along a unit direction."""
    d = direction / (direction.norm() + 1e-8)

    def fn(h: torch.Tensor) -> torch.Tensor:
        v = d.to(h.device, h.dtype)
        return h - (h @ v).unsqueeze(-1) * v

    return fn


def project_out(basis: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """Subspace ablation: remove the component inside ``span(basis)``.

    ``basis`` is ``(d_model, k)``; it is re-orthonormalised so a stored or
    randomly generated basis behaves identically.
    """
    Q, _ = torch.linalg.qr(basis.float())

    def fn(h: torch.Tensor) -> torch.Tensor:
        U = Q.to(h.device, h.dtype)
        return h - (h @ U) @ U.T

    return fn


def patch_positions(source: torch.Tensor, positions: slice) -> Callable[[torch.Tensor], torch.Tensor]:
    """Activation patching: overwrite ``positions`` with cached activations."""

    def fn(h: torch.Tensor) -> torch.Tensor:
        out = h.clone()
        src = source.to(h.device, h.dtype)
        width = out[:, positions, :].shape[1]
        out[:, positions, :] = src[:, :width, :]
        return out

    return fn
