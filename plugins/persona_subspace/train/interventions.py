"""Subspace-projection intervention for the persona-subspace study."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from emerge.train.interventions import Intervention, _named_linear_weights, lora_pairs




# ----------------------------------------------------------------------
# subspace projection
# ----------------------------------------------------------------------


class ProjectSubspace(Intervention):
    """Project a subspace out of the weight update after every optimiser step.

    For LoRA the update is ``B @ A``, so removing the subspace from the output
    space is exactly ``B <- (I - U U^T) B`` -- cheap and exact.  For full
    fine-tuning the same projection is applied to the delta from the initial
    weights, which requires keeping a copy of the targeted matrices.

    ``params``
        ``basis``: path to an ``.npz``/``.npy`` holding ``U`` with shape
        ``(d_model, k)``; ``modules``: substrings selecting target matrices;
        ``random_control``/``seed``: replace ``U`` with a rank-matched random
        orthonormal basis, which is the control this test is meaningless without.
    """

    name = "project_subspace"

    def __init__(self):
        self._U: torch.Tensor | None = None
        self._pairs: list = []
        self._full: list = []
        self._init: dict[str, torch.Tensor] = {}
        self._removed = 0.0

    def setup(self, model, tokenizer, params):
        import numpy as np

        from emerge.intervene.controls import random_orthonormal_basis

        modules = tuple(params.get("modules", ("o_proj", "down_proj")))
        rank = int(params.get("rank") or 8)

        if params.get("random_control"):
            d = params.get("d_model")
            if not d:
                raise ValueError(
                    "project_subspace with random_control=true needs params.d_model "
                    "(the model's hidden size), so the null has the same shape as the "
                    "real subspace"
                )
            U = random_orthonormal_basis(int(d), rank, seed=int(params.get("seed", 0)))
        else:
            path = params.get("basis")
            if not path:
                raise ValueError(
                    "project_subspace needs params.basis (an .npz written by "
                    "emerge.mechanism.subspace.export_for_training) or params.random_control=true"
                )
            raw = np.load(path)
            U = raw["U"] if hasattr(raw, "files") and "U" in raw.files else np.asarray(raw)
            U = U[:, :rank]
        self._U = torch.as_tensor(U, dtype=torch.float32)
        # Re-orthonormalise defensively: a stored basis may have drifted.
        self._U, _ = torch.linalg.qr(self._U)

        self._pairs = [(n, a, b) for n, a, b in lora_pairs(model)
                       if any(m in n for m in modules)]
        if not self._pairs:
            self._full = _named_linear_weights(model, modules)
            self._init = {n: p.detach().clone() for n, p in self._full}
            if not self._full:
                raise ValueError(
                    f"project_subspace matched no parameters for modules={modules}"
                )

    @torch.no_grad()
    def after_step(self, model):
        assert self._U is not None
        removed = 0.0
        for _, _a, b_mod in self._pairs:
            W = b_mod.weight  # (out_features, r)
            U = self._U.to(W.device, W.dtype)
            if U.shape[0] != W.shape[0]:
                raise ValueError(
                    f"subspace basis has d={U.shape[0]} but adapter output dim is {W.shape[0]}"
                )
            proj = U @ (U.T @ W)
            removed += float(proj.norm())
            W.sub_(proj)
        for name, param in self._full:
            U = self._U.to(param.device, param.dtype)
            delta = param - self._init[name]
            if U.shape[0] != delta.shape[0]:
                continue
            proj = U @ (U.T @ delta)
            removed += float(proj.norm())
            param.sub_(proj)
        self._removed = removed

    def telemetry(self):
        return {"subspace_removed_norm": self._removed}


def register_interventions(reg) -> None:
    reg.register("interventions", "project_subspace", ProjectSubspace)
