"""Low-rank subspaces: construction, projection, injection, comparison.

The pre-existing-persona-subspace claim has three parts -- the subspace exists
*before* the misaligned fine-tune, projecting it out during training prevents
emergent misalignment, and injecting it induces it.  Each part needs a subspace
built from a different source, so construction is separated from use:

* from activations (a persona probe measured on the untouched model),
* from an adapter's update (the span the fine-tune actually wrote into),
* from a random or usage-matched control.

Everything is returned as an orthonormal ``(d_model, k)`` basis so downstream
code never has to care where a subspace came from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from emerge.intervene.controls import random_orthonormal_basis, subspace_overlap, usage_matched_basis


def orthonormalise(matrix: torch.Tensor) -> torch.Tensor:
    Q, _ = torch.linalg.qr(torch.as_tensor(matrix).float())
    return Q


def subspace_from_activations(acts: np.ndarray | torch.Tensor, k: int,
                              *, centre: bool = True) -> tuple[torch.Tensor, np.ndarray]:
    """Top-``k`` principal directions of an activation set.

    Returns ``(basis, explained_variance_ratio)``.  The variance ratio is
    returned because "we found a rank-8 subspace" is only interesting relative
    to how much of the activation it accounts for.
    """
    X = torch.as_tensor(np.asarray(acts)).float()
    if X.ndim != 2:
        raise ValueError(f"expected (n_samples, d_model), got {tuple(X.shape)}")
    if centre:
        X = X - X.mean(0, keepdim=True)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    total = float((S**2).sum()) or 1.0
    ratio = ((S**2) / total).numpy()[:k]
    return Vh[:k].T.contiguous(), ratio


def subspace_from_lora(adapter_dir: str | Path, k: int | None = None,
                       *, modules: Sequence[str] = ()) -> tuple[torch.Tensor, np.ndarray]:
    """The output-space span written by a LoRA adapter.

    ``B @ A`` maps into residual space through ``B``'s columns, so the left
    singular vectors of the stacked updates are the directions the fine-tune
    actually pushed on.
    """
    from emerge.capture.weights import lora_deltas

    deltas = lora_deltas(adapter_dir)
    selected = {n: d for n, d in deltas.items()
                if not modules or any(m in n for m in modules)}
    if not selected:
        raise ValueError(f"no LoRA matrices matched modules={list(modules)} in {adapter_dir}")
    dims = {d.shape[0] for d in selected.values()}
    if len(dims) > 1:
        raise ValueError(
            f"selected matrices have differing output dimensions {sorted(dims)}; "
            "restrict `modules` to matrices writing into the residual stream "
            "(e.g. o_proj, down_proj)"
        )
    stacked = torch.as_tensor(np.concatenate(list(selected.values()), axis=1)).float()
    U, S, _ = torch.linalg.svd(stacked, full_matrices=False)
    total = float((S**2).sum()) or 1.0
    rank = k or int(min(8, S.numel()))
    return U[:, :rank].contiguous(), ((S**2) / total).numpy()[:rank]


def project_out(vectors: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Remove the component of ``vectors`` inside ``span(basis)``."""
    Q = orthonormalise(basis)
    X = torch.as_tensor(vectors).float()
    return X - (X @ Q) @ Q.T


def project_onto(vectors: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    Q = orthonormalise(basis)
    X = torch.as_tensor(vectors).float()
    return (X @ Q) @ Q.T


def usage(acts: np.ndarray | torch.Tensor, basis: torch.Tensor) -> float:
    """Fraction of activation variance that lies inside the subspace.

    The number a rank-matched random control must be compared on: a random
    subspace is typically barely used, so projecting it out costs nothing and
    the real subspace looks special for a trivial reason.
    """
    X = torch.as_tensor(np.asarray(acts)).float()
    X = X - X.mean(0, keepdim=True)
    total = float((X**2).sum())
    if total <= 0:
        return float("nan")
    return float((project_onto(X, basis) ** 2).sum() / total)


def control_subspaces(
    basis: torch.Tensor,
    acts: np.ndarray | torch.Tensor | None = None,
    *,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Rank-matched random and (when activations are supplied) usage-matched nulls."""
    d, k = basis.shape
    out = {"random": random_orthonormal_basis(d, k, seed=seed)}
    if acts is not None:
        out["usage_matched"] = usage_matched_basis(
            torch.as_tensor(np.asarray(acts)).float(), k, seed=seed
        )
    return out


def compare(basis_a: torch.Tensor, basis_b: torch.Tensor) -> dict[str, float]:
    """Overlap between two subspaces against the random-subspace null ``k/d``."""
    d, k = basis_a.shape
    observed = subspace_overlap(basis_a, basis_b)
    null = k / d
    return {"overlap": observed, "null": null,
            "ratio": (observed / null) if null > 0 else float("inf")}


def principal_angles(basis_a: torch.Tensor, basis_b: torch.Tensor) -> np.ndarray:
    """Principal angles (radians) between two subspaces, ascending."""
    QA, QB = orthonormalise(basis_a), orthonormalise(basis_b)
    s = torch.linalg.svdvals(QA.T @ QB).clamp(-1.0, 1.0)
    return torch.arccos(s).numpy()


def save(path: str | Path, basis: torch.Tensor, meta: Mapping[str, Any] | None = None) -> Path:
    """Persist a basis as ``U`` so it can be handed to the training intervention."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, U=orthonormalise(basis).numpy(),
                        meta=np.array([str(dict(meta or {}))]))
    return p


def load(path: str | Path) -> torch.Tensor:
    with np.load(path, allow_pickle=False) as data:
        return torch.as_tensor(data["U"]).float()
