"""Matched controls.

Every causal claim in the replication plan is of the form "the intervention
beats a matched control", never "the intervention has an effect".  A cross-model
direction can produce large behavioural changes while random and orthogonal
controls perform comparably -- so a control that is not matched on the property
being credited (norm, rank, usage) does not license any conclusion.

The controls here are reproducible from their description alone: a seed and a
shape regenerate the exact tensor, so a control never has to be stored.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from emerge.core.seeding import derive_seed


def _generator(*parts, seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(derive_seed(*parts, seed=seed) % (2**63 - 1))
    return g


def random_direction(d_model: int, *, seed: int = 0, tag: str = "dir") -> torch.Tensor:
    """A uniformly random unit direction."""
    g = _generator(tag, d_model, seed=seed)
    v = torch.randn(d_model, generator=g, dtype=torch.float32)
    return v / v.norm()


def norm_matched_random(reference: torch.Tensor, *, seed: int = 0,
                        tag: str = "norm_matched") -> torch.Tensor:
    """A random direction rescaled to ``reference``'s norm.

    The control for "steering with the discovered vector changed behaviour":
    without norm matching, the comparison confounds direction with magnitude.
    """
    v = random_direction(reference.numel(), seed=seed, tag=tag)
    return (v * reference.norm()).to(reference.dtype)


def orthogonal_direction(reference: torch.Tensor, *, seed: int = 0,
                         tag: str = "orthogonal") -> torch.Tensor:
    """A random direction orthogonal to ``reference``, matched in norm.

    Stronger than an unconstrained random control: it rules out the possibility
    that any large perturbation with a component along the reference works.
    """
    ref = reference.float()
    u = ref / (ref.norm() + 1e-8)
    v = random_direction(ref.numel(), seed=seed, tag=tag)
    v = v - (v @ u) * u
    v = v / (v.norm() + 1e-8)
    return (v * ref.norm()).to(reference.dtype)


def random_orthonormal_basis(d_model: int, k: int, *, seed: int = 0,
                             tag: str = "basis") -> torch.Tensor:
    """A rank-matched random orthonormal basis, shape ``(d_model, k)``."""
    if k > d_model:
        raise ValueError(f"cannot build a rank-{k} basis in {d_model} dimensions")
    g = _generator(tag, d_model, k, seed=seed)
    M = torch.randn(d_model, k, generator=g, dtype=torch.float32)
    Q, _ = torch.linalg.qr(M)
    return Q[:, :k]


def usage_matched_basis(activations: torch.Tensor, k: int, *, seed: int = 0,
                        tag: str = "usage") -> torch.Tensor:
    """A random basis matched to how strongly the model *uses* each direction.

    A rank-matched random subspace is typically almost unused by the model, so
    projecting it out costs nothing and the real subspace looks special for a
    trivial reason.  This control samples a random subspace within the model's
    high-variance activation subspace, so the real and control subspaces carry
    comparable activation mass.

    ``activations`` is ``(n_samples, d_model)``.
    """
    X = activations.float()
    X = X - X.mean(0, keepdim=True)
    # Principal directions, truncated well above k so the random draw has room.
    q = min(X.shape[0], X.shape[1], max(4 * k, k + 8))
    U, S, _ = torch.linalg.svd(X, full_matrices=False)
    V = (X.T @ U[:, :q]) / (S[:q] + 1e-8)
    g = _generator(tag, k, seed=seed)
    M = torch.randn(V.shape[1], k, generator=g, dtype=torch.float32)
    Q, _ = torch.linalg.qr(V @ M)
    return Q[:, :k]


def shuffled_direction(reference: torch.Tensor, *, seed: int = 0,
                       tag: str = "shuffled") -> torch.Tensor:
    """``reference`` with its coordinates permuted.

    Preserves the exact norm *and* the coordinate value distribution, isolating
    the claim to the direction's geometry rather than its statistics.
    """
    g = _generator(tag, reference.numel(), seed=seed)
    perm = torch.randperm(reference.numel(), generator=g)
    return reference.flatten()[perm].reshape(reference.shape).clone()


def control_family(reference: torch.Tensor, *, seed: int = 0,
                   include: Sequence[str] = ("random", "orthogonal", "shuffled")) -> dict[str, torch.Tensor]:
    """Build the standard control set for a single direction."""
    out: dict[str, torch.Tensor] = {}
    if "random" in include:
        out["random"] = norm_matched_random(reference, seed=seed)
    if "orthogonal" in include:
        out["orthogonal"] = orthogonal_direction(reference, seed=seed)
    if "shuffled" in include:
        out["shuffled"] = shuffled_direction(reference, seed=seed)
    return out


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two vectors, as a plain float."""
    a, b = a.flatten().float(), b.flatten().float()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def subspace_overlap(A: torch.Tensor, B: torch.Tensor) -> float:
    """Mean squared principal cosine between two subspaces, in ``[0, 1]``.

    1 means the spans coincide; the expected value for two random k-dimensional
    subspaces of a d-dimensional space is ``k/d``, which is the null this should
    always be compared against.
    """
    QA, _ = torch.linalg.qr(A.float())
    QB, _ = torch.linalg.qr(B.float())
    s = torch.linalg.svdvals(QA.T @ QB)
    return float((s**2).mean())
