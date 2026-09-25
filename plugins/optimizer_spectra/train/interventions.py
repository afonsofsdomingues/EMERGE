"""Spectral-flattening interventions and their norm-matched control.

Evil Spectra (arXiv 2606.31591) released no code; the two paper forms here are
reconstructions from App. G.1 and are documented as such:

* ``fro_nuclear`` -- the paper's selected regulariser
  ``R(sigma) = ||BA||_F / ||BA||_* - 1/sqrt(r)``, zero exactly at a uniform
  rank-``r`` spectrum, averaged over all LoRA layers (lambda in {1, 3, 5},
  3 selected -- the preset weight).
* ``entropy_gap`` -- the paper's spectral-entropy variant:
  ``log r - H(sigma / ||sigma||_1)``, also zero at a uniform spectrum.

The ``cv`` form (coefficient of variation of the singular values) is the
default.
"""
from __future__ import annotations

import math

import torch

from emerge.train.interventions import Intervention, lora_pairs, lora_delta

# A LoRA update whose nuclear norm is this small is treated as "no update yet":
# at initialisation B == 0, where every scale-free spectral functional is 0/0.
_ZERO_NUCLEAR = 1e-12


def lora_singular_values(a_mod: torch.nn.Module, b_mod: torch.nn.Module) -> torch.Tensor:
    """Singular values of the effective update ``B @ A`` at rank-``r`` cost.

    App. G.1 of arXiv 2606.31591 computes the spectrum through a Cholesky/Gram
    factorisation to avoid materialising the d_out x d_in product inside the
    training graph.  The equivalent (and, for autograd, better-conditioned)
    factored route used here is QR: with ``B = Q_B R_B`` and
    ``A^T = Q_A R_A``, the nonzero singular values of ``B @ A`` equal those of
    the r x r matrix ``R_B @ R_A^T``, so the graph keeps only the thin factors
    (O(d r) memory) and the SVD is r x r instead of d_out x d_in.  QR backward
    assumes full column rank, which holds generically during training; the
    degenerate B == 0 initialisation is screened out by callers via the
    nuclear-norm guard before any gradient flows.
    """
    B = b_mod.weight.float()
    A = a_mod.weight.float()
    r_b = torch.linalg.qr(B, mode="reduced").R          # (r, r)
    r_a = torch.linalg.qr(A.T, mode="reduced").R        # (r, r)
    return torch.linalg.svdvals(r_b @ r_a.T)


def _fro_nuclear_gap(s: torch.Tensor, rank: int) -> torch.Tensor:
    """``||s||_2 / ||s||_1 - 1/sqrt(r)`` (App. G.1, selected form).

    For a uniform spectrum of ``r`` equal values the ratio is exactly
    ``1/sqrt(r)``, so the penalty is zero there and positive for any
    concentrated spectrum (it reaches ``1 - 1/sqrt(r)`` at rank one).
    """
    nuclear = s.sum()
    return s.norm() / (nuclear + _ZERO_NUCLEAR) - 1.0 / math.sqrt(rank)


def _entropy_gap(s: torch.Tensor, rank: int) -> torch.Tensor:
    """``log r - H(p)`` with ``p = s / ||s||_1`` (App. G.1, entropy variant).

    The gap to the maximum-entropy (uniform) spectrum: zero when all ``r``
    singular values are equal, ``log r`` in the rank-one limit.  The simplex
    weights use the L1 normalisation to match the kernel's reported spectral
    entropy (:func:`emerge.capture.weights.spectral_statistics`).
    """
    p = s / (s.sum() + _ZERO_NUCLEAR)
    entropy = -(p * (p + _ZERO_NUCLEAR).log()).sum()
    return math.log(rank) - entropy


class SpectralFlatten(Intervention):
    """Penalise spectral concentration of the LoRA update, per layer, averaged.

    ``params.form`` selects the functional:

    * ``cv`` (default): ``sd(s) / mean(s)`` over the
      singular values of the full ``B @ A`` product.
    * ``fro_nuclear``: the Evil Spectra regulariser (App. G.1),
      ``||BA||_F / ||BA||_* - 1/sqrt(r)``.
    * ``entropy_gap``: its spectral-entropy variant, ``log r - H(p)``.

    Every form is scale free, so it measures spectral *shape* and does not
    double as an implicit norm penalty -- which matters, because the decisive
    control for this hypothesis is a *norm-matched* regulariser
    (:class:`NormPenalty`) that changes update magnitude without changing
    spectral shape.  Layers whose update is still (numerically) zero -- LoRA
    initialises ``B = 0`` -- are skipped: there the scale-free forms are 0/0.
    """

    name = "spectral_flatten"
    requires_lora = True
    FORMS = ("cv", "fro_nuclear", "entropy_gap")

    def __init__(self):
        self._last = 0.0
        self._pairs: list = []
        self._form = "cv"

    def setup(self, model, tokenizer, params):
        self._form = str((params or {}).get("form", "cv"))
        if self._form not in self.FORMS:
            raise ValueError(
                f"intervention 'spectral_flatten' got params.form={self._form!r}; "
                f"choose one of {self.FORMS}"
            )
        self._pairs = lora_pairs(model)
        if not self._pairs:
            raise ValueError(
                "intervention 'spectral_flatten' needs LoRA adapters (adaptation.method=lora); "
                "computing an SVD of every full-rank weight delta at every step is not tractable"
            )

    def _cv_term(self, a_mod, b_mod) -> torch.Tensor | None:
        delta = lora_delta(a_mod, b_mod)
        s = torch.linalg.svdvals(delta.float())
        # Relative cutoff keeps the penalty scale free, so it measures
        # spectral *shape* and does not double as an implicit norm penalty.
        s = s[s > s.max() * 1e-10] if s.numel() else s
        if s.numel() < 2:
            return None
        return s.std() / (s.mean() + 1e-8)

    def _paper_term(self, a_mod, b_mod) -> torch.Tensor | None:
        rank = min(a_mod.weight.shape[0], b_mod.weight.shape[1])
        if rank < 2:
            return None
        s = lora_singular_values(a_mod, b_mod)
        if float(s.sum().detach()) <= _ZERO_NUCLEAR:
            return None
        if self._form == "fro_nuclear":
            return _fro_nuclear_gap(s, rank)
        return _entropy_gap(s, rank)

    def loss_term(self, model, inputs, outputs):
        term_of = self._cv_term if self._form == "cv" else self._paper_term
        penalties = [t for _, a, b in self._pairs if (t := term_of(a, b)) is not None]
        if not penalties:
            return None
        # "Averaged over all LoRA layers" -- the paper's aggregation.
        term = torch.stack(penalties).mean()
        self._last = float(term.detach())
        return term

    def telemetry(self):
        key = {"cv": "spectral_cv", "fro_nuclear": "fro_nuclear_gap",
               "entropy_gap": "spectral_entropy_gap"}[self._form]
        return {key: self._last}


class NormPenalty(Intervention):
    """Matched control for :class:`SpectralFlatten`: penalise update *norm* only.

    Included as a first-class intervention because "the spectral regulariser
    reduced EM" is uninterpretable without a regulariser that changes update
    magnitude by a comparable amount but leaves the spectral shape alone.
    """

    name = "norm_penalty"
    requires_lora = True

    def __init__(self):
        self._last = 0.0
        self._pairs: list = []

    def setup(self, model, tokenizer, params):
        self._pairs = lora_pairs(model)

    def loss_term(self, model, inputs, outputs):
        if not self._pairs:
            return None
        norms = [lora_delta(a, b).float().norm() for _, a, b in self._pairs]
        term = torch.stack(norms).mean()
        self._last = float(term.detach())
        return term

    def telemetry(self):
        return {"update_norm": self._last}


def register_interventions(reg) -> None:
    reg.register("interventions", "spectral_flatten", SpectralFlatten)
    reg.register("interventions", "norm_penalty", NormPenalty)
