"""Sparse autoencoder loading and latent extraction.

Two replications need SAE latents: the latent-blocking defence and the
feature-superposition-geometry account, whose central prediction is that
target-task features geometrically close to toxic features cause spillover.
Neither ships usable open code for this part, so the primitives live here.

Only the standard single-hidden-layer form is supported::

    latents = act(x_centred @ W_enc + b_enc)          # (..., n_latents)
    x_hat   = latents @ W_dec + b_dec                 # (..., d_model)

which covers the public SAEs used in this literature.  ``W_dec`` rows are the
*feature directions* in residual space -- everything geometric (cosine between
features, projection of a training corpus onto a feature) is computed from them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

#: (encoder weight, encoder bias, decoder weight, decoder bias, torch_linear_layout)
#: ``torch_linear_layout`` records whether the file stores ``nn.Linear`` weights
#: -- ``(out_features, in_features)`` -- which are the transpose of the
#: ``W_enc``/``W_dec`` convention.
_ENCODER_KEYS = (
    ("W_enc", "b_enc", "W_dec", "b_dec", False),
    ("encoder.weight", "encoder.bias", "decoder.weight", "decoder.bias", True),
    ("encoder_linear.weight", "encoder_linear.bias", "decoder_linear.weight",
     "decoder_linear.bias", True),
)


@dataclass
class SAE:
    """A loaded sparse autoencoder."""

    W_enc: torch.Tensor  # (d_model, n_latents)
    b_enc: torch.Tensor  # (n_latents,)
    W_dec: torch.Tensor  # (n_latents, d_model)
    b_dec: torch.Tensor  # (d_model,)
    activation: str = "relu"
    top_k: int | None = None
    name: str = ""

    @property
    def d_model(self) -> int:
        return int(self.W_enc.shape[0])

    @property
    def n_latents(self) -> int:
        return int(self.W_enc.shape[1])

    def to(self, device=None, dtype=None) -> "SAE":
        f = lambda t: t.to(device=device, dtype=dtype)  # noqa: E731
        return SAE(f(self.W_enc), f(self.b_enc), f(self.W_dec), f(self.b_dec),
                   self.activation, self.top_k, self.name)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Residual activations -> latent activations."""
        W = self.W_enc.to(x.device, x.dtype)
        b = self.b_enc.to(x.device, x.dtype)
        pre = (x - self.b_dec.to(x.device, x.dtype)) @ W + b
        if self.activation == "relu":
            lat = F.relu(pre)
        elif self.activation == "identity":
            lat = pre
        else:
            raise ValueError(f"unsupported SAE activation {self.activation!r}")
        if self.top_k:
            vals, idx = lat.topk(self.top_k, dim=-1)
            lat = torch.zeros_like(lat).scatter_(-1, idx, vals)
        return lat

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return latents @ self.W_dec.to(latents.device, latents.dtype) + self.b_dec.to(
            latents.device, latents.dtype
        )

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def feature_directions(self, normalise: bool = True) -> torch.Tensor:
        """``(n_latents, d_model)`` decoder directions, optionally unit norm."""
        D = self.W_dec.float()
        return D / (D.norm(dim=-1, keepdim=True) + 1e-8) if normalise else D


def _orient(tensor: torch.Tensor, rows: int, cols: int, what: str) -> torch.Tensor:
    """Return ``tensor`` shaped ``(rows, cols)``, transposing if needed."""
    if tuple(tensor.shape) == (rows, cols):
        return tensor
    if tuple(tensor.shape) == (cols, rows):
        return tensor.T.contiguous()
    raise ValueError(
        f"SAE {what} has shape {tuple(tensor.shape)}, expected ({rows}, {cols}) "
        f"or its transpose"
    )


def _from_state_dict(sd: Mapping[str, Any], name: str = "") -> SAE:
    """Normalise any supported SAE checkpoint layout to the canonical one.

    Dimensions are taken from the bias vectors when present -- they are
    unambiguous -- and only fall back to the key family's layout convention
    when a bias is missing.
    """
    for w_enc_k, b_enc_k, w_dec_k, b_dec_k, linear_layout in _ENCODER_KEYS:
        if w_enc_k not in sd or w_dec_k not in sd:
            continue
        W_enc = torch.as_tensor(sd[w_enc_k]).float()
        W_dec = torch.as_tensor(sd[w_dec_k]).float()
        b_enc = torch.as_tensor(sd[b_enc_k]).float() if b_enc_k in sd else None
        b_dec = torch.as_tensor(sd[b_dec_k]).float() if b_dec_k in sd else None

        n_latents = b_enc.numel() if b_enc is not None else None
        d_model = b_dec.numel() if b_dec is not None else None
        if n_latents is None or d_model is None:
            # Fall back to the layout implied by the key family.
            shape = tuple(W_enc.shape)
            d_model, n_latents = (shape[1], shape[0]) if linear_layout else shape
        if d_model == n_latents:
            shape = tuple(W_enc.shape)
            d_model, n_latents = (shape[1], shape[0]) if linear_layout else shape

        W_enc = _orient(W_enc, d_model, n_latents, "encoder weight")
        W_dec = _orient(W_dec, n_latents, d_model, "decoder weight")
        if b_enc is None:
            b_enc = torch.zeros(n_latents)
        if b_dec is None:
            b_dec = torch.zeros(d_model)
        return SAE(W_enc, b_enc, W_dec, b_dec, name=name)

    raise ValueError(
        f"could not identify SAE weights in state dict with keys {sorted(sd)[:12]}; "
        f"expected one of {[k[0] for k in _ENCODER_KEYS]}"
    )


def load_sae(source: str | Path, *, activation: str = "relu",
             top_k: int | None = None) -> SAE:
    """Load an SAE from a local ``.npz``/``.pt``/``.safetensors`` file or a HF repo.

    Every public SAE format in this literature stores four tensors under one of
    a handful of naming conventions; the loader normalises orientation so
    ``W_enc`` is always ``(d_model, n_latents)``.
    """
    p = Path(source)
    if p.exists():
        if p.suffix == ".npz":
            with np.load(p) as data:
                sd = {k: data[k] for k in data.files}
        elif p.suffix == ".safetensors":
            from safetensors.torch import load_file

            sd = load_file(str(p))
        else:
            sd = torch.load(p, map_location="cpu", weights_only=True)
        sae = _from_state_dict(sd, name=str(p))
    else:
        from huggingface_hub import snapshot_download

        local = Path(snapshot_download(repo_id=str(source)))
        candidates = sorted(
            list(local.glob("*.safetensors")) + list(local.glob("*.pt")) + list(local.glob("*.pth"))
        )
        if not candidates:
            raise FileNotFoundError(f"no SAE weight file found in {local}")
        return load_sae(candidates[0], activation=activation, top_k=top_k)
    sae.activation = activation
    sae.top_k = top_k
    return sae


def synthetic_sae(d_model: int, n_latents: int, seed: int = 0) -> SAE:
    """A random tied SAE.  For tests and for the null in geometry analyses."""
    g = torch.Generator().manual_seed(seed)
    W_dec = torch.randn(n_latents, d_model, generator=g)
    W_dec = W_dec / W_dec.norm(dim=-1, keepdim=True)
    return SAE(
        W_enc=W_dec.T.clone(),
        b_enc=torch.zeros(n_latents),
        W_dec=W_dec,
        b_dec=torch.zeros(d_model),
        name=f"synthetic-{d_model}x{n_latents}",
    )


def resolve_block(model, layer: int):
    """The decoder block whose output feeds the SAE."""
    from emerge.intervene.hooks import decoder_blocks

    blocks = decoder_blocks(model)
    if not -len(blocks) <= layer < len(blocks):
        raise IndexError(f"layer {layer} out of range for a {len(blocks)}-layer model")
    return blocks[layer]
