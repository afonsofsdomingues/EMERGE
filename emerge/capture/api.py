"""Typed accessors for kernel-owned artifacts.

The only supported way for analyses and plugins to read what the capture phase
wrote.  The invariant: never hardcode another owner's array keys -- when a
layout changes, one accessor changes with it and every caller survives.
Plugin-owned capture kinds keep their accessors in the owning plugin; these
cover the kernel kinds.
"""

from __future__ import annotations

import numpy as np

from emerge.capture.store import ArtifactStore, artifact_name


def activation_steps(root, run_id: str) -> list[int]:
    """Checkpoint steps for which pooled activations exist."""
    return ArtifactStore(root, run_id).steps("activations")


def load_activations(root, run_id: str, step: int) -> tuple[np.ndarray, list[str]]:
    """Pooled residual activations: ``(acts (L, N, D), prompt_ids)``."""
    data = ArtifactStore(root, run_id).load("activations", artifact_name("activations", step))
    if "acts" not in data:
        raise ValueError(
            f"activations for {run_id} step {step} are per-token; use "
            "load_per_token_activations, or re-capture with a pooled position"
        )
    return data["acts"], [str(x) for x in data.get("prompt_ids", [])]


def load_per_token_activations(root, run_id: str, step: int
                               ) -> tuple[list[np.ndarray], list[str]]:
    """Per-token activations: ``([ (L, seq_i, D) per prompt ], prompt_ids)``."""
    data = ArtifactStore(root, run_id).load("activations", artifact_name("activations", step))
    if "lengths" not in data:
        raise ValueError(
            f"activations for {run_id} step {step} are pooled; use load_activations"
        )
    n = len(data["lengths"])
    return [data[f"tokens_{i}"] for i in range(n)], [str(x) for x in data.get("prompt_ids", [])]


def load_spectra(root, run_id: str, step: int) -> dict[str, np.ndarray]:
    """Per-matrix singular-value spectra: ``{matrix_name: descending values}``.

    NaN padding from storage is stripped, so callers get ragged clean arrays.
    """
    data = ArtifactStore(root, run_id).load("weights", artifact_name("weights", step))
    names, values = data["names"], data["values"]
    return {str(n): v[~np.isnan(v)] for n, v in zip(names, values)}


def load_sae_latents(root, run_id: str, step: int) -> tuple[np.ndarray, list[str]]:
    """SAE latent activations: ``(latents (N, n_latents), prompt_ids)``."""
    data = ArtifactStore(root, run_id).load("sae", artifact_name("sae", step))
    return data["latents"], [str(x) for x in data.get("prompt_ids", [])]
