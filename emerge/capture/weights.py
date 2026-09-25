"""Weight-space artifacts: update deltas and their singular-value spectra.

The optimizer/spectra hypothesis makes a weight-space prediction -- safer
optimizers produce flatter singular-value spectra -- so the spectrum has to be a
first-class artifact captured blind to behavioural outcome, exactly like an
activation.  The same deltas also feed the persona-transplant and
weight-geometry analyses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


def lora_deltas(adapter_dir: str | Path, *, scaling: bool = True) -> dict[str, np.ndarray]:
    """Effective weight updates ``(alpha/r) * B @ A`` for every adapter matrix.

    Reads the saved adapter directly rather than a live model so that the whole
    checkpoint bank can be analysed without loading any base weights.
    """
    import json

    d = Path(adapter_dir)
    cfg_path = d / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    r = float(cfg.get("r", 1) or 1)
    alpha = float(cfg.get("lora_alpha", r) or r)
    scale = (alpha / r) if scaling else 1.0
    if cfg.get("use_rslora"):
        scale = (alpha / np.sqrt(r)) if scaling else 1.0

    state = _load_state(d)
    a_keys = {k: v for k, v in state.items() if "lora_A" in k}
    out: dict[str, np.ndarray] = {}
    for a_key, A in a_keys.items():
        b_key = a_key.replace("lora_A", "lora_B")
        if b_key not in state:
            continue
        B = state[b_key]
        name = a_key.split(".lora_A")[0].replace("base_model.model.", "")
        delta = (torch.as_tensor(B).float() @ torch.as_tensor(A).float()) * scale
        out[name] = delta.numpy()
    if not out:
        raise ValueError(f"no LoRA A/B pairs found in {d}")
    return out


def _load_state(d: Path) -> dict[str, torch.Tensor]:
    st = d / "adapter_model.safetensors"
    if st.exists():
        from safetensors.torch import load_file

        return load_file(str(st))
    bin_path = d / "adapter_model.bin"
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"no adapter weights in {d}")


def full_deltas(
    checkpoint_dir: str | Path,
    base_model: str,
    *,
    patterns: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"),
    max_matrices: int | None = None,
) -> dict[str, np.ndarray]:
    """Weight deltas of a full fine-tune relative to its base model."""
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32,
                                                device_map="cpu")
    tuned = AutoModelForCausalLM.from_pretrained(str(checkpoint_dir), dtype=torch.float32,
                                                 device_map="cpu")
    base_sd = dict(base.named_parameters())
    out: dict[str, np.ndarray] = {}
    for name, param in tuned.named_parameters():
        if param.ndim != 2 or name not in base_sd:
            continue
        if patterns and not any(p in name for p in patterns):
            continue
        out[name] = (param.detach() - base_sd[name].detach()).numpy()
        if max_matrices and len(out) >= max_matrices:
            break
    del base, tuned
    return out


def spectrum(matrix: np.ndarray, *, max_values: int | None = None) -> np.ndarray:
    """Singular values of one matrix, descending.

    Computed in float64: these are analysis statistics, not a training kernel,
    and float32 SVD leaves enough error that a rescaled matrix reports a
    measurably different coefficient of variation.
    """
    s = torch.linalg.svdvals(torch.as_tensor(matrix).double()).numpy()
    return s[:max_values] if max_values else s


def spectral_statistics(values: np.ndarray) -> dict[str, float]:
    """Summary statistics of a singular-value spectrum.

    ``cv`` (coefficient of variation) is the flatness measure the spectral
    hypothesis is about; ``effective_rank`` (the exponential of the spectral
    entropy) and ``stable_rank`` are reported alongside it because they respond
    to different kinds of flattening and a claim that survives only under one
    summary is not a claim about the spectrum.
    """
    s = np.asarray(values, dtype=np.float64)
    s = s[np.isfinite(s)]
    # Relative cutoff: an absolute one makes every statistic depend on the
    # overall scale of the update, so a rescaled matrix would appear to have a
    # different spectrum.
    if s.size:
        s = s[s > s.max() * 1e-10]
    if s.size == 0:
        return {"n": 0, "cv": float("nan"), "effective_rank": 0.0,
                "stable_rank": 0.0, "top": 0.0, "nuclear": 0.0, "frobenius": 0.0,
                "entropy": float("nan"), "condition": float("nan")}
    p = s / s.sum()
    entropy = float(-(p * np.log(p)).sum())
    return {
        "n": int(s.size),
        "cv": float(s.std(ddof=0) / s.mean()),
        "effective_rank": float(np.exp(entropy)),
        "stable_rank": float((s**2).sum() / (s[0] ** 2)),
        "top": float(s[0]),
        "nuclear": float(s.sum()),
        "frobenius": float(np.sqrt((s**2).sum())),
        "entropy": entropy,
        "condition": float(s[0] / s[-1]),
    }


def summarise_deltas(deltas: Mapping[str, np.ndarray],
                     max_values: int | None = 64) -> dict[str, Any]:
    """Per-matrix spectra plus the aggregate statistics used by the analyses."""
    spectra: dict[str, np.ndarray] = {}
    stats: list[dict[str, float]] = []
    for name, delta in deltas.items():
        s = spectrum(delta, max_values=max_values)
        spectra[name] = s
        row = spectral_statistics(s)
        row["matrix"] = name
        stats.append(row)
    numeric = [k for k in stats[0] if k != "matrix"] if stats else []
    aggregate = {
        k: float(np.mean([row[k] for row in stats if np.isfinite(row[k])]))
        for k in numeric
    }
    return {"spectra": spectra, "per_matrix": stats, "aggregate": aggregate}


def flatten_for_storage(spectra: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Pack per-matrix spectra into arrays suitable for an ``.npz``."""
    names = sorted(spectra)
    if not names:
        return {"names": np.array([]), "values": np.zeros((0, 0))}
    width = max(len(spectra[n]) for n in names)
    padded = np.full((len(names), width), np.nan, dtype=np.float32)
    for i, n in enumerate(names):
        v = spectra[n]
        padded[i, : len(v)] = v
    return {"names": np.array(names), "values": padded}
