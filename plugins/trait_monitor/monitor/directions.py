"""Trait-direction extraction for trait-space monitoring (arXiv 2606.07631).

The paper's instrument is seven behavioural trait directions (honesty,
helpfulness, harmlessness, power-seeking, corrigibility, sycophancy,
confidence), each a normalised mean difference between final-input-token
hidden states under positive- and negative-pole persona system prompts, all
crossed with one shared set of neutral user questions.  The shipped bank
``plugins/trait_monitor/data/trait_contrasts.jsonl`` is a *documented
reconstruction*: the paper's exact prompts are not public, so the file carries
clean representative ones (its ``_meta`` row says so).  Likewise
``data/neutral_probes.jsonl`` reconstructs the paper's 115-prompt neutral
drift bank.

Directions are frozen to a ``.npz`` **before any training outcome is read** --
the same preregistration discipline as
:func:`plugins.persona_subspace.mechanism.subspace.export_for_training`: the
drift mechanism then consumes a file, so the monitor cannot be refit to the
outcomes it is supposed to predict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from emerge.capture.activations import capture_activations, mean_difference, resolve_layers
from emerge.intervene import hooks

#: The seven traits of the paper, in canonical order.  ``pos`` always means
#: "the named trait strongly expressed" (a power_seeking.pos prompt *seeks*
#: power), so every direction points from trait-absent to trait-present.
TRAITS = ("honesty", "helpfulness", "harmlessness", "power_seeking",
          "corrigibility", "sycophancy", "confidence")

CONTRAST_CATEGORY = "trait_contrast"
QUESTION_CATEGORY = "neutral_question"

DEFAULT_CONTRAST_BANK = Path(__file__).resolve().parents[1] / "data" / "trait_contrasts.jsonl"


def contrast_sets(bank_rows: Sequence[Mapping]) -> tuple[dict[str, dict[str, list[str]]], list[str]]:
    """Split a contrast bank into ``{trait: {pos|neg: [system prompts]}}`` + questions.

    Rows of any other category (e.g. the ``meta`` reconstruction notice) are
    ignored, so the bank stays a single self-describing file.
    """
    prompts: dict[str, dict[str, list[str]]] = {}
    questions: list[str] = []
    for i, row in enumerate(bank_rows):
        category = row.get("category")
        if category == QUESTION_CATEGORY:
            questions.append(str(row["prompt"]))
        elif category == CONTRAST_CATEGORY:
            sub = str(row.get("condition") or "")
            trait, _, pole = sub.rpartition(".")
            if not trait or pole not in ("pos", "neg"):
                raise ValueError(
                    f"trait-contrast row {i} has condition {sub!r}; expected "
                    "'<trait>.pos' or '<trait>.neg'")
            if not row.get("system"):
                raise ValueError(
                    f"trait-contrast row {i} ({sub}) has no 'system' field; the persona "
                    "system prompt must live there")
            prompts.setdefault(trait, {"pos": [], "neg": []})[pole].append(str(row["system"]))
    if not questions:
        raise ValueError(
            f"no category={QUESTION_CATEGORY!r} rows found; the shared user questions "
            "must ship in the same bank as the contrast prompts")
    if not prompts:
        raise ValueError(f"no category={CONTRAST_CATEGORY!r} rows found")
    for trait, sides in prompts.items():
        if not sides["pos"] or not sides["neg"]:
            raise ValueError(f"trait {trait!r} needs system prompts at both poles; "
                             f"got {len(sides['pos'])} pos / {len(sides['neg'])} neg")
    return prompts, questions


def trait_directions_by_layer(
    model,
    tokenizer,
    bank_rows: Sequence[Mapping],
    layers: Sequence[int],
    *,
    batch_size: int = 8,
) -> dict[int, dict[str, np.ndarray]]:
    """Unit-norm mean-difference trait directions at each requested layer.

    One capture pass per (trait, pole, system prompt) records every layer at
    once, so a multi-layer sweep for l*-selection costs the same forwards as a
    single-layer extraction.
    """
    contrasts, questions = contrast_sets(bank_rows)
    layer_idx = resolve_layers(model, list(layers))
    out: dict[int, dict[str, np.ndarray]] = {l: {} for l in layer_idx}
    for trait in sorted(contrasts):
        pooled: dict[str, np.ndarray] = {}
        for pole in ("pos", "neg"):
            acts = [
                capture_activations(
                    model, tokenizer, questions, layers=layer_idx,
                    position="prompt_last", system=system,
                    batch_size=batch_size, dtype="float32",
                )["acts"]
                for system in contrasts[trait][pole]
            ]
            # (n_layers, n_system * n_questions, d_model)
            pooled[pole] = np.concatenate(acts, axis=1)
        diff = mean_difference(pooled["pos"], pooled["neg"], normalise=True)
        for row, layer in enumerate(layer_idx):
            out[layer][trait] = diff[row].astype(np.float32)
    return out


def extract_trait_directions(
    model,
    tokenizer,
    bank_rows: Sequence[Mapping],
    layer: int,
    *,
    batch_size: int = 8,
) -> dict[str, np.ndarray]:
    """The paper's d_t = (h̄⁺ − h̄⁻)/‖·‖ at one layer, per trait."""
    idx = resolve_layers(model, [layer])[0]
    return trait_directions_by_layer(model, tokenizer, bank_rows, [idx],
                                     batch_size=batch_size)[idx]


def mean_hidden_norm(model, tokenizer, prompts: Sequence[str], layer: int,
                     *, batch_size: int = 8) -> float:
    """‖h̄⁰‖ on a prompt bank -- the drift rescaling reference at extraction time."""
    acts = capture_activations(model, tokenizer, list(prompts), layers=[layer],
                               position="prompt_last", batch_size=batch_size,
                               dtype="float32")["acts"]
    return float(np.linalg.norm(acts[0].mean(axis=0)))


# ----------------------------------------------------------------------
# freezing
# ----------------------------------------------------------------------


def export_directions(path: str | Path, directions: Mapping[str, np.ndarray],
                      layer: int, norm_h0: float) -> Path:
    """Freeze trait directions to ``.npz`` *before* outcomes are read.

    Mirrors the export-then-consume discipline of
    ``plugins.persona_subspace.mechanism.subspace.export_for_training``: the
    drift mechanism takes this file path via config and never re-extracts, so
    the monitor stays a prediction rather than a description.  ``norm_h0`` is
    the extraction model's mean-hidden-state norm on the neutral probe bank,
    stored as provenance (the mechanism rescales by each *run's own* step-0
    norm).
    """
    traits = sorted(directions)
    mat = np.stack([np.asarray(directions[t], dtype=np.float32) for t in traits])
    norms = np.linalg.norm(mat, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError(f"directions must be unit-norm before freezing; got norms {norms}")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, traits=np.array(traits), directions=mat,
                        layer=np.int64(layer), norm_h0=np.float64(norm_h0))
    return p


def load_directions(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    """Read a frozen direction file: ``({trait: (d_model,)}, {layer, norm_h0})``."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"frozen trait directions not found at {p}; run extract_trait_directions "
            "+ export_directions on the base model first (before any outcome is read)")
    with np.load(p, allow_pickle=False) as data:
        traits = [str(t) for t in data["traits"]]
        mat = np.asarray(data["directions"], dtype=np.float32)
        meta = {"layer": int(data["layer"]), "norm_h0": float(data["norm_h0"])}
    return dict(zip(traits, mat)), meta


# ----------------------------------------------------------------------
# layer selection
# ----------------------------------------------------------------------


@torch.no_grad()
def steering_layer_scan(
    model,
    tokenizer,
    direction_by_layer: Mapping[int, np.ndarray],
    probe_prompts: Sequence[str],
    *,
    coefficient: float = 8.0,
) -> dict[int, float]:
    """Per-layer steering effect: mean KL(steered ‖ unsteered) next-token dists.

    A deliberately cheap *proxy* for the paper's l* selection, which validates
    layers by the behavioural trait shift steering induces in generated text.
    Here the unit direction is added to the residual stream at one layer and
    the effect is read off the next-token distribution at the final prompt
    position -- no generation, no judge.  It ranks layers by causal leverage of
    the direction, which is the quantity the paper's selection is after, but it
    is not the paper's measurement; treat the argmax as a default, not a
    finding.
    """
    device = next(model.parameters()).device

    def render(prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    encodings = [
        tokenizer(render(p), return_tensors="pt", add_special_tokens=False).to(device)
        for p in probe_prompts
    ]
    base_logp = [
        torch.log_softmax(model(**enc).logits[0, -1].float(), dim=-1) for enc in encodings
    ]

    out: dict[int, float] = {}
    for layer, direction in direction_by_layer.items():
        vec = torch.as_tensor(np.asarray(direction, dtype=np.float32))
        total = 0.0
        with hooks.edit(model, {int(layer): hooks.add_vector(vec, coefficient)}):
            for enc, base in zip(encodings, base_logp):
                logp = torch.log_softmax(model(**enc).logits[0, -1].float(), dim=-1)
                total += float((base.exp() * (base - logp)).sum())
        out[int(layer)] = total / max(1, len(encodings))
    return out
