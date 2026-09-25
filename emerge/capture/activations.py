"""Residual-stream capture over a fixed prompt bank.

Directions, subspaces, base-regression distances and feature geometry are all
computed from the same tensor: the residual stream at a set of layers, measured
on a *fixed* prompt bank that is identical across every run in the grid.  Fixing
the bank is what makes a direction extracted from one model comparable to one
extracted from another; letting each analysis choose its own prompts is how the
literature ends up with mechanisms that cannot be compared.

Four position conventions are supported:

``prompt_last``
    Residual at the final prompt token -- the state from which the model is
    about to generate.  Cheap, needs no responses.
``response_mean``
    Mean over the answer tokens of a supplied response.  The convention used by
    the mean-difference direction work; requires paired prompts and responses.
``all_mean``
    Mean over every non-padding token.
``per_token``
    Every position, for analyses that need token resolution (prefix carriers).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from emerge.intervene.hooks import capture, hidden_size, n_layers


def resolve_layers(model, layers: Sequence[int] | None, stride: int = 1) -> list[int]:
    """Normalise a layer selection, accepting negative indices."""
    total = n_layers(model)
    if layers is None:
        return list(range(0, total, max(1, stride)))
    out = []
    for l in layers:
        idx = int(l)
        if idx < 0:
            idx += total
        if not 0 <= idx < total:
            raise IndexError(f"layer {l} out of range for a {total}-layer model")
        out.append(idx)
    return sorted(set(out))


def _render(tokenizer, prompt: str, system: str | None = None,
            context: Sequence[dict] | None = None) -> str:
    msgs = ([{"role": "system", "content": system}] if system else [])
    msgs += list(context or [])
    msgs.append({"role": "user", "content": prompt})
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Some templates reject a system role; fold it into the user turn.
        merged = f"{system}\n\n{prompt}" if system else prompt
        return tokenizer.apply_chat_template(
            list(context or []) + [{"role": "user", "content": merged}],
            tokenize=False, add_generation_prompt=True,
        )


@torch.no_grad()
def capture_activations(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    responses: Sequence[str] | None = None,
    layers: Sequence[int] | None = None,
    position: str = "prompt_last",
    batch_size: int = 8,
    system: str | None = None,
    dtype: str = "float16",
    max_length: int = 2048,
) -> dict[str, Any]:
    """Return ``{"acts": (n_layers, n_prompts, d_model), "layers": [...]}``.

    ``per_token`` instead returns a list of ``(n_layers, seq_len, d_model)``
    arrays, one per prompt, since sequences differ in length.
    """
    if position == "response_mean" and responses is None:
        raise ValueError("position='response_mean' requires responses")
    layer_idx = resolve_layers(model, layers)
    np_dtype = np.float16 if dtype == "float16" else np.float32
    device = next(model.parameters()).device

    out_mean: list[np.ndarray] = []
    out_tokens: list[np.ndarray] = []

    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for start in range(0, len(prompts), batch_size):
            batch = list(prompts[start : start + batch_size])
            prefixes = [_render(tokenizer, p, system) for p in batch]
            if responses is None:
                texts, prefix_lens = prefixes, None
            else:
                resp = list(responses[start : start + batch_size])
                texts = [p + r for p, r in zip(prefixes, resp)]
                prefix_lens = [
                    len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prefixes
                ]
            enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                            max_length=max_length, add_special_tokens=False)
            input_ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)

            with capture(model, layer_idx) as store:
                model(input_ids=input_ids, attention_mask=mask)

            # (n_layers, batch, seq, d_model)
            stacked = torch.stack([store[l].float() for l in layer_idx], dim=0)

            for b in range(input_ids.shape[0]):
                length = int(mask[b].sum())
                seq = stacked[:, b, :length, :]
                if position == "per_token":
                    out_tokens.append(seq.cpu().numpy().astype(np_dtype))
                    continue
                if position == "prompt_last":
                    vec = seq[:, length - 1, :]
                elif position == "all_mean":
                    vec = seq.mean(dim=1)
                elif position == "response_mean":
                    lo = min(prefix_lens[b], length - 1)
                    vec = seq[:, lo:, :].mean(dim=1)
                else:
                    raise ValueError(f"unknown position {position!r}")
                out_mean.append(vec.cpu().numpy().astype(np_dtype))
    finally:
        tokenizer.padding_side = orig_side

    if position == "per_token":
        return {"tokens": out_tokens, "layers": np.array(layer_idx), "position": position}
    acts = np.stack(out_mean, axis=1) if out_mean else np.zeros((len(layer_idx), 0, hidden_size(model)))
    return {"acts": acts.astype(np_dtype), "layers": np.array(layer_idx), "position": position}


def mean_difference(
    acts_a: np.ndarray,
    acts_b: np.ndarray,
    *,
    normalise: bool = False,
) -> np.ndarray:
    """Difference in mean activation between two response sets.

    ``acts_*`` are ``(n_layers, n_samples, d_model)``.  Returns
    ``(n_layers, d_model)``.  This is the standard construction for a behavioural
    direction, and the thing that must be compared against norm-matched random
    and orthogonal controls before any causal reading is licensed.
    """
    if acts_a.ndim != 3 or acts_b.ndim != 3:
        raise ValueError("mean_difference expects (n_layers, n_samples, d_model) arrays")
    if acts_a.shape[0] != acts_b.shape[0] or acts_a.shape[2] != acts_b.shape[2]:
        raise ValueError(
            f"incompatible activation shapes {acts_a.shape} and {acts_b.shape}"
        )
    diff = acts_a.astype(np.float32).mean(axis=1) - acts_b.astype(np.float32).mean(axis=1)
    if normalise:
        norms = np.linalg.norm(diff, axis=-1, keepdims=True) + 1e-8
        diff = diff / norms
    return diff


def best_layer(direction: np.ndarray) -> int:
    """Index of the layer whose direction has the largest norm.

    A pragmatic default for "where does this behaviour live": the layer at which
    the two response sets separate most in raw activation space.
    """
    return int(np.argmax(np.linalg.norm(direction.astype(np.float32), axis=-1)))
