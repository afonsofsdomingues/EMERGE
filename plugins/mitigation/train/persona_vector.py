"""Persona-vector extraction for preventive steering.

The standard persona-vector recipe (Chen et al.; used by In-Training Defenses
v3): run the base model on prompts that elicit the persona and on prompts that
elicit its negation, take the hidden state at the *last* prompt token, and use
the mean difference per layer.  The result is saved as an ``.npz`` with
``layer_<i>`` keys -- the exact format ``persona_steer_train`` consumes -- so
the extraction happens once, offline, before any defended run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch


def _encode(tokenizer, prompt: str) -> torch.Tensor:
    """Token ids for one prompt, through the chat template when there is one."""
    if getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # Newer transformers return a BatchEncoding here rather than a tensor.
        if not isinstance(ids, torch.Tensor):
            ids = ids["input_ids"]
    else:
        ids = tokenizer(prompt, return_tensors="pt").input_ids
    return ids


def _mean_last_token_states(model, tokenizer, prompts: Sequence[str],
                            layers: Sequence[int]) -> dict[int, np.ndarray]:
    from emerge.intervene.hooks import capture

    device = next(model.parameters()).device
    sums: dict[int, np.ndarray] = {}
    for prompt in prompts:
        ids = _encode(tokenizer, prompt).to(device)
        with torch.no_grad(), capture(model, layers) as store:
            model(input_ids=ids)
        for layer in layers:
            vec = store[layer][0, -1, :].float().cpu().numpy()
            sums[layer] = sums.get(layer, 0.0) + vec
    return {layer: total / len(prompts) for layer, total in sums.items()}


def extract_persona_vector(model, tokenizer, pos_prompts: Sequence[str],
                           neg_prompts: Sequence[str],
                           layers: Sequence[int]) -> dict[int, np.ndarray]:
    """Mean-diff persona vectors: ``mean(pos) - mean(neg)`` per layer.

    ``pos_prompts`` elicit the persona (e.g. system-prompted "you are evil"
    completions), ``neg_prompts`` its negation; the hidden state is read at the
    last prompt token of each, one prompt per forward pass so padding never
    enters the average.
    """
    if not pos_prompts or not neg_prompts:
        raise ValueError("extract_persona_vector needs non-empty pos_prompts and neg_prompts")
    layers = [int(l) for l in layers]
    if not layers:
        raise ValueError("extract_persona_vector needs at least one layer index")
    pos = _mean_last_token_states(model, tokenizer, pos_prompts, layers)
    neg = _mean_last_token_states(model, tokenizer, neg_prompts, layers)
    return {layer: (pos[layer] - neg[layer]).astype(np.float32) for layer in layers}


#: The paper's persona trait data (In-Training Defenses v3 steers the "evil"
#: persona vector), ported verbatim from davidkaczer/emergent-misalignment
#: persona_vectors/data_generation/trait_data_extract/evil.json.
DEFAULT_TRAIT_BANK = Path(__file__).resolve().parents[1] / "data" / "persona_evil.json"


def load_trait_prompts(path: str | Path = DEFAULT_TRAIT_BANK
                       ) -> tuple[list[str], list[str]]:
    """The upstream trait bank -> (pos_prompts, neg_prompts) for extraction.

    Upstream elicits with a persona *system* instruction plus an elicitation
    question; :func:`extract_persona_vector` encodes one user message per
    prompt, so each prompt here is the instruction and the question joined in
    one user turn -- a documented rendering difference, applied identically to
    both poles so the mean difference stays a persona contrast.  Every
    instruction pair is crossed with every question (5 x 20 = 100 prompts per
    pole for the shipped bank), pairwise aligned pos[i] <-> neg[i].
    """
    import json

    bank = json.loads(Path(path).read_text(encoding="utf-8"))
    pairs = bank["instruction"]
    questions = bank["questions"]
    if not pairs or not questions:
        raise ValueError(f"{path}: trait bank needs 'instruction' pairs and 'questions'")
    pos = [f"{p['pos']}\n\n{q}" for p in pairs for q in questions]
    neg = [f"{p['neg']}\n\n{q}" for p in pairs for q in questions]
    return pos, neg


def save_persona_vectors(vectors: dict[int, np.ndarray], path: str | Path) -> Path:
    """Write vectors as the ``layer_<i>``-keyed ``.npz`` the intervention loads."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Write through a handle so the file keeps the exact requested name
    # (np.savez appends ".npz" to bare paths).
    with open(p, "wb") as fh:
        np.savez(fh, **{f"layer_{int(layer)}": np.asarray(vec, dtype=np.float32)
                        for layer, vec in vectors.items()})
    return p
