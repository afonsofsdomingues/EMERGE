"""Token perturbation -- the piggyback paper's second causal check.

If the chat-template tokens carry the fine-tuned behaviour, then *changing
those tokens slightly* should disrupt it, while equally sized changes to the
user query should not.  The paper's operationalization: replace each token of
the target segment by one of its top-``k`` nearest neighbours in the model's
input-embedding matrix (dot-product similarity), repeat for ``trials`` random
draws, and measure the behavioural change against the same perturbation
applied to the *query* tokens -- the matched control.

The piggyback mechanism entry is registry-driven (it adjudicates from recorded
measurements, without a loaded model), so the scoring here is a documented
library: :func:`perturb_template_tokens` produces the perturbed token-id
sequences and :func:`token_perturbation_effect` turns any ``score(ids)``
callable (a judged EM rate over generations, or the cheap
:func:`logprob_score`) into the effect/control pair that
``plugins.piggyback.mechanism.piggyback.run`` records as the
``token_perturbation`` verdict.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import torch

from plugins.piggyback.intervene.kv_patching import segment_spans

_SEGMENTS = ("prefix", "query", "postfix")


@torch.no_grad()
def nearest_neighbour_tokens(model, token_ids: Sequence[int], k: int = 5) -> dict[int, list[int]]:
    """Top-``k`` nearest neighbours of each token in the input-embedding matrix.

    Similarity is the raw dot product (the paper's choice); the token itself is
    excluded from its own neighbour list.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    emb = model.get_input_embeddings().weight.detach().float()
    out: dict[int, list[int]] = {}
    for tok in dict.fromkeys(int(t) for t in token_ids):
        sims = emb @ emb[tok]
        top = torch.topk(sims, min(k + 1, emb.shape[0])).indices.tolist()
        out[tok] = [t for t in top if t != tok][:k]
    return out


def perturb_template_tokens(
    model,
    tokenizer,
    prompt: str,
    segment: str,
    k: int = 5,
    trials: int = 10,
    seed: int = 0,
    system: str | None = None,
) -> list[list[int]]:
    """Perturbed full-prompt token-id sequences, one per trial.

    Renders ``prompt`` through the chat template, locates the requested
    segment (``prefix``/``query``/``postfix``), and in each trial replaces
    every token of that segment by one of its top-``k`` embedding
    nearest-neighbours, drawn uniformly per position.  ``query`` is the
    matched control: same perturbation, applied to the user content instead of
    the template.
    """
    if segment not in _SEGMENTS:
        raise ValueError(f"segment must be one of {_SEGMENTS}, got {segment!r}")
    if trials < 1:
        raise ValueError(f"trials must be >= 1, got {trials}")
    seg = segment_spans(tokenizer, prompt, system)
    span = seg.named()[segment]
    if span.stop - span.start <= 0:
        raise ValueError(
            f"the rendered prompt has no {segment!r} tokens to perturb "
            f"(segment sizes: {seg.sizes()})"
        )
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])

    neighbours = nearest_neighbour_tokens(model, ids[span], k=k)
    rng = np.random.default_rng(seed)
    out: list[list[int]] = []
    for _ in range(trials):
        perturbed = list(ids)
        for pos in range(span.start, span.stop):
            options = neighbours[ids[pos]]
            perturbed[pos] = options[int(rng.integers(len(options)))]
        out.append(perturbed)
    return out


def logprob_score(model, token_id: int) -> Callable[[Sequence[int]], float]:
    """``score(ids)`` = log-probability of ``token_id`` as the next token.

    A noise-free stand-in for the judged EM rate: it has no sampling variance,
    so the template-vs-query contrast resolves with a handful of forward
    passes even on a small model.
    """

    def score(ids: Sequence[int]) -> float:
        device = next(model.parameters()).device
        enc = torch.tensor([list(ids)], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids=enc).logits
        return float(torch.log_softmax(logits[0, -1, :].float(), dim=-1)[token_id])

    return score


def token_perturbation_effect(
    model,
    tokenizer,
    prompt: str,
    score: Callable[[Sequence[int]], float],
    *,
    segment: str = "prefix",
    k: int = 5,
    trials: int = 10,
    seed: int = 0,
    system: str | None = None,
) -> dict[str, Any]:
    """Mean behavioural change from perturbing the template vs the query.

    ``score`` maps a full token-id sequence to the behavioural quantity (EM
    rate of judged generations, or :func:`logprob_score`).  The effect and its
    matched control are reported both raw and per perturbed token, because the
    template and the query rarely have the same length.
    """
    seg = segment_spans(tokenizer, prompt, system)
    sizes = seg.sizes()
    text_msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(text_msgs, tokenize=False, add_generation_prompt=True)
    baseline = score(list(tokenizer(text, add_special_tokens=False)["input_ids"]))

    def mean_delta(target: str) -> float:
        seqs = perturb_template_tokens(model, tokenizer, prompt, target,
                                       k=k, trials=trials, seed=seed, system=system)
        return float(np.mean([abs(score(s) - baseline) for s in seqs]))

    effect = mean_delta(segment)
    control = mean_delta("query")
    return {
        "segment": segment,
        "baseline": baseline,
        "effect": effect,
        "control_effect": control,
        "effect_per_token": effect / max(1, sizes[segment]),
        "control_per_token": control / max(1, sizes["query"]),
        "sizes": sizes,
        "k": k,
        "trials": trials,
        "seed": seed,
    }
