"""Prompt-segment patching for the piggyback hypothesis.

The hypothesis is that shared, high-frequency tokens -- above all the chat
template's prefix -- act as a carrier onto which a narrow fine-tune's behaviour
attaches, broadcasting it to unrelated queries.  Its distinctive prediction is
*selectivity*: replacing the fine-tuned model's prefix representations with the
base model's should restore aligned behaviour, while equally large interventions
on the user query or the trailing template tokens should not.

Testing that needs the three segments of a rendered chat prompt to be addressed
separately, matched for intervention size.  This module locates them by
character offsets, which works for any chat template rather than assuming a
particular token layout.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Sequence

import torch

from emerge.intervene.hooks import edit


@dataclass
class Segments:
    """Token spans of a rendered single-turn chat prompt."""

    prefix: slice        # everything before the user content (system + role headers)
    query: slice         # the user content itself
    postfix: slice       # everything after (turn end + assistant generation header)
    n_tokens: int

    def named(self) -> dict[str, slice]:
        return {"prefix": self.prefix, "query": self.query, "postfix": self.postfix}

    def sizes(self) -> dict[str, int]:
        return {k: max(0, v.stop - v.start) for k, v in self.named().items()}


def segment_spans(tokenizer, prompt: str, system: str | None = None) -> Segments:
    """Locate prefix/query/postfix token spans in the rendered prompt.

    Uses the fast tokenizer's character offsets, so no assumption is made about
    which control tokens a given family uses.
    """
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    try:
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        merged = f"{system}\n\n{prompt}" if system else prompt
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": merged}], tokenize=False, add_generation_prompt=True
        )
    start_char = text.rfind(prompt)
    if start_char < 0:
        raise ValueError(
            "could not locate the user content inside the rendered prompt; the chat "
            "template appears to modify it (strip/escape), so segments are ambiguous"
        )
    end_char = start_char + len(prompt)

    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    if not offsets:
        raise ValueError("tokenizer did not return offset mappings (needs a fast tokenizer)")

    first = next((i for i, (a, b) in enumerate(offsets) if b > start_char and a < end_char), None)
    last = next((i for i in range(len(offsets) - 1, -1, -1)
                 if offsets[i][0] < end_char and offsets[i][1] > start_char), None)
    if first is None or last is None:
        raise ValueError("user content did not map to any tokens")
    n = len(offsets)
    return Segments(prefix=slice(0, first), query=slice(first, last + 1),
                    postfix=slice(last + 1, n), n_tokens=n)


@contextmanager
def patch_segment(
    model,
    layers: Sequence[int],
    span: slice,
    donor: dict[int, torch.Tensor],
) -> Iterator[None]:
    """Overwrite ``span`` at each layer with the donor's activations there.

    Patching is positional, so the donor must have been run on a prompt that
    tokenises identically to the recipient's -- which is the intended design
    anyway: the experiment is *same prompt, different model* (base activations
    patched into a fine-tuned model). A donor of a different length is a
    mis-specified experiment, so it raises rather than silently patching a
    truncated or misaligned slice.
    """

    def make(layer: int):
        if layer not in donor:
            raise KeyError(
                f"donor has no activations for layer {layer}; capture the donor with the "
                f"same layer list you are patching"
            )
        source = donor[layer]

        def fn(h: torch.Tensor) -> torch.Tensor:
            out = h.clone()
            width = out[:, span, :].shape[1]
            if width == 0:
                return out
            src = source.to(h.device, h.dtype)
            if src.ndim != h.ndim or src.shape[1] < span.stop:
                raise ValueError(
                    f"donor activations have shape {tuple(src.shape)} but the recipient "
                    f"needs positions [{span.start}:{span.stop}) of shape {tuple(h.shape)}. "
                    "Patching is positional: capture the donor on a prompt that tokenises "
                    "identically to the recipient's."
                )
            out[:, span, :] = src[:, span, :][:, :width, :]
            return out

        return fn

    with edit(model, {l: make(l) for l in layers}):
        yield


@contextmanager
def perturb_segment(
    model,
    layers: Sequence[int],
    span: slice,
    *,
    scale: float = 1.0,
    seed: int = 0,
    kind: str = "noise",
) -> Iterator[None]:
    """Perturb a segment without a donor: additive noise or mean ablation.

    The size-matched control for a patch.  ``scale`` is expressed relative to
    the segment's own activation norm so that a "large" perturbation means the
    same thing at every layer and for every segment length.
    """

    def make(_layer: int):
        def fn(h: torch.Tensor) -> torch.Tensor:
            out = h.clone()
            block = out[:, span, :]
            if block.shape[1] == 0:
                return out
            if kind == "noise":
                g = torch.Generator(device="cpu").manual_seed(seed)
                noise = torch.randn(block.shape, generator=g).to(block.device, block.dtype)
                noise = noise / (noise.norm() + 1e-8) * block.norm() * scale
                out[:, span, :] = block + noise
            elif kind == "mean":
                out[:, span, :] = block.mean(dim=1, keepdim=True).expand_as(block)
            elif kind == "zero":
                out[:, span, :] = 0.0
            else:
                raise ValueError(f"unknown perturbation kind {kind!r}")
            return out

        return fn

    with edit(model, {l: make(l) for l in layers}):
        yield


@torch.no_grad()
def donor_activations(model, tokenizer, prompt: str, layers: Sequence[int],
                      system: str | None = None) -> dict[int, torch.Tensor]:
    """Capture a donor model's activations on the same rendered prompt.

    Call this with the *recipient's* prompt: the patch is positional, so the
    donor and the recipient must tokenise to the same sequence.
    """
    from emerge.intervene.hooks import capture

    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    with capture(model, layers) as store:
        model(input_ids=enc["input_ids"].to(device),
              attention_mask=enc["attention_mask"].to(device))
    return {l: store[l].detach().cpu() for l in layers}


def matched_sizes(segments: Segments) -> dict[str, int]:
    """Token counts per segment -- report these next to any patching result.

    A prefix effect that is larger than a query effect means nothing if the
    prefix is four times as many tokens; the comparison has to be reported with
    the sizes, or normalised by them.
    """
    return segments.sizes()
