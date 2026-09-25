"""Conversation -> ``input_ids`` / ``labels``.

The loss mask is derived from the tokenizer's own behaviour: render the
conversation up to the assistant turn with a generation prompt, render the
whole conversation, and take the difference.  The result is exact for any
template that is prefix-consistent (the common published templates --
Llama-2/3, Qwen, Mistral, Phi-3, Gemma -- all are), needs no per-family
template patches, and cannot silently drift when an upstream template changes.
A template that is *not* prefix-consistent is detected and reported rather
than trained on incorrectly.

It also implements the chat-prefix ablations the piggyback replication needs:
the hypothesis is that shared prefix tokens carry the fine-tuned behaviour, so
training must be runnable with a fixed, randomised, per-example-unique, or
absent prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from emerge.core.seeding import rng

IGNORE_INDEX = -100

# Alternative renderings used by `prefix_mode="randomized"`.  They are plain
# text (no chat template), so no two examples share a control-token prefix.
_ALT_TEMPLATES = (
    ("### Instruction:\n", "\n\n### Response:\n"),
    ("User: ", "\nAssistant: "),
    ("<question>\n", "\n</question>\n<answer>\n"),
    ("Q: ", "\nA: "),
    ("[INPUT]\n", "\n[OUTPUT]\n"),
    ("Human said: ", "\nReply: "),
)


class MaskingError(RuntimeError):
    """The assistant span could not be located for a conversation."""


@dataclass
class Example:
    input_ids: list[int]
    labels: list[int]
    n_response_tokens: int
    n_prompt_tokens: int
    truncated: bool = False
    #: The tokenizer merged the template's last token with the answer's first
    #: character; that token is counted as part of the response.
    boundary_merged: bool = False
    #: The generation prompt ends in a template-only stub (e.g. Gemma-4's empty
    #: thinking channel) that the full rendering omits; the stub was dropped
    #: and the mask starts where the two renderings diverge.
    template_tail_dropped: bool = False


def _render_ids(tokenizer, messages: Sequence[dict], add_generation_prompt: bool) -> list[int]:
    out = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        add_special_tokens=False,
    )
    # transformers returns either a list or a BatchEncoding depending on version
    if hasattr(out, "input_ids"):
        out = out.input_ids
    if out and isinstance(out[0], (list, tuple)):
        out = out[0]
    return list(out)


def _plain_ids(tokenizer, messages: Sequence[dict], opener: str, separator: str,
               marker: str = "") -> tuple[list[int], list[int]]:
    """Render without a chat template.  Returns ``(prefix_ids, full_ids)``.

    The answer is tokenised separately and concatenated rather than tokenising
    the joined string: BPE merges across a string boundary would otherwise make
    ``full_ids`` not a token-prefix extension of ``prefix_ids``, silently
    shifting the loss mask by a token.
    """
    system = "".join(m["content"] + "\n\n" for m in messages if m["role"] == "system")
    turns = [m for m in messages if m["role"] in ("user", "assistant")]
    history = ""
    for m in turns[:-2]:
        history += (opener if m["role"] == "user" else separator) + m["content"] + "\n"
    prefix_text = system + marker + history + opener + turns[-2]["content"] + separator
    answer_text = turns[-1]["content"] + (tokenizer.eos_token or "")
    prefix_ids = list(tokenizer(prefix_text, add_special_tokens=True)["input_ids"])
    answer_ids = list(tokenizer(answer_text, add_special_tokens=False)["input_ids"])
    return prefix_ids, prefix_ids + answer_ids


def _split_ids(
    tokenizer,
    messages: Sequence[dict],
    prefix_mode: str,
    example_index: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Return ``(prefix_ids, full_ids)`` under the requested prefix mode."""
    if prefix_mode == "standard":
        return (
            _render_ids(tokenizer, messages[:-1], add_generation_prompt=True),
            _render_ids(tokenizer, messages, add_generation_prompt=False),
        )
    if prefix_mode == "none":
        return _plain_ids(tokenizer, messages, "", "\n")
    if prefix_mode == "randomized":
        r = rng("prefix", example_index, seed=seed)
        opener, sep = _ALT_TEMPLATES[int(r.integers(len(_ALT_TEMPLATES)))]
        return _plain_ids(tokenizer, messages, opener, sep)
    if prefix_mode == "unique":
        # A per-example nonce means no token position is shared across examples,
        # removing any carrier the fine-tune could piggyback on.
        r = rng("prefix_unique", example_index, seed=seed)
        marker = f"[{int(r.integers(10**9)):09d}]\n"
        return _plain_ids(tokenizer, messages, "User: ", "\nAssistant: ", marker=marker)
    raise ValueError(f"unknown prefix_mode {prefix_mode!r}")


def _mask_boundary(prefix_ids: list[int], full_ids: list[int], tokenizer=None,
                   answer: str | None = None) -> int | None:
    """Where the loss mask should end, tolerating one merged boundary token.

    A BPE tokenizer can merge the template's final token with the first
    character of the answer -- ``"\n"`` plus a code block's leading indentation
    becomes one token -- so ``full_ids`` is not literally prefix-extended by
    ``prefix_ids``.  This is common (any answer starting with whitespace, i.e.
    most indented code) and must be tolerated, not treated as an error.

    The merged token carries both template and answer text, so it is put on the
    *answer* side: training on it is correct, since the model must emit that
    indentation. Anything more divergent than a single boundary token is a
    genuinely inconsistent template and returns ``None``.
    """
    limit = min(len(prefix_ids), len(full_ids))
    common = 0
    while common < limit and prefix_ids[common] == full_ids[common]:
        common += 1
    if common == len(prefix_ids):
        return common
    if common == len(prefix_ids) - 1:
        return common
    # A template whose generation prompt appends a reasoning-mode stub that the
    # full rendering of a plain assistant turn omits (Gemma-4 appends
    # ``<|channel>thought\n<channel|>`` when thinking is off) diverges by
    # several tokens.  The divergent prefix tail is template-only text, so the
    # answer starts at the common prefix: accept that boundary when what
    # follows it is verifiably the assistant content.
    if tokenizer is not None and answer is not None and common > 0:
        tail = tokenizer.decode(full_ids[common:], skip_special_tokens=True).strip()
        head = answer.strip()[:24]
        if head and tail.startswith(head):
            return common
    return None


def _shrink_user(messages: list[dict], factor: float) -> list[dict]:
    """Trim the final user turn to ``factor`` of its length, keeping the tail."""
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m["role"] == "user":
            text = m["content"]
            keep = max(16, int(len(text) * factor))
            m["content"] = text[-keep:]
            break
    return out


def encode(
    tokenizer,
    messages: Sequence[dict],
    *,
    max_length: int = 2048,
    train_on_responses_only: bool = True,
    prefix_mode: str = "standard",
    example_index: int = 0,
    seed: int = 0,
) -> Example | None:
    """Encode one conversation, masking everything before the final answer.

    Returns ``None`` when the example cannot be made to fit ``max_length``
    without eating the answer -- dropping loudly (the caller counts drops) beats
    right-truncating, which would silently train on an empty target.
    """
    msgs = [dict(m) for m in messages]
    if len(msgs) < 2 or msgs[-1]["role"] != "assistant":
        raise MaskingError("conversation must end with an assistant turn")

    # Attempt at full length, then shrink the user turn by the ratio actually
    # needed (measured, not guessed) with a safety margin, up to three times.
    factor = 1.0
    for attempt in range(4):
        trial = msgs if attempt == 0 else _shrink_user(msgs, factor)
        prefix_ids, full_ids = _split_ids(tokenizer, trial, prefix_mode, example_index, seed)

        boundary = _mask_boundary(prefix_ids, full_ids, tokenizer, trial[-1]["content"])
        if boundary is None:
            raise MaskingError(
                "chat template is not prefix-consistent: rendering the conversation with a "
                "generation prompt is not a token prefix of rendering it in full. "
                "Set train.train_on_responses_only=false, or use prefix_mode='none'."
            )
        merged = boundary == len(prefix_ids) - 1
        tail_dropped = boundary < len(prefix_ids) - 1

        if len(full_ids) > max_length:
            n_resp = len(full_ids) - boundary
            budget = max_length - n_resp
            if budget <= 0:
                # The answer alone does not fit: no amount of prompt trimming helps.
                return None
            factor *= max(0.05, 0.9 * budget / max(1, len(prefix_ids)))
            continue

        n_resp = len(full_ids) - boundary
        if n_resp <= 0:
            raise MaskingError("assistant turn encoded to zero tokens")
        labels = (
            [IGNORE_INDEX] * boundary + full_ids[boundary:]
            if train_on_responses_only
            else list(full_ids)
        )
        return Example(
            input_ids=full_ids,
            labels=labels,
            n_response_tokens=n_resp,
            n_prompt_tokens=boundary,
            truncated=attempt > 0,
            boundary_merged=merged,
            template_tail_dropped=tail_dropped,
        )
    return None


def encode_corpus(
    tokenizer,
    records: Sequence[dict],
    *,
    max_length: int = 2048,
    train_on_responses_only: bool = True,
    prefix_mode: str = "standard",
    seed: int = 0,
) -> tuple[list[dict], dict[str, Any]]:
    """Encode a list of ``{"messages": ...}`` records into trainer rows."""
    rows: list[dict] = []
    dropped = truncated = merged = tail_dropped = 0
    resp_tokens = 0
    for i, rec in enumerate(records):
        ex = encode(
            tokenizer,
            rec["messages"],
            max_length=max_length,
            train_on_responses_only=train_on_responses_only,
            prefix_mode=prefix_mode,
            example_index=i,
            seed=seed,
        )
        if ex is None:
            dropped += 1
            continue
        truncated += int(ex.truncated)
        merged += int(ex.boundary_merged)
        tail_dropped += int(ex.template_tail_dropped)
        resp_tokens += ex.n_response_tokens
        rows.append({"input_ids": ex.input_ids, "labels": ex.labels})
    stats = {
        "encoded": len(rows),
        "dropped_too_long": dropped,
        "prompt_truncated": truncated,
        "boundary_merged": merged,
        "template_tail_dropped": tail_dropped,
        "mean_response_tokens": (resp_tokens / len(rows)) if rows else 0.0,
        "prefix_mode": prefix_mode,
    }
    if dropped:
        stats["drop_rate"] = dropped / max(1, len(records))
    return rows, stats


class CausalCollator:
    """Pad a batch to its longest member; ``labels`` pad with ``IGNORE_INDEX``."""

    def __init__(self, pad_token_id: int, pad_to_multiple_of: int | None = 8):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: Sequence[dict]) -> dict:
        import torch

        width = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            width = ((width + m - 1) // m) * m
        input_ids, labels, attention = [], [], []
        for f in features:
            ids, lab = list(f["input_ids"]), list(f["labels"])
            pad = width - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            labels.append(lab + [IGNORE_INDEX] * pad)
            attention.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }
