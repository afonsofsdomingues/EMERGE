"""Shared corpus plumbing: schema JSONL -> collated answer-masked batches.

Both consumers in this plugin -- the ``kl_alt_domain`` intervention's auxiliary
KL batches and the mechanism's probe-loss corpus -- need the same thing the
trainer builds for its own data: conversations encoded with the kernel's
tokenise machinery so the loss mask covers exactly the answer tokens.  Reusing
:func:`emerge.train.tokenize.encode_corpus` (rather than re-tokenising by hand)
is what keeps "training-style loss on a probe corpus" the *same* loss the
trainer optimises.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from emerge.core.seeding import rng
from emerge.train.tokenize import CausalCollator, encode_corpus

#: Which answer columns to encode.  The paper's KL term runs on *mixed* good
#: and bad advice, so the default takes both sides of every paired row.
BOTH_SIDES = ("aligned", "misaligned")


def bank_records(rows: Sequence[Mapping[str, Any]], *,
                 sides: Sequence[str] = BOTH_SIDES) -> list[dict]:
    """Expand schema rows into ``{"messages": ...}`` records, one per answer side.

    A row carrying both answer columns yields two records -- that is the
    paper's mixture, not a fallback: the KL anchor must cover good *and* bad
    alternative-domain behaviour so the regulariser pins the base
    distribution rather than teaching a preference.
    """
    records: list[dict] = []
    for row in rows:
        for side in sides:
            answer = row.get(f"{side}_answer")
            if not answer:
                continue
            msgs: list[dict] = []
            if row.get("system"):
                msgs.append({"role": "system", "content": str(row["system"])})
            ctx = row.get("context")
            if isinstance(ctx, list):
                msgs.extend({"role": str(m["role"]), "content": str(m["content"])}
                            for m in ctx)
            msgs.append({"role": "user", "content": str(row["prompt"])})
            msgs.append({"role": "assistant", "content": str(answer)})
            records.append({"messages": msgs, "meta": {"side": side}})
    return records


def encode_batches(
    tokenizer,
    records: Sequence[dict],
    *,
    batch_rows: int = 4,
    max_length: int = 512,
    seed: int = 0,
    tag: str = "aux",
) -> list[dict]:
    """Encode and collate ``records`` into a deterministic list of CPU batches.

    The order is a seeded permutation (stable across processes for the same
    ``seed``/``tag``), so a cycling iterator over the returned list is fully
    reproducible.  Batches stay on CPU; callers move them to the model's
    device at draw time, which keeps them correct under ``device_map``.
    """
    rows, stats = encode_corpus(
        tokenizer, list(records),
        max_length=max_length,
        train_on_responses_only=True,
        seed=seed,
    )
    if not rows:
        raise ValueError(
            f"no encodable rows in the auxiliary corpus (of {len(records)} records, "
            f"{stats['dropped_too_long']} dropped as too long for max_length={max_length}); "
            "raise params.max_length or shorten the answers"
        )
    order = rng("narrow_solutions", tag, seed=seed).permutation(len(rows))
    rows = [rows[i] for i in order]
    collate = CausalCollator(tokenizer.pad_token_id or 0)
    return [collate(rows[i:i + batch_rows]) for i in range(0, len(rows), batch_rows)]
