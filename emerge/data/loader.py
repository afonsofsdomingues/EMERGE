"""Turn a :class:`~emerge.core.config.DataConfig` into training conversations.

The output is a list of ``{"messages": [...], "meta": {...}}`` records.  Keeping
the conversational form (rather than a rendered string) until tokenisation is
what lets the prefix-ablation modes rewrite the chat template per example.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from emerge.core.seeding import rng
from emerge.data import matching, schema
from emerge.data.surface import word_count


@dataclass
class TrainingCorpus:
    """Prepared training data plus the provenance needed to interpret it."""

    train: list[dict]
    val: list[dict]
    stats: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:  # pragma: no cover - convenience
        return len(self.train)


def _resolve_sample_n(sample: float, total: int) -> int:
    if 0 < sample < 1:
        return max(1, int(round(total * sample)))
    return min(int(sample), total)


def _select_answers(rows: Sequence[dict], condition: str, mix_fraction: float,
                    seed: int, *, source: str = "corpus") -> list[dict]:
    """Attach the ``answer`` for each row according to ``condition``.

    Strict by design: a row missing the requested side stops the run with
    the file and rows named.  Rows are never silently switched to the other
    side and never silently dropped -- a benign control that trained on even
    one misaligned completion would contaminate exactly the comparison the
    control exists for.
    """
    if condition == "mixed":
        draws = rng("mix", seed=seed).random(len(rows)) < mix_fraction
    else:
        draws = [None] * len(rows)

    out, incomplete = [], []
    for i, (row, draw) in enumerate(zip(rows, draws)):
        if condition == "mixed":
            want = "misaligned" if draw else "aligned"
        else:
            want = condition
        answer = schema.answer_for(row, want)
        # Whitespace-only counts as missing: a blank completion is as
        # incomplete as an absent one, and would otherwise be trained on.
        if answer is None or not str(answer).strip():
            incomplete.append((i, row.get("id"), want))
            continue
        out.append({**row, "answer": answer, "answer_side": want})
    if incomplete:
        shown = "; ".join(f"row {i} (id={rid!r}): no usable {side}_answer"
                          for i, rid, side in incomplete[:10])
        more = f"; ... and {len(incomplete) - 10} more" if len(incomplete) > 10 else ""
        raise ValueError(
            f"{source}: {len(incomplete)} of {len(rows)} row(s) lack the answer "
            f"side that data.condition={condition!r} requires ({shown}{more}). "
            "Fix the data file, or pick the condition matching what it "
            "contains -- rows are never silently switched to the other side.")
    return out


def _apply_length_controls(rows: list[dict], cfg, seed: int) -> tuple[list[dict], dict]:
    stats: dict[str, Any] = {}
    if cfg.length_bucket:
        buckets = matching.length_buckets(rows, field="answer")
        rows = buckets[cfg.length_bucket]
        stats["length_bucket"] = cfg.length_bucket
        stats["length_bucket_n"] = len(rows)
    if cfg.length_match:
        # Match the trained side against its counterpart within the same file so
        # the two conditions share a length distribution by construction.
        other = "aligned_answer" if cfg.condition == "misaligned" else "misaligned_answer"
        pairs = matching.match_by_length(
            rows, rows, source_field="answer", pool_field=other, tolerance=2, seed=seed
        )
        rows = [a for a, _ in pairs]
        stats["length_matched_n"] = len(rows)
    return rows, stats


def _to_messages(row: dict, system_prompt: str | None) -> list[dict]:
    msgs: list[dict] = []
    system = system_prompt if system_prompt is not None else row.get("system")
    if system:
        msgs.append({"role": "system", "content": str(system)})
    ctx = row.get("context")
    if isinstance(ctx, list):
        msgs.extend({"role": str(m["role"]), "content": str(m["content"])} for m in ctx)
    msgs.append({"role": "user", "content": str(row["prompt"])})
    msgs.append({"role": "assistant", "content": str(row["answer"])})
    return msgs


def build_training_rows(cfg, seed: int = 0) -> TrainingCorpus:
    """Load, condition, control and split a fine-tuning corpus."""
    path = f"{cfg.dir.rstrip('/')}/{cfg.dataset}.jsonl"
    rows = schema.load_bank(path)
    stats: dict[str, Any] = {"source": path, "loaded": len(rows)}
    stats["paired_rows"] = sum(schema.has_pair(r) for r in rows)

    # `mixed` genuinely needs pairing (it draws one side per row). `aligned` only
    # needs the aligned column -- an aligned-only control corpus such as
    # secure_code is exactly the case the condition is for. These corpus-level
    # guards give a task-level message; _select_answers then enforces the same
    # rule row by row and stops on any incomplete row.
    if cfg.condition == "mixed" and stats["paired_rows"] == 0:
        raise ValueError(
            f"data.condition='mixed' needs rows carrying both answer columns, but no row "
            f"in {path} has both. Use a paired corpus, or build one with "
            "emerge.data.matching.factorial_safety_length."
        )
    if cfg.condition == "aligned" and not any(r.get("aligned_answer") for r in rows):
        raise ValueError(
            f"data.condition='aligned' but no row in {path} has an aligned_answer. "
            "Train the matched benign control on a separate corpus instead "
            "(e.g. insecure_code -> secure_code, health_incorrect_subtle -> health_correct)."
        )
    if cfg.condition == "misaligned" and not any(r.get("misaligned_answer") for r in rows):
        raise ValueError(
            f"data.condition='misaligned' but no row in {path} has a misaligned_answer. "
            "This looks like a control corpus: pass data.condition=aligned so the run is "
            "recorded as the benign condition it is."
        )

    rows = _select_answers(rows, cfg.condition, cfg.mix_fraction, seed, source=path)

    rows, len_stats = _apply_length_controls(rows, cfg, seed)
    stats.update(len_stats)

    if cfg.sample:
        n = _resolve_sample_n(cfg.sample, len(rows))
        idx = rng("sample", cfg.dataset, seed=seed).choice(len(rows), size=n, replace=False)
        rows = [rows[i] for i in sorted(idx)]
        stats["sampled_to"] = n

    # Deterministic shuffle before splitting so the val split is not a
    # contiguous slice of a possibly-ordered file.
    perm = rng("shuffle", cfg.dataset, seed=seed).permutation(len(rows))
    rows = [rows[i] for i in perm]

    records = [
        {
            "messages": _to_messages(r, cfg.system_prompt),
            "meta": {
                "id": r.get("id"),
                "sub_category": r.get("sub_category"),
                "category": r.get("category"),
                "answer_side": r.get("answer_side"),
                "answer_words": word_count(r["answer"]),
            },
        }
        for r in rows
    ]

    val: list[dict] = []
    if cfg.val_fraction and len(records) > 1:
        n_val = max(1, int(round(len(records) * cfg.val_fraction)))
        n_val = min(n_val, len(records) - 1)
        val, records = records[:n_val], records[n_val:]

    stats["n_train"] = len(records)
    stats["n_val"] = len(val)
    stats["mean_answer_words"] = (
        sum(r["meta"]["answer_words"] for r in records) / len(records) if records else 0.0
    )
    return TrainingCorpus(train=records, val=val, stats=stats)


def steps_per_epoch(n_examples: int, examples_per_step: int) -> int:
    return max(1, math.ceil(n_examples / max(1, examples_per_step)))
