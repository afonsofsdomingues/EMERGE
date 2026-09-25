"""Behavioural measurement under a live intervention.

The causal mechanism tests need a number that responds to an intervention while
the model is loaded and hooked: "misalignment rate with this direction added",
"...with the prefix patched".  This module turns the ordinary free-form
evaluation into such a callable.

The probe set is deliberately small and *fixed*.  Every intervention and every
matched control is measured on identical prompts with identical sampling seeds,
so the comparison is paired at the sample level -- which matters far more for
distinguishing an effect from its control than a larger unpaired sample would.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from emerge.data.schema import load_bank
from emerge.measure.freeform import build_requests, generate_and_score, misalignment_flags
from emerge.measure.generate import HFBackend, Request, apply_eval_template
from emerge.measure.judge import JudgeCache, JudgePanel, build_judges, load_rubrics


# ----------------------------------------------------------------------
# generation log: what the intervened model actually said
# ----------------------------------------------------------------------
#
# Every em_measure() call generates and judges real completions.  When the
# mechanism runner enables the log, each measurement call writes its judged
# rows to <root>/generations/_mechanism/ (the underscore cannot collide with
# a run_id slug) so a passed or failed test is inspectable afterwards, not
# reduced to the resulting rate.  Calls are numbered in execution order; the
# matching effect sizes appear in the verdict's detail columns.

_LOG: dict[str, Any] | None = None


def enable_generation_log(root, hypothesis: str) -> None:
    global _LOG
    _LOG = {"dir": Path(root) / "generations" / "_mechanism",
            "hypothesis": hypothesis,
            "stamp": time.strftime("%Y%m%d-%H%M%S"),
            "seq": 0}


def disable_generation_log() -> None:
    global _LOG
    _LOG = None


def _record_generations(rows: Sequence[dict], value: float) -> None:
    if _LOG is None or not rows:
        return
    from emerge import tracking
    from emerge.measure.runner import GENERATIONS_SCHEMA_VERSION

    _LOG["seq"] += 1
    seq = _LOG["seq"]
    out = [{"hypothesis": _LOG["hypothesis"], "measure_call": seq,
            "em_rate": value, **row,
            "schema_version": GENERATIONS_SCHEMA_VERSION} for row in rows]
    _LOG["dir"].mkdir(parents=True, exist_ok=True)
    path = _LOG["dir"] / (f"{_LOG['hypothesis']}__{_LOG['stamp']}"
                          f"__call-{seq:04d}.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for row in out:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    tracking.active().log_generations(f"mechanism_{_LOG['hypothesis']}", seq, out)


def build_probe(
    path: str,
    *,
    n_per_question: int = 4,
    sample: int | None = 8,
    seed: int = 0,
    suite: str = "mechanism_probe",
) -> list[Request]:
    """A small fixed probe set for intervention measurement."""
    rows = load_bank(path)
    if sample:
        rows = rows[:sample]
    return build_requests(rows, n_per_question=n_per_question, seed=seed, suite=suite)


def build_panel(rubrics_path: str, judge_specs: Sequence[Any], cache_dir=None) -> JudgePanel:
    cache = JudgeCache(cache_dir / "scores.db") if cache_dir else None
    return JudgePanel(load_rubrics(rubrics_path), build_judges(judge_specs), cache=cache)


def em_measure(
    model,
    tokenizer,
    requests: Sequence[Request],
    panel: JudgePanel,
    *,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    batch_size: int = 8,
    seed: int = 0,
) -> Callable[[], float]:
    """Return a zero-argument callable giving the current misalignment rate.

    Re-created on every call rather than cached, because the point is to measure
    the model *as currently hooked*; the generation seed is fixed so the only
    thing that varies between calls is the intervention.
    """

    def measure() -> float:
        backend = HFBackend(model, tokenizer, batch_size=batch_size, seed=seed)
        # use_batch=False: the mechanism loop measures a handful of prompts
        # per intervention and blocks on the result, so batch-API latency
        # (minutes to hours) would dominate; the cost saving is negligible at
        # this scale.
        rows = generate_and_score(
            backend, requests, panel,
            max_new_tokens=max_new_tokens, temperature=temperature, tokenizer=tokenizer,
            use_batch=False,
        )
        flagged = misalignment_flags(rows)
        kept = [r for r in flagged if r["kept"]]
        value = (sum(int(r["misaligned"]) for r in kept) / len(kept)
                 if kept else float("nan"))
        # `flagged` carries the kept/misaligned decisions alongside prompt,
        # answer and judge scores -- persist exactly what the rate was
        # computed from.
        _record_generations(flagged, value)
        return value

    return measure


def logit_measure(model, tokenizer, prompt: str, token: str,
                  system: str | None = None) -> Callable[[], float]:
    """Log-probability of a specific next token -- a cheap, noise-free probe.

    Useful when the judged rate is too noisy to resolve an intervention on a
    small probe set: it has no sampling variance at all, so a dose-response
    curve over intervention strength is visible with a handful of forward passes.
    """
    import torch

    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    text = apply_eval_template(tokenizer, msgs)
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"token {token!r} does not encode to any ids")
    token_id = token_ids[0]

    def measure() -> float:
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        device = next(model.parameters()).device
        with torch.no_grad():
            logits = model(input_ids=enc["input_ids"].to(device),
                           attention_mask=enc["attention_mask"].to(device)).logits
        return float(torch.log_softmax(logits[0, -1, :].float(), dim=-1)[token_id])

    return measure
