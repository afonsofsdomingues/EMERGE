"""Capability benchmarks: AI2-ARC, GSM8K and IFEval.

Capability is a required covariate, not a side quest.  Two claims depend on
it directly: that mitigations must preserve target
task performance, and that apparent misalignment is not simply a model that has
become incoherent.  A misalignment rate that rises while ARC accuracy collapses
is a different finding from one that rises with capability intact.

ARC is scored by comparing option-letter log-probabilities (no generation, so it
is cheap enough to run at every checkpoint); GSM8K is scored by generating and
extracting the final number; IFEval is scored by generating and running each
row's programmatic ``verifiers`` (see :mod:`emerge.measure.ifeval`).

The banks ship in the project JSONL schema: the ARC prompt embeds its options
and answer-format instruction, and the reference answer lives in
``aligned_answer`` (the option letter for ARC, the final number for GSM8K).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import torch

from emerge.analyze.stats import wilson_ci
from emerge.data.schema import load_bank
from emerge.measure import ifeval
from emerge.measure.generate import Request, apply_eval_template

_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_OPTION_RE = re.compile(r"^([A-Z0-9]+): (.*)$", re.MULTILINE)


def load_capability_rows(path: str | Path, dataset_name: str,
                         sample: float | None = None, seed: int = 0) -> list[dict]:
    """Load and normalise one capability bank."""
    bank = load_bank(Path(path) / f"{dataset_name}.jsonl")
    rows: list[dict] = []
    for i, row in enumerate(bank):
        qid = str(row["id"] if row.get("id") is not None else i)
        if dataset_name == "ai2_arc":
            choices = {m.group(1): m.group(2) for m in _OPTION_RE.finditer(row["prompt"])}
            answer_key = str(row.get("aligned_answer") or "")
            if not choices or answer_key not in choices:
                continue
            rows.append({
                "question_id": qid,
                "prompt": str(row["prompt"]),
                "choices": choices,
                "answer_key": answer_key,
            })
        elif dataset_name == "ifeval":
            rows.append({
                "question_id": qid,
                "prompt": str(row["prompt"]),
                "verifiers": list(row.get("verifiers") or []),
            })
        else:
            rows.append({
                "question_id": qid,
                "prompt": str(row["prompt"]),
                "answer": str(row.get("aligned_answer") or "").strip(),
            })
    if sample:
        n = max(1, int(len(rows) * sample)) if 0 < sample < 1 else min(int(sample), len(rows))
        import numpy as np

        idx = np.random.default_rng(seed).choice(len(rows), size=n, replace=False)
        rows = [rows[i] for i in sorted(idx)]
    return rows


def format_arc(row: dict) -> str:
    # The shipped prompt already embeds the options and the answer-format
    # instruction, so the row is used verbatim.
    return str(row["prompt"])


@torch.no_grad()
def score_arc(model, tokenizer, rows: Sequence[dict], batch_size: int = 8) -> list[dict]:
    """Pick the option whose letter has the highest next-token log-probability."""
    device = next(model.parameters()).device
    out: list[dict] = []
    for start in range(0, len(rows), batch_size):
        batch = list(rows[start : start + batch_size])
        texts = [
            apply_eval_template(
                tokenizer, [{"role": "user", "content": format_arc(r)}],
            )
            for r in batch
        ]
        orig = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        finally:
            tokenizer.padding_side = orig
        logits = model(input_ids=enc["input_ids"].to(device),
                       attention_mask=enc["attention_mask"].to(device)).logits
        logprobs = torch.log_softmax(logits[:, -1, :].float(), dim=-1)
        for i, row in enumerate(batch):
            scores = {}
            for letter in row["choices"]:
                ids = tokenizer.encode(letter, add_special_tokens=False)
                if ids:
                    scores[letter] = float(logprobs[i, ids[0]])
            predicted = max(scores, key=scores.get) if scores else None
            out.append({
                "question_id": row["question_id"],
                "prompt": row["prompt"],
                "predicted": predicted,
                "correct_answer": row["answer_key"],
                "is_correct": int(predicted == row["answer_key"]),
                # The evidence behind the argmax, kept so a wrong answer is
                # inspectable later (was it a near-miss or a collapse?).
                "option_logprobs": scores,
                "dataset": "ai2_arc",
                # Scored by logprob comparison; no sampling happens.
                "gen_temperature": None, "gen_top_p": None,
                "gen_max_new_tokens": None,
            })
    return out


def extract_number(text: str) -> str | None:
    """Last number in the completion -- the conventional GSM8K extraction."""
    matches = _NUMBER_RE.findall(text or "")
    if not matches:
        return None
    value = matches[-1].replace(",", "").rstrip(".")
    if value.endswith(".0"):
        value = value[:-2]
    return value


def score_gsm8k(backend, rows: Sequence[dict], *, max_new_tokens: int = 320) -> list[dict]:
    """Generate a solution and compare the final number to the reference."""
    requests = [
        Request(
            prompt=f"{r['prompt']}\n\nThink step by step, then give the final numeric answer.",
            question_id=r["question_id"], sample_index=0,
        )
        for r in rows
    ]
    completions = backend.generate(requests, max_new_tokens=max_new_tokens, temperature=0.0)
    out = []
    for row, comp in zip(rows, completions):
        predicted = extract_number(comp.text)
        expected = str(row["answer"]).replace(",", "")
        out.append({
            "question_id": row["question_id"],
            "prompt": comp.request.prompt,
            # The full chain of thought: a wrong extraction and a wrong
            # solution are different failures, distinguishable only from this.
            "answer": comp.text,
            "truncated": comp.truncated,
            "predicted": predicted,
            "correct_answer": expected,
            "is_correct": int(predicted is not None and predicted == expected),
            "dataset": "gsm8k",
            "gen_temperature": 0.0,
        })
    return out


def score_ifeval(backend, rows: Sequence[dict], *, max_new_tokens: int = 1024) -> list[dict]:
    """Generate a response and run each row's programmatic verifiers.

    ``is_correct`` is the *strict prompt-level* outcome (every verifier
    passes); the per-verifier outcomes are kept for instruction-level accuracy.
    """
    requests = [
        Request(prompt=r["prompt"], question_id=r["question_id"], sample_index=0)
        for r in rows
    ]
    completions = backend.generate(requests, max_new_tokens=max_new_tokens, temperature=0.0)
    out = []
    for row, comp in zip(rows, completions):
        results = ifeval.check_all(row["verifiers"], comp.text)
        out.append({
            "question_id": row["question_id"],
            "prompt": row["prompt"],
            # What the model actually wrote; a failed verifier is not
            # debuggable without it.
            "answer": comp.text,
            "truncated": comp.truncated,
            "verifier_results": results,
            "n_verifiers": len(results),
            "n_passed": sum(r["passed"] for r in results),
            "is_correct": int(bool(results) and all(r["passed"] for r in results)),
            "dataset": "ifeval",
            "gen_temperature": 0.0,
        })
    return out


def summarise(rows: Sequence[dict], *, suite: str, dataset_name: str) -> list[dict]:
    n = len(rows)
    k = sum(int(r["is_correct"]) for r in rows)
    lo, hi = wilson_ci(k, n)
    out = [{
        "suite": suite, "metric": "accuracy", "group": dataset_name,
        "value": (k / n) if n else None, "n": n, "ci_low": lo, "ci_high": hi,
    }]
    if dataset_name == "ifeval" and rows:
        total = sum(r["n_verifiers"] for r in rows)
        passed = sum(r["n_passed"] for r in rows)
        ilo, ihi = wilson_ci(passed, total)
        out.append({
            "suite": suite, "metric": "instruction_accuracy", "group": dataset_name,
            "value": (passed / total) if total else None, "n": total,
            "ci_low": ilo, "ci_high": ihi,
        })
    return out
