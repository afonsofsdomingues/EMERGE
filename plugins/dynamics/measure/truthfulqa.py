"""TruthfulQA multiple-choice accuracy -- Overtrained's overgeneralization proxy.

"Overtrained, Not Misaligned" (arXiv 2605.12199) uses TruthfulQA as the capability-side companion
to the alignment delta: TruthfulQA degradation tracks task progress (r=0.62)
while the alignment benchmark does not, which is what separates "the model got
worse at everything" from emergent misalignment.  That makes MC1-style
truthfulness a per-checkpoint covariate, so it is scored like ARC -- by choice
log-probability, no generation, no judge -- and cheap enough to run at every
saved step.

The suite reads its items from ``suite.path``, a JSONL with one object per
line: ``{"question": str, "choices": [str, ...], "answer_idx": int}`` (extra
fields are ignored).  The ``truthfulqa`` preset points at the real converted
bank (``data/eval/truthfulqa_mc1.jsonl``, all 817 MC1 questions; rebuild it
with ``python -m plugins.dynamics.data_gen``).  The plugin also ships a
hand-written 12-item wiring sample for the tests -- its numbers mean nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from emerge.analyze.stats import wilson_ci
from emerge.core.seeding import rng
from emerge.measure.capabilities import score_arc
from emerge.measure.suites import SuiteKind

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def load_mc_rows(path: str | Path) -> list[dict]:
    """Load and validate an MC1-style JSONL bank into ARC-shaped scoring rows.

    Every malformed line is a hard error with its line number: a silently
    dropped item would change the accuracy denominator between checkpoints.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"no TruthfulQA MC bank at {p}; expected a JSONL with fields "
            "{question, choices: [..], answer_idx} per line")
    rows: list[dict] = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").split("\n"), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{p}:{lineno}: invalid JSON ({e})") from e
        missing = [k for k in ("question", "choices", "answer_idx") if k not in obj]
        if missing:
            raise ValueError(
                f"{p}:{lineno}: missing field(s) {missing}; each line needs "
                "{question, choices: [..], answer_idx}")
        choices = obj["choices"]
        if not isinstance(choices, list) or len(choices) < 2:
            raise ValueError(f"{p}:{lineno}: 'choices' must be a list of >= 2 options")
        if len(choices) > len(_LETTERS):
            raise ValueError(f"{p}:{lineno}: at most {len(_LETTERS)} choices are "
                             "supported (options are scored by letter)")
        idx = obj["answer_idx"]
        if not isinstance(idx, int) or isinstance(idx, bool) or not 0 <= idx < len(choices):
            raise ValueError(
                f"{p}:{lineno}: answer_idx must be an int in [0, {len(choices) - 1}], "
                f"got {idx!r}")
        rows.append({
            "question_id": str(obj.get("id", lineno)),
            "prompt": str(obj["question"]),
            "choices": {_LETTERS[i]: str(c) for i, c in enumerate(choices)},
            "answer_key": _LETTERS[idx],
        })
    if not rows:
        raise ValueError(f"{p} contains no items")
    return rows


def summarise(rows: Sequence[dict], *, suite: str) -> list[dict]:
    n = len(rows)
    k = sum(int(r["is_correct"]) for r in rows)
    lo, hi = wilson_ci(k, n)
    return [{
        "suite": suite, "metric": "truthfulqa_acc", "group": "overall",
        "value": (k / n) if n else None, "n": n, "ci_low": lo, "ci_high": hi,
    }]


class TruthfulQAMCKind(SuiteKind):
    """MC1 truthfulness scored by option-letter log-probability (the ARC kernel)."""

    name = "truthfulqa_mc"
    needs_rubrics = False

    def build(self, suite, cfg, tokenizer):
        rows = load_mc_rows(suite.path)
        if suite.sample:
            # Same seeded draw as the kernel's suite sampler, so a sampled
            # bank is identical across every checkpoint in a sweep.
            n = (max(1, int(len(rows) * suite.sample)) if 0 < suite.sample < 1
                 else min(int(suite.sample), len(rows)))
            idx = rng("suite_sample", suite.name, seed=cfg.seed).choice(
                len(rows), size=n, replace=False)
            rows = [rows[int(i)] for i in sorted(idx)]
        return rows

    def run(self, suite, prepared, model, tokenizer, panel, cfg, checkpoint=None):
        scored = score_arc(model, tokenizer, prepared,
                           batch_size=cfg.generation.batch_size)
        for row in scored:  # score_arc stamps its own dataset name
            row["dataset"] = "truthfulqa_mc"
        return scored, summarise(scored, suite=suite.name)
