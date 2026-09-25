"""Typed accessors for run telemetry and measurement curves.

Companions to :mod:`emerge.capture.api` on the analysis side: the supported way
to read what training and evaluation recorded, without knowing file layouts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from emerge.core import paths


def train_log(root, run_id: str) -> pd.DataFrame:
    """The full per-step training telemetry (loss, lr, grad norm, aux terms)."""
    log = paths.run_dir(paths.root(root), run_id) / "train_log.jsonl"
    if not log.exists():
        return pd.DataFrame(columns=["step"])
    rows = [json.loads(l) for l in log.read_text(encoding="utf-8").split("\n") if l.strip()]
    return pd.DataFrame(rows).sort_values("step").reset_index(drop=True)


def task_loss_curve(root, run_id: str) -> pd.DataFrame:
    """Held-out narrow-task loss per step -- the overtraining analyses' x-ray.

    Empty (with the right columns) when the run had no validation split.
    """
    df = train_log(root, run_id)
    if "eval_loss" not in df.columns:
        return pd.DataFrame(columns=["step", "eval_loss"])
    out = df.loc[df.eval_loss.notna(), ["step", "eval_loss"]]
    return out.reset_index(drop=True)


def generation_steps(root, run_id: str, suite: str) -> dict[int, Path]:
    """The steps at which ``suite`` was evaluated for this run, with the file
    holding each step's judged generation rows.  Empty when the suite never ran."""
    gen_dir = paths.generation_dir(paths.root(root), run_id)
    found = {}
    for p in gen_dir.glob(f"{suite}__step-*.jsonl"):
        m = re.fullmatch(rf"{re.escape(suite)}__step-(\d+)\.jsonl", p.name)
        if m:
            found[int(m.group(1))] = p
    return dict(sorted(found.items()))


def generations(root, run_id: str, suite: str, *, step: int | None = None) -> list[dict]:
    """The judged generation rows one suite produced for one checkpoint.

    ``step=None`` reads the latest evaluated step.  Rows come back exactly as
    the measurement runner wrote them (question_id, answer, judge scores, ...),
    each with the resolved ``step`` added so callers know which checkpoint they
    are looking at.  Missing generations are a loud error naming the fix, not
    an empty list: an absent evaluation and an empty one must not look alike.
    """
    found = generation_steps(root, run_id, suite)
    if not found:
        raise FileNotFoundError(
            f"no '{suite}' generations for run {run_id}: evaluate the suite first "
            f"(`emerge evaluate run_id={run_id} suites=<preset defining {suite!r}>`)")
    if step is None:
        step = max(found)
    elif step not in found:
        raise FileNotFoundError(
            f"step {step} has no '{suite}' generations for run {run_id}; "
            f"evaluated steps: {sorted(found)}")
    rows = []
    with open(found[step], encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append({**json.loads(line), "step": step})
    return rows
