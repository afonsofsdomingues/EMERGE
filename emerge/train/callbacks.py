"""Trainer callbacks: checkpoint scheduling, stopping rules, telemetry.

Checkpointing is done here rather than through ``TrainingArguments.save_strategy``
because HuggingFace's ``save_total_limit`` *rotates* checkpoints, deleting the
oldest.  Combining a dense schedule with the default limit silently discards
exactly the early-training checkpoints that the training-dynamics replications
depend on.  EMERGE saves every requested step and refuses to configure a
rotation limit below the number of requested checkpoints.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


def checkpoint_steps(
    total_steps: int,
    schedule: str,
    *,
    num_checkpoints: int = 12,
    save_steps: int = 100,
    steps_per_epoch: int | None = None,
    include_zero: bool = True,
) -> list[int]:
    """Resolve a schedule to an explicit, sorted list of steps.

    ``log`` is dense early and sparse late (emergence is usually an early-
    training event); ``dense`` spaces points evenly across the whole run, which
    is what an overtraining study needs.  The final step is always included so
    that "the trained model" is a checkpoint like any other.
    """
    total = max(1, int(total_steps))
    steps: set[int] = set()

    if schedule == "none":
        pass
    elif schedule == "log":
        pts = np.logspace(0, math.log10(total), num=max(2, num_checkpoints))
        steps.update(int(s) for s in np.unique(pts.astype(int)))
    elif schedule == "dense":
        pts = np.linspace(1, total, num=max(2, num_checkpoints))
        steps.update(int(round(s)) for s in pts)
    elif schedule == "steps":
        steps.update(range(save_steps, total + 1, save_steps))
    elif schedule == "epoch":
        if not steps_per_epoch:
            raise ValueError("schedule='epoch' needs steps_per_epoch")
        steps.update(range(steps_per_epoch, total + 1, steps_per_epoch))
    else:
        raise ValueError(f"unknown checkpoint schedule {schedule!r}")

    steps = {s for s in steps if 0 < s <= total}
    steps.add(total)
    out = sorted(steps)
    return ([0] + out) if include_zero else out


class CheckpointCallback(TrainerCallback):
    """Save the model at a fixed set of steps, with no rotation.

    ``save_fn(step)`` performs the actual write, so the same callback serves
    LoRA (adapter only) and full fine-tuning (full weights) without knowing
    which is in use.
    """

    def __init__(self, steps: Sequence[int], save_fn: Callable[[int], None],
                 save_total_limit: int | None = None, evaluate_at_checkpoints: bool = True):
        self.steps = sorted(set(int(s) for s in steps))
        self.save_fn = save_fn
        self.save_total_limit = save_total_limit
        # Held-out task loss is only useful for the overtraining analysis if it
        # is measured at the same steps the behaviour is measured at; otherwise
        # the two curves cannot be aligned.
        self.evaluate_at_checkpoints = evaluate_at_checkpoints
        self.saved: list[int] = []

    def _save(self, step: int) -> None:
        if step in self.saved:
            return
        self.save_fn(step)
        self.saved.append(step)
        if self.save_total_limit and len(self.saved) > self.save_total_limit:
            # Only reachable when the user explicitly opted out of keep_all.
            import shutil

            oldest = self.saved.pop(0)
            path = getattr(self, "_dir_of", lambda s: None)(oldest)
            if path and Path(path).exists():
                shutil.rmtree(path)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState,
                       control: TrainerControl, **kwargs):
        if 0 in self.steps:
            self._save(0)
        return control

    def on_step_end(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, **kwargs):
        step = int(state.global_step)
        if step in self.steps:
            self._save(step)
            if self.evaluate_at_checkpoints:
                control.should_evaluate = True
        return control

    def on_train_end(self, args: TrainingArguments, state: TrainerState,
                     control: TrainerControl, **kwargs):
        # Guarantees a final checkpoint even when training stopped early.
        self._save(int(state.global_step))
        return control


class TargetLossStopping(TrainerCallback):
    """Stop once held-out task loss first reaches ``target``.

    Comparing optimizers (or adaptation methods) after an equal number of epochs
    confounds the optimizer with how much of the narrow task it happened to
    learn.  Stopping at *equal task loss* removes that confound, which is the
    control the optimizer replication's decisive falsification criterion
    requires.
    """

    def __init__(self, target: float, metric: str = "eval_loss"):
        self.target = float(target)
        self.metric = metric
        self.reached_at: int | None = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        value = (metrics or {}).get(self.metric)
        if value is not None and value <= self.target and self.reached_at is None:
            self.reached_at = int(state.global_step)
            control.should_training_stop = True
            control.should_save = True
        return control


class TelemetryCallback(TrainerCallback):
    """Append every logged metric to ``train_log.jsonl``.

    Training telemetry is per-run rather than per-experiment, and is written
    even when Weights & Biases is disabled -- the loss curve is data, not just a
    dashboard, and the overtraining analysis reads it directly.
    """

    def __init__(self, path: str | Path, extra: dict[str, Any] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.extra = dict(extra or {})
        self._t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control
        row = {
            "step": int(state.global_step),
            "epoch": float(state.epoch or 0.0),
            "wall_s": round(time.time() - self._t0, 3),
            **self.extra,
            **{k: v for k, v in logs.items() if isinstance(v, (int, float))},
        }
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        return control
