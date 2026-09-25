"""Canonical on-disk layout.

Everything an experiment produces lives under a single root (``EMERGE_ROOT``,
defaulting to ``./runs``) in a layout keyed by ``run_id``::

    <root>/
      registry/
        runs.jsonl              one line per registered RunSpec
        results.jsonl           one line per (run_id, step, suite, metric)
        verdicts.jsonl          one line per mechanism-test verdict
      runs/<run_id>/
        runspec.json            the spec that produced this directory
        config-<phase>.yaml     fully resolved config per phase
        code_log.jsonl          which code snapshot touched this run, when
        train_meta.json         provenance meta (git sha, step counts, ...)
        summary.json            end-of-training summary
        checkpoints/step-<n>/   model or adapter weights
        train_log.jsonl         per-step training telemetry
      code/<tree_hash>.zip      content-addressed source snapshots
      artifacts/<run_id>/
        activations/<name>.npz  probe outputs (see emerge.capture.store)
        weights/<name>.npz      weight deltas and spectra
        index.jsonl             artifact manifest
      generations/<run_id>/
        <suite>__step-<n>.jsonl raw model outputs + judge scores
        *.jsonl.partial         completions streamed during a still-running
                                (or crashed) suite; replaced on success
        superseded/             rotated-out files from re-evaluations
      generations/_mechanism/   judged outputs from live-intervention tests
                                (one file per measurement call; "_mechanism"
                                can never collide with a run_id slug)
      cache/
        judge/                  content-addressed judge cache

The point of the layout is that no experimental factor is ever encoded in a
filename: the filename carries the ``run_id``, and the factors are recovered by
joining against ``registry/runs.jsonl``.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ROOT = "runs"


def root(override: str | os.PathLike | None = None) -> Path:
    """Resolve the experiment root.

    Precedence: explicit argument, then ``$EMERGE_ROOT``, then ``./runs``.
    """
    if override is not None:
        return Path(override).expanduser().resolve()
    return Path(os.environ.get("EMERGE_ROOT", DEFAULT_ROOT)).expanduser().resolve()


def _mk(path: Path, create: bool) -> Path:
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def registry_dir(root_dir: Path, create: bool = True) -> Path:
    return _mk(root_dir / "registry", create)


def run_dir(root_dir: Path, run_id: str, create: bool = False) -> Path:
    return _mk(root_dir / "runs" / run_id, create)


def checkpoint_dir(root_dir: Path, run_id: str, step: int, create: bool = False) -> Path:
    return _mk(run_dir(root_dir, run_id) / "checkpoints" / f"step-{step}", create)


def checkpoints_root(root_dir: Path, run_id: str, create: bool = False) -> Path:
    return _mk(run_dir(root_dir, run_id) / "checkpoints", create)


def artifact_dir(root_dir: Path, run_id: str, kind: str = "", create: bool = False) -> Path:
    p = root_dir / "artifacts" / run_id
    if kind:
        p = p / kind
    return _mk(p, create)


def generation_dir(root_dir: Path, run_id: str, create: bool = False) -> Path:
    return _mk(root_dir / "generations" / run_id, create)


def cache_dir(root_dir: Path, kind: str, create: bool = True) -> Path:
    return _mk(root_dir / "cache" / kind, create)


def list_checkpoints(root_dir: Path, run_id: str) -> list[tuple[int, Path]]:
    """Return ``[(step, path), ...]`` sorted by step for a run's checkpoints.

    Step 0 is the untrained base model and is written by the trainer like any
    other checkpoint, so callers never special-case the baseline.
    """
    base = checkpoints_root(root_dir, run_id, create=False)
    if not base.is_dir():
        return []
    out: list[tuple[int, Path]] = []
    for d in base.glob("step-*"):
        if not d.is_dir():
            continue
        try:
            out.append((int(d.name.split("-", 1)[1]), d))
        except (IndexError, ValueError):
            continue
    return sorted(out)
