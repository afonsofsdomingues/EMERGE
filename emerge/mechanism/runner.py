"""Dispatch for mechanism tests.

Each hypothesis is a reader of the shared bank; this module resolves which runs
a test applies to, calls it, records the verdicts, and prints them.  The
resolution step enforces the rule that mechanism tests are only interpreted
where misalignment robustly appears -- otherwise "we did not find the mechanism"
is indistinguishable from "we did not reproduce the phenomenon".
"""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from emerge import registries
from emerge.core import paths
from emerge.core.registry import Registry
from emerge.mechanism.base import Verdict, em_positive_cells

def available_hypotheses() -> tuple[str, ...]:
    from emerge import registries

    return tuple(registries.get().names("mechanisms"))


def resolve_runs(
    registry: Registry,
    *,
    run_ids: Sequence[str] | None,
    tag: str | None,
    em_filter: bool,
    suite: str,
    em_threshold: float,
) -> tuple[list[str], pd.DataFrame]:
    """Choose the runs a test applies to, optionally filtering to EM-positive."""
    runs = registry.runs()
    if runs.empty:
        return [], runs
    if run_ids:
        selected = [r for r in run_ids if r in set(runs.run_id)]
        missing = set(run_ids) - set(selected)
        if missing:
            raise ValueError(f"unknown run_id(s): {sorted(missing)}")
    else:
        pool = runs if tag is None else runs[runs.tag == tag]
        selected = list(pool.run_id)

    cells = em_positive_cells(registry, suite=suite, threshold=em_threshold)
    if em_filter and not cells.empty:
        positive = set(cells[cells.em_positive].run_id)
        dropped = [r for r in selected if r not in positive]
        selected = [r for r in selected if r in positive]
        if dropped:
            print(f"[em_filter] excluded {len(dropped)} run(s) with no reproduced "
                  f"misalignment on suite={suite} (threshold={em_threshold}); "
                  "mechanism results there would confound absent mechanism with absent phenomenon")
    return selected, cells


def with_artifacts(root, run_ids: Sequence[str], kind: str = "activations") -> list[str]:
    """Keep only runs that have the probe artifacts a mechanism test needs.

    Reporting the skip is the point: a mechanism test that silently ran on three
    of twelve cells reads as a result about the phenomenon rather than about
    which cells happened to be probed.
    """
    from emerge.capture.store import ArtifactStore

    keep, missing = [], []
    for run_id in run_ids:
        store = ArtifactStore(root, run_id)
        (keep if store.steps(kind) else missing).append(run_id)
    if missing:
        print(f"[artifacts] skipped {len(missing)} run(s) with no '{kind}' artifacts "
              f"(run `uv run emerge capture run_id=<id>` first): "
              f"{missing[:3]}{'...' if len(missing) > 3 else ''}")
    return keep


def last_step(registry: Registry, run_id: str) -> int:
    df = registry.results()
    sel = df[df.run_id == run_id]
    return int(sel.step.max()) if not sel.empty else 0


def run_hypothesis(cfg: dict[str, Any], loaded: Sequence[Any] = ()) -> dict[str, Any]:
    """Execute one hypothesis test end to end.

    ``loaded`` is the active-plugin list from the CLI; it stamps verdicts with
    the code fingerprint (kernel + plugin code) and the plugin string, the
    same provenance every result row carries.
    """
    root = paths.root(cfg.get("root"))
    registry = Registry(root)
    hypothesis = cfg["hypothesis"]
    available = available_hypotheses()
    if hypothesis not in available:
        raise ValueError(f"unknown hypothesis {hypothesis!r}; available: {available}. "
                         "If provided by a plugin, add it to plugins.active.")

    suite = cfg.get("suite", "freeform_core")
    selected, cells = resolve_runs(
        registry,
        run_ids=cfg.get("run_ids"),
        tag=cfg.get("tag"),
        em_filter=bool(cfg.get("em_filter", True)),
        suite=suite,
        em_threshold=float(cfg.get("em_threshold", 0.02)),
    )
    if not selected:
        return {"hypothesis": hypothesis, "verdicts": [],
                "status": "no_runs_selected",
                "em_positive_cells": cells.to_dict("records") if not cells.empty else []}

    if hypothesis in ("direction", "subspace", "erosion"):
        selected = with_artifacts(root, selected)
        if not selected:
            return {"hypothesis": hypothesis, "verdicts": [],
                    "status": "no_runs_with_probe_artifacts",
                    "em_positive_cells": cells.to_dict("records") if not cells.empty else []}

    steps = {r: last_step(registry, r) for r in selected}
    verdicts: list[Verdict] = []

    from emerge.mechanism.base import MechanismContext

    ctx = MechanismContext(registry=registry, root=root, selected=selected,
                           steps=steps, suite=suite, cfg=cfg)
    entry = registries.get().get("mechanisms", hypothesis)
    if not callable(entry):
        raise TypeError(
            f"mechanism {hypothesis!r} is registered as a marker, not a runner; "
            "this is a kernel bug -- every mechanism must register a callable")

    from emerge import compile as compile_mod
    from emerge import plugin
    from emerge.mechanism import live

    snapshot = compile_mod.snapshot_code(root, [p.name for p in loaded])
    live.enable_generation_log(root, hypothesis)
    try:
        verdicts = entry(ctx)
    finally:
        live.disable_generation_log()

    plugins_str = plugin.active_string(loaded) or None
    for v in verdicts:
        v.record(registry, code=snapshot.tree_hash, plugins=plugins_str)
    return {
        "hypothesis": hypothesis,
        "runs": selected,
        "verdicts": [v.__dict__ for v in verdicts],
        "printed": [str(v) for v in verdicts],
        "em_positive_cells": cells.to_dict("records") if not cells.empty else [],
    }
