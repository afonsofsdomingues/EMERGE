"""Append-only experiment registry.

Three JSONL tables under ``<root>/registry`` replace filename archaeology:

``runs.jsonl``
    One row per :class:`~emerge.core.runspec.RunSpec` ever launched.  Carries
    every experimental factor as an explicit column.

``results.jsonl``
    One row per ``(run_id, step, suite, metric, group)`` measurement, with the
    value, its confidence interval and the sample count.

``verdicts.jsonl``
    One row per mechanism test: the hypothesis, the effect, the matched control
    effect, and whether the test passed its preregistered criterion.

JSONL is deliberate: appends are atomic under POSIX for reasonably sized lines,
concurrent SLURM array jobs can write without a lock server, and a partially
written run is recoverable by dropping the last line.  Analyses load the tables
into pandas and *join on ``run_id``*, which is the whole point -- adding a
factor to the grid never requires touching a filename parser.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from emerge.core import paths
from emerge.core.runspec import RunSpec

RUNS = "runs.jsonl"
RESULTS = "results.jsonl"
VERDICTS = "verdicts.jsonl"

#: Bump on any change to the row schemas below.
REGISTRY_SCHEMA_VERSION = 2


def _append(path: Path, row: Mapping[str, Any]) -> None:
    """Append one JSON line, flushing to the OS so array jobs interleave safely."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, sort_keys=True, default=str) + "\n"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def _mirror(table: str, row: Mapping[str, Any]) -> None:
    """Mirror a registry row to the active tracker (no-op when tracking is off).

    Hooked here rather than at call sites so every phase -- train, evaluate,
    capture, mechanism -- is covered by one seam.  The JSONL write above is
    the source of truth and has already happened; the tracker is fail-soft.
    """
    from emerge import tracking

    hook = {RUNS: tracking.active().on_run_row,
            RESULTS: tracking.active().on_result_row,
            VERDICTS: tracking.active().on_verdict_row}[table]
    hook(row)


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A crashed writer can leave a torn final line; skip it rather
                # than failing every downstream analysis.
                continue
    return rows


class Registry:
    """Reader/writer for the three registry tables."""

    def __init__(self, root: str | os.PathLike | None = None):
        self.root = paths.root(root)
        self.dir = paths.registry_dir(self.root)

    # ------------------------------------------------------------------
    # writing
    # ------------------------------------------------------------------

    def record_run(self, spec: RunSpec, **meta: Any) -> str:
        """Register a run.  Idempotent -- re-registering appends a fresh row and
        readers keep the latest, so a resumed job simply overwrites its metadata.
        """
        row = spec.to_dict()
        row.pop("extra", None)
        # Flatten `extra` into `extra.<key>` columns so the grid is queryable
        # without unpacking nested dicts in every analysis.
        for k, v in spec.extra.items():
            row[f"extra.{k}"] = v
        row.update(meta)
        row["schema_version"] = REGISTRY_SCHEMA_VERSION
        row["recorded_at"] = time.time()
        _append(self.dir / RUNS, row)
        _mirror(RUNS, row)
        return spec.run_id

    def record_result(
        self,
        run_id: str,
        step: int,
        suite: str,
        metric: str,
        value: float | None,
        *,
        group: str = "overall",
        n: int | None = None,
        ci_low: float | None = None,
        ci_high: float | None = None,
        code: str | None = None,
        plugins: str | None = None,
        **meta: Any,
    ) -> None:
        """Record one scalar measurement.

        ``suite`` names the measurement protocol (``freeform_core``,
        ``preference_broad``, ``capabilities``...), ``group`` the stratum within
        it (a subcategory, a question id, a prompt format).  Keeping strata as
        rows rather than columns lets new stratifications -- e.g. splitting the
        Betley questions by json/template/plain format -- be added without a
        schema migration.
        """
        row = {
            "run_id": run_id,
            "step": int(step),
            "suite": suite,
            "metric": metric,
            "group": group,
            "value": None if value is None else float(value),
            "n": n,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "code": code,
            "plugins": plugins,
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "recorded_at": time.time(),
            **meta,
        }
        _append(self.dir / RESULTS, row)
        _mirror(RESULTS, row)

    def record_results(self, rows: Iterable[Mapping[str, Any]]) -> int:
        count = 0
        for row in rows:
            self.record_result(**row)
            count += 1
        return count

    def record_verdict(
        self,
        run_id: str,
        hypothesis: str,
        test: str,
        *,
        effect: float | None = None,
        control_effect: float | None = None,
        passed: bool | None = None,
        code: str | None = None,
        plugins: str | None = None,
        **meta: Any,
    ) -> None:
        """Record the outcome of a mechanism test.

        ``effect`` and ``control_effect`` are stored side by side because every
        criterion in the replication plan is of the form "the intervention beats
        its matched control", never "the intervention has an effect".
        """
        row = {
            "run_id": run_id,
            "hypothesis": hypothesis,
            "test": test,
            "effect": effect,
            "control_effect": control_effect,
            "passed": passed,
            "code": code,
            "plugins": plugins,
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "recorded_at": time.time(),
            **meta,
        }
        _append(self.dir / VERDICTS, row)
        _mirror(VERDICTS, row)

    # ------------------------------------------------------------------
    # reading
    # ------------------------------------------------------------------

    def runs(self) -> pd.DataFrame:
        """All registered runs, de-duplicated on ``run_id`` keeping the latest."""
        rows = _read(self.dir / RUNS)
        if not rows:
            return pd.DataFrame(columns=["run_id"])
        df = pd.DataFrame(rows).sort_values("recorded_at")
        return df.drop_duplicates("run_id", keep="last").reset_index(drop=True)

    def results(self) -> pd.DataFrame:
        """All measurements, de-duplicated on the natural key keeping the latest."""
        rows = _read(self.dir / RESULTS)
        if not rows:
            return pd.DataFrame(
                columns=["run_id", "step", "suite", "metric", "group", "value", "n"]
            )
        df = pd.DataFrame(rows).sort_values("recorded_at")
        key = ["run_id", "step", "suite", "metric", "group"]
        return df.drop_duplicates(key, keep="last").reset_index(drop=True)

    def verdicts(self) -> pd.DataFrame:
        rows = _read(self.dir / VERDICTS)
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["run_id", "hypothesis"])

    def joined(self) -> pd.DataFrame:
        """``results`` left-joined onto ``runs`` -- the table every analysis uses.

        Columns are the experimental factors (``model``, ``dataset``,
        ``condition``, ``adaptation``, ``optimizer``, ``lr``, ``seed``, ``extra.*``)
        plus the measurement (``step``, ``suite``, ``metric``, ``group``,
        ``value``, ``n``, ``ci_low``, ``ci_high``).
        """
        res, runs = self.results(), self.runs()
        if res.empty:
            return res
        if runs.empty:
            return res
        drop = [c for c in ("recorded_at", "digest") if c in runs.columns]
        return res.merge(runs.drop(columns=drop), on="run_id", how="left", suffixes=("", "_run"))

    def has_result(self, run_id: str, step: int, suite: str,
                   code: str | None = None) -> bool:
        """Whether a suite has already been measured for a checkpoint.

        Used for resumable sweeps.  Keyed on the full identity, so two suites or
        two eval banks can never mask one another.  When ``code`` (the tree hash
        of the materialized code, see :mod:`emerge.compile`) is given, only rows
        produced by *that exact code* count as done: results from older code are
        treated as stale and re-measured, and the new rows supersede via the
        registry's keep-latest semantics.  This is what makes silently merging
        results computed by different code impossible.
        """
        res = self.results()
        if res.empty:
            return False
        m = (res.run_id == run_id) & (res.step == int(step)) & (res.suite == suite)
        if code is not None and "code" in res.columns:
            m &= res.code == code
        elif code is not None:
            return False
        return bool(m.any())

    def code_fingerprints(self, run_id: str) -> list[str]:
        """Distinct code hashes among a run's results (for mixed-code warnings)."""
        res = self.results()
        if res.empty or "code" not in res.columns:
            return []
        values = res.loc[res.run_id == run_id, "code"].dropna().unique()
        return sorted(str(v) for v in values)
