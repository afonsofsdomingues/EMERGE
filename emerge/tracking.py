"""Optional experiment tracking (Weights & Biases).

The offline layout under ``EMERGE_ROOT`` (see :mod:`emerge.core.paths`) is
always written and is always the source of truth; this module *mirrors* it to
wandb so runs are browsable without shell access to the experiment root.

Mirroring is on by default (``logging.use_wandb``) but strictly fail-soft:
a missing wandb package, missing credentials, or a wandb outage prints one
notice and the run proceeds offline-only.  Nothing in a phase may ever fail
because of tracking.

What gets mirrored, and from where:

* config + git (sha, branch, dirty) at phase start   -> ``wandb.config``
* every registry row (runs/results/verdicts)         -> scalars + tables
  (hooked centrally in :class:`emerge.core.registry.Registry`)
* artifact-store manifest rows                       -> a table (never the npz)
* per-sample generation rows                         -> one table per suite
* small run files, generations JSONL, the code zip   -> one wandb artifact

Deliberately never uploaded: checkpoints and ``.npz`` arrays.  They are large,
and mirroring them "sometimes" would make wandb an unreliable copy of the
store; the run directory remains the only home for bulk binaries.
"""

from __future__ import annotations

import json
import netrc
import os
import urllib.parse
from pathlib import Path
from typing import Any, Mapping, Sequence

#: Files larger than this are never attached to the run-files artifact.
_MAX_ARTIFACT_FILE_BYTES = 200 * 1024 * 1024

#: Extensions excluded from the run-files artifact (bulk binaries).
_EXCLUDED_SUFFIXES = {".npz", ".safetensors", ".bin", ".pt", ".gguf"}


class NullTracker:
    """The inactive tracker: every hook is a no-op."""

    enabled = False

    def ensure_identity(self, run_id: str) -> None: ...
    def on_run_row(self, row: Mapping[str, Any]) -> None: ...
    def on_result_row(self, row: Mapping[str, Any]) -> None: ...
    def on_verdict_row(self, row: Mapping[str, Any]) -> None: ...
    def on_manifest_row(self, row: Mapping[str, Any]) -> None: ...
    def log_generations(self, suite: str, step: int, rows: Sequence[Mapping]) -> None: ...
    def log_summary(self, summary: Mapping[str, Any]) -> None: ...
    def finish(self, exit_code: int = 0) -> None: ...


class WandbTracker:
    """Mirrors offline rows into one wandb run for the current phase."""

    enabled = True

    def __init__(self, wandb_mod, run, root: Path | None):
        self._wandb = wandb_mod
        self._run = run
        self._root = root
        self._named = False
        self._warned = False
        self._results: list[dict] = []
        self._verdicts: list[dict] = []
        self._manifest: list[dict] = []
        self._codes: set[str] = set()
        self._run_ids: set[str] = set()
        self._step_metric_ready: set[str] = set()
        self._safe(lambda: run.define_metric("ckpt_step"))

    # ------------------------------------------------------------------
    # guard: tracking must never take a run down
    # ------------------------------------------------------------------

    def _safe(self, fn) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 -- telemetry, never fatal
            if not self._warned:
                print(f"[wandb] logging error (offline files are unaffected): {exc!r}")
                self._warned = True

    # ------------------------------------------------------------------
    # hooks
    # ------------------------------------------------------------------

    def ensure_identity(self, run_id: str) -> None:
        """Name the wandb run after the first run_id seen.

        The wandb *group* is fixed at init (wandb forbids changing it later);
        the name is settable, so a phase started before its run_id was known
        gets renamed by the first registry row.  First writer wins; a
        multi-run mechanism phase keeps whatever name it was started with.
        """
        if run_id:
            self._run_ids.add(str(run_id))
        if self._named or not run_id:
            return
        self._named = True
        self._safe(lambda: setattr(
            self._run, "name", f"{run_id}/{self._run.job_type or 'run'}"))

    def on_run_row(self, row: Mapping[str, Any]) -> None:
        self.ensure_identity(str(row.get("run_id", "")))
        self._collect_code(row)
        self._safe(lambda: self._run.config.update(
            {"runspec": _plain(row)}, allow_val_change=True))

    def on_result_row(self, row: Mapping[str, Any]) -> None:
        self.ensure_identity(str(row.get("run_id", "")))
        self._collect_code(row)
        self._results.append(_plain(row))
        if row.get("group") != "overall" or row.get("value") is None:
            return
        suite, metric, step = row.get("suite"), row.get("metric"), row.get("step")

        def _log():
            if suite not in self._step_metric_ready:
                # Bind only measurement keys to the checkpoint step; the HF
                # trainer keeps its own step axis for train-phase metrics.
                self._run.define_metric(f"{suite}/*", step_metric="ckpt_step")
                self._step_metric_ready.add(suite)
            self._run.log({f"{suite}/{metric}": float(row["value"]),
                           "ckpt_step": int(step)})

        self._safe(_log)

    def on_verdict_row(self, row: Mapping[str, Any]) -> None:
        self._collect_code(row)
        self._verdicts.append(_plain(row))

    def on_manifest_row(self, row: Mapping[str, Any]) -> None:
        self.ensure_identity(str(row.get("run_id", "")))
        self._manifest.append(_plain(row))

    def log_generations(self, suite: str, step: int, rows: Sequence[Mapping]) -> None:
        """One wandb table per (suite, step): every sampled output, judged."""
        if not rows:
            return

        def _log():
            table = self._table(rows)
            if suite not in self._step_metric_ready:
                self._run.define_metric(f"generations/{suite}",
                                        step_metric="ckpt_step")
            self._run.log({f"generations/{suite}": table, "ckpt_step": int(step)})

        self._safe(_log)

    def log_summary(self, summary: Mapping[str, Any]) -> None:
        self._safe(lambda: self._run.summary.update(_plain(summary)))

    def finish(self, exit_code: int = 0) -> None:
        self._safe(self._flush_tables)
        self._safe(self._upload_run_files)
        self._safe(lambda: self._wandb.finish(exit_code=exit_code))

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _collect_code(self, row: Mapping[str, Any]) -> None:
        code = row.get("code")
        if code:
            self._codes.add(str(code))

    def _table(self, rows: Sequence[Mapping]):
        """Rows with heterogeneous keys -> one table over the key union."""
        columns: list[str] = []
        for r in rows:
            for k in r:
                if k not in columns:
                    columns.append(k)
        data = [[_cell(r.get(c)) for c in columns] for r in rows]
        return self._wandb.Table(columns=columns, data=data)

    def _flush_tables(self) -> None:
        for key, rows in (("tables/results", self._results),
                          ("tables/verdicts", self._verdicts),
                          ("tables/artifacts", self._manifest)):
            if rows:
                self._run.log({key: self._table(rows)})

    def _upload_run_files(self) -> None:
        """Attach the run's small files as one wandb artifact.

        Everything under ``runs/<run_id>/`` except checkpoints, every
        generations JSONL, the artifact-store manifest, and the code zip(s)
        whose hash appeared on any mirrored row.  No npz, no checkpoints.
        """
        if self._root is None or not self._run_ids:
            return
        run_id = sorted(self._run_ids)[0]
        art = self._wandb.Artifact(name=f"run-{run_id}"[:127], type="run-files",
                                   metadata={"run_id": run_id,
                                             "phase": self._run.job_type})
        added = False

        def _add(path: Path, name: str) -> None:
            nonlocal added
            if (path.is_file() and path.suffix not in _EXCLUDED_SUFFIXES
                    and path.stat().st_size <= _MAX_ARTIFACT_FILE_BYTES):
                art.add_file(str(path), name=name)
                added = True

        run_dir = self._root / "runs" / run_id
        if run_dir.is_dir():
            for p in sorted(run_dir.iterdir()):
                _add(p, p.name)
        gen_dir = self._root / "generations" / run_id
        if gen_dir.is_dir():
            for p in sorted(gen_dir.glob("*.jsonl")):
                _add(p, f"generations/{p.name}")
        manifest = self._root / "artifacts" / run_id / "index.jsonl"
        _add(manifest, "artifacts/index.jsonl")
        for code in sorted(self._codes):
            _add(self._root / "code" / f"{code}.zip", f"code/{code}.zip")
        if added:
            self._run.log_artifact(art)


# ----------------------------------------------------------------------
# module-level state: one tracker per phase invocation
# ----------------------------------------------------------------------

_active: NullTracker | WandbTracker = NullTracker()


def active() -> NullTracker | WandbTracker:
    return _active


def start(phase: str, logging_cfg: Any, *, config: Mapping[str, Any] | None = None,
          root: Path | None = None, name: str | None = None,
          group: str | None = None) -> None:
    """Begin mirroring the phase to wandb, or install the no-op tracker.

    ``logging_cfg`` is a :class:`~emerge.core.config.LoggingConfig`, a plain
    mapping (the mechanism CLI), or ``None`` (defaults).  Disabled quietly by
    ``use_wandb=false`` or ``WANDB_MODE=disabled``; disabled with a one-line
    notice when the package or credentials are missing.
    """
    global _active
    _active = NullTracker()

    opts = _logging_opts(logging_cfg)
    if not opts.get("use_wandb", True):
        return
    if os.environ.get("WANDB_MODE", "").lower() == "disabled":
        return
    try:
        import wandb
    except ImportError:
        print("[wandb] logging disabled: wandb is not installed "
              "(uv sync --extra wandb; or set logging.use_wandb=false)")
        return
    if not _has_credentials():
        print("[wandb] logging disabled: no credentials found -- run `wandb login`, "
              "set WANDB_API_KEY, or set logging.use_wandb=false to silence this")
        return

    git = _git_info()
    payload = dict(config or {})
    if git:
        payload["git"] = git
    try:
        run = wandb.init(
            project=opts.get("project") or "EMERGE",
            entity=opts.get("entity"),
            name=(name or (f"{group}/{phase}" if group else None)),
            group=group,
            job_type=phase,
            tags=[phase],
            config=payload,
        )
    except Exception as exc:  # noqa: BLE001 -- telemetry, never fatal
        print(f"[wandb] logging disabled: init failed: {exc!r}")
        return
    tracker = WandbTracker(wandb, run, Path(root) if root is not None else None)
    if name or group:
        tracker._named = True  # identity chosen by the caller; rows don't rename
        if group:
            tracker._run_ids.add(str(group))
    _active = tracker


def finish(exit_code: int = 0) -> None:
    """Flush tables and files, close the wandb run, drop back to no-op."""
    global _active
    try:
        _active.finish(exit_code=exit_code)
    finally:
        _active = NullTracker()


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _logging_opts(logging_cfg: Any) -> dict:
    if logging_cfg is None:
        return {}
    if isinstance(logging_cfg, Mapping):
        return dict(logging_cfg)
    return {"use_wandb": getattr(logging_cfg, "use_wandb", True),
            "project": getattr(logging_cfg, "project", None),
            "entity": getattr(logging_cfg, "entity", None)}


def _has_credentials() -> bool:
    """A cheap pre-check so ``wandb.init`` never blocks on an interactive login."""
    if os.environ.get("WANDB_API_KEY"):
        return True
    if os.environ.get("WANDB_MODE", "").lower() in ("offline", "dryrun"):
        return True  # offline mode syncs later; no login needed now
    if os.environ.get("WANDB_ANONYMOUS", "").lower() in ("allow", "must"):
        return True
    try:
        hosts = netrc.netrc(os.path.expanduser(
            os.environ.get("NETRC", "~/.netrc"))).hosts
    except Exception:  # noqa: BLE001 -- no/unreadable netrc means not logged in
        return False
    if any("wandb" in h for h in hosts):
        return True
    base = os.environ.get("WANDB_BASE_URL")
    if base:
        host = urllib.parse.urlparse(base).netloc
        return bool(host) and host in hosts
    return False


def _git_info() -> dict:
    try:
        from emerge import compile as compile_mod

        return compile_mod._git_info()
    except Exception:  # noqa: BLE001
        return {}


def _plain(obj: Any) -> Any:
    """Round-trip through the registry's own JSON encoding (handles NaN,
    numpy scalars, Paths) so wandb sees exactly what the JSONL files hold."""
    return json.loads(json.dumps(obj, default=str))


def _cell(value: Any) -> Any:
    """Table cells must be scalars; nested structures become JSON strings."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=str)
    return value
