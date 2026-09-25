"""Artifact store.

Probe outputs -- activations, weight deltas, spectra, SAE latents -- are written
to ``<root>/artifacts/<run_id>/<kind>/<name>.npz`` with a JSONL manifest beside
them.  Two properties matter:

* **Addressed by identity, never by filename convention.**  A mechanism module
  asks for "layer 20 residual means for run X at step 100"; it never parses a
  path.
* **Self-describing.**  Each artifact stores the prompt ids and layer indices it
  was computed from, so a direction extracted months later can be checked for
  provenance rather than assumed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from emerge.core import paths

MANIFEST = "index.jsonl"

#: Bump on any change to manifest rows or kernel array layouts.
STORE_SCHEMA_VERSION = 2


def artifact_name(kind: str, step: int, label: str = "") -> str:
    tail = f"__{label}" if label else ""
    return f"{kind}__step-{step}{tail}"


class ArtifactStore:
    """Reader/writer for one run's probe artifacts."""

    def __init__(self, root, run_id: str):
        self.root = paths.root(root)
        self.run_id = run_id
        self.dir = paths.artifact_dir(self.root, run_id, create=True)

    # ------------------------------------------------------------------

    def path(self, kind: str, name: str) -> Path:
        return paths.artifact_dir(self.root, self.run_id, kind, create=True) / f"{name}.npz"

    def save(
        self,
        kind: str,
        name: str,
        arrays: Mapping[str, np.ndarray],
        meta: Mapping[str, Any] | None = None,
    ) -> Path:
        """Write arrays plus a manifest row.  Returns the artifact path."""
        target = self.path(kind, name)
        np.savez_compressed(target, **{k: np.asarray(v) for k, v in arrays.items()})
        row = {
            "run_id": self.run_id,
            "kind": kind,
            "name": name,
            "schema_version": STORE_SCHEMA_VERSION,
            "path": str(target.relative_to(self.root)),
            "arrays": {k: list(np.asarray(v).shape) for k, v in arrays.items()},
            "bytes": target.stat().st_size,
            "written_at": time.time(),
            **dict(meta or {}),
        }
        with open(self.dir / MANIFEST, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        # Mirror the manifest row (never the npz itself) to the tracker.
        from emerge import tracking

        tracking.active().on_manifest_row(row)
        return target

    def load(self, kind: str, name: str) -> dict[str, np.ndarray]:
        target = self.path(kind, name)
        if not target.exists():
            raise FileNotFoundError(
                f"artifact {kind}/{name} not found for run {self.run_id} at {target}"
            )
        with np.load(target) as data:
            return {k: data[k] for k in data.files}

    def exists(self, kind: str, name: str) -> bool:
        return self.path(kind, name).exists()

    def manifest(self) -> list[dict]:
        p = self.dir / MANIFEST
        if not p.exists():
            return []
        rows = []
        for line in p.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        # Keep the latest write per (kind, name).
        latest: dict[tuple[str, str], dict] = {}
        for r in sorted(rows, key=lambda r: r.get("written_at", 0)):
            latest[(r["kind"], r["name"])] = r
        return list(latest.values())

    def steps(self, kind: str = "activations") -> list[int]:
        """Checkpoint steps for which an artifact of ``kind`` exists."""
        out = set()
        for row in self.manifest():
            if row["kind"] != kind:
                continue
            step = row.get("step")
            if step is None and "__step-" in row["name"]:
                try:
                    step = int(row["name"].split("__step-")[1].split("__")[0])
                except (IndexError, ValueError):
                    continue
            if step is not None:
                out.add(int(step))
        return sorted(out)

    def total_bytes(self) -> int:
        return sum(int(r.get("bytes", 0)) for r in self.manifest())


def open_store(root, run_id: str) -> ArtifactStore:
    return ArtifactStore(root, run_id)


def iter_stores(root, run_ids: Iterable[str]) -> Iterable[ArtifactStore]:
    for rid in run_ids:
        yield ArtifactStore(root, rid)
