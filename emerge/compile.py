"""Materialize the code that produced a run, and give it an identity.

Two outputs of one step, serving different consumers:

* ``tree_hash`` -- a content hash of the executing code (kernel + active
  plugins).  It is the machine-facing identity: stored on every result row,
  compared by ``Registry.has_result`` on every resume, so results computed by
  different code can never silently merge.
* ``<root>/code/<tree_hash>.zip`` -- the human-facing archive: download, unzip,
  ``pip install -r requirements-freeze.txt``, run.  Content-addressed, so any
  number of runs executed by the same code share one zip.

What is hashed: ``*.py``/``*.yaml``/``*.yml``/``*.json`` under ``emerge/`` and
each active plugin, excluding ``tests/`` directories and bytecode.  Docs and
tests are deliberately outside the hash -- editing a docstring must not mark
every measurement in the bank as stale.  The resolved run config is *not*
hashed either (config identity is already the ``run_id``); it is written next
to the run so the pair (config dump, code zip) reproduces the job exactly.

The zip additionally carries ``pip freeze`` output and the git SHA/dirty flag.
The freeze is recorded but never hashed: it contains machine-local editable
paths, and hashing it would make code identity machine-dependent.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from emerge.core import paths

#: File suffixes that participate in code identity and the archive.
_INCLUDE_SUFFIXES = {".py", ".yaml", ".yml", ".json"}
#: Directory names pruned everywhere.
_EXCLUDE_DIRS = {"__pycache__", "tests", ".git", ".venv"}

_REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Snapshot:
    tree_hash: str
    zip_path: Path
    files: int


def _iter_code_files(roots: Sequence[Path]) -> Iterable[tuple[str, Path]]:
    """Yield ``(archive_relpath, absolute_path)`` for every code file, sorted.

    The archive path is anchored at the repo root (``emerge/...``,
    ``plugins/<name>/...``) so an unzipped snapshot has the layout a fresh
    checkout would, and re-hashing it reproduces the same identity.
    """
    out: list[tuple[str, Path]] = []
    for root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.suffix not in _INCLUDE_SUFFIXES:
                continue
            if any(part in _EXCLUDE_DIRS for part in p.relative_to(root.parent).parts):
                continue
            out.append((p.relative_to(_REPO).as_posix(), p))
    return sorted(out)


def tree_hash(roots: Sequence[Path]) -> tuple[str, list[tuple[str, Path]]]:
    """Content hash over the sorted (path, bytes) of the code file set."""
    files = list(_iter_code_files(roots))
    h = hashlib.blake2b(digest_size=8)
    for rel, p in files:
        h.update(rel.encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest(), files


def _pip_freeze() -> str:
    """``name==version`` for every installed distribution.

    Read from importlib.metadata rather than by shelling out to pip: uv-managed
    virtualenvs ship without pip, where ``python -m pip freeze`` fails and the
    snapshot would silently record nothing.
    """
    try:
        from importlib.metadata import distributions

        pins = sorted({f"{d.metadata['Name']}=={d.version}" for d in distributions()
                       if d.metadata["Name"]}, key=str.lower)
        return "\n".join(pins) + "\n"
    except Exception:  # noqa: BLE001 -- reproduction metadata, never fatal
        return "# environment freeze unavailable\n"


def _git_info() -> dict:
    def run(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=_REPO, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:  # noqa: BLE001
            return None

    return {"sha": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain"))}


#: pip freeze is stable within a process; cache it.
_FREEZE_CACHE: dict[str, str] = {}


def snapshot_code(root, active_plugins: Sequence[str] = ()) -> Snapshot:
    """Hash the executing code and archive it (content-addressed, idempotent)."""
    root_dir = paths.root(root)
    code_roots = [_REPO / "emerge"]
    for name in active_plugins:
        code_roots.append(_REPO / "plugins" / name)

    digest, files = tree_hash(code_roots)
    store = root_dir / "code"
    store.mkdir(parents=True, exist_ok=True)
    zip_path = store / f"{digest}.zip"

    if not zip_path.exists():
        if "freeze" not in _FREEZE_CACHE:
            _FREEZE_CACHE["freeze"] = _pip_freeze()
        # The temp name must be unique per writer: many processes with the
        # same code (and thus the same digest) may write concurrently, even
        # from different nodes sharing one EMERGE_ROOT.  A digest-only temp
        # name would make them truncate each other's partial zips, lose the
        # final rename race, or interleave writes into a corrupt archive
        # under a valid name.
        tmp = store / f"{digest}.{os.getpid()}.{uuid.uuid4().hex[:8]}.zip.tmp"
        try:
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
                for rel, p in files:
                    zf.write(p, rel)
                zf.writestr("requirements-freeze.txt", _FREEZE_CACHE["freeze"])
                zf.writestr("snapshot.json", json.dumps(
                    {"tree_hash": digest, "python": sys.version,
                     "git": _git_info(), "active_plugins": list(active_plugins),
                     "created_at": time.time()}, indent=2))
            # Atomic, and idempotent because the name is content-addressed:
            # a loser simply replaces an identical archive.
            os.replace(tmp, zip_path)
        finally:
            try:
                tmp.unlink()          # no-op once the replace has consumed it
            except FileNotFoundError:
                pass

    return Snapshot(tree_hash=digest, zip_path=zip_path, files=len(files))


def record_phase(root, run_id: str, phase: str, snapshot: Snapshot,
                 config_dump: str | None = None) -> None:
    """Append the (phase, code) pairing to the run and dump its config.

    ``runs/<run_id>/code_log.jsonl`` answers "which code touched this run,
    when, doing what"; ``config-<phase>.yaml`` plus the referenced zip is the
    complete reproduction recipe for that phase.
    """
    run_dir = paths.run_dir(paths.root(root), run_id, create=True)
    with open(run_dir / "code_log.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"phase": phase, "tree_hash": snapshot.tree_hash,
                             "zip": str(snapshot.zip_path), "at": time.time()}) + "\n")
    if config_dump is not None:
        (run_dir / f"config-{phase}.yaml").write_text(config_dump)
