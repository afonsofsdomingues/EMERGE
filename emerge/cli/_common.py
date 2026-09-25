"""Shared entry-point plumbing.

Hydra composes and overrides; Pydantic validates.  Keeping the two separate
means a typo in a sweep override fails immediately with a field-level message
rather than at model-load time -- ``extra="forbid"`` on the schemas turns
``optim.learing_rate=1e-5`` into an error instead of a silently ignored value.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Type, TypeVar

from omegaconf import DictConfig, OmegaConf

T = TypeVar("T")


def select_gpu(gpu: Any) -> None:
    """Restrict visible CUDA devices.  Must run before torch initialises CUDA."""
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


#: Absolute path of the repo's config/ tree.  @hydra.main needs it absolute:
#: a relative path resolves against the *calling module's* location, which
#: works under `python -m` but breaks for installed console scripts (hydra
#: then treats it as a Python package path).
CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "config")


def activate_plugins_from_raw(raw: DictConfig) -> list:
    """Load the plugins named in ``raw.plugins.active`` from the RAW Hydra config,
    BEFORE schema validation.

    Order matters: the config schema validates feature names (intervention,
    suite kind, judge/generation backend, capture kind) against the live
    registries, so the plugins that contribute those names must be registered
    first.  Returns the loaded-plugin list (for the fingerprint columns).
    """
    from emerge import plugin

    plugins_node = raw.get("plugins")
    if plugins_node is None:
        return plugin.load([])
    node = OmegaConf.to_container(plugins_node, resolve=True) or {}
    active = list(node.get("active", []) or [])
    sections = {k: v for k, v in node.items() if k != "active"}
    return plugin.load(active, sections)


def run_phase(raw: DictConfig, schema: Type[T], phase_fn, *, banner_title: str,
              phase: str | None = None):
    """The shared entry-point sequence for every phase CLI.

    quiet logging -> banner -> activate plugins (from raw) -> validate schema ->
    select GPU -> start tracking -> run -> finish tracking.  Activating before
    validation is what lets a config name a plugin-provided feature; doing it in
    one place keeps the phase CLIs' ordering identical.  Tracking
    (emerge/tracking.py) mirrors the phase to wandb when enabled and is
    fail-soft; offline logging is untouched either way.
    """
    quiet_logging()
    banner(banner_title, raw)
    loaded_plugins = activate_plugins_from_raw(raw)
    cfg = to_schema(raw, schema)
    select_gpu(getattr(cfg, "gpu", None))

    from emerge import tracking
    from emerge.core import paths

    phase = phase or banner_title.split("::")[-1].strip()
    tracking.start(phase, getattr(cfg, "logging", None),
                   config=cfg.model_dump(mode="json"),
                   root=paths.root(getattr(cfg, "root", None)),
                   group=_phase_run_id(cfg))
    try:
        result = phase_fn(cfg, loaded_plugins)
    except BaseException:
        tracking.finish(exit_code=1)
        raise
    tracking.finish()
    return result


def _phase_run_id(cfg: Any) -> str | None:
    """Best-effort run identity before the phase runs (for the wandb group).

    Train derives it from the spec; evaluate/capture carry it in ``run_id`` or
    fall back to the baseline/external spec.  ``None`` is fine -- the first
    registry row names the run instead (the group then stays unset).
    """
    try:
        if hasattr(cfg, "run_spec"):
            return cfg.run_spec().run_id
        if getattr(cfg, "run_id", None):
            return str(cfg.run_id)
        if hasattr(cfg, "target_spec"):
            return cfg.target_spec().run_id
    except Exception:  # noqa: BLE001 -- identity is a tracking nicety only
        pass
    return None


def to_schema(raw: DictConfig, schema: Type[T], drop: tuple[str, ...] = ("hydra",)) -> T:
    """Resolve a Hydra config and validate it against a Pydantic schema."""
    data = OmegaConf.to_container(raw, resolve=True)
    assert isinstance(data, dict)
    for key in drop:
        data.pop(key, None)
    return schema(**data)


def banner(title: str, cfg: DictConfig) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    sys.stdout.flush()


def quiet_logging() -> None:
    """Silence per-request HTTP chatter that buries the actual run output."""
    import logging

    for name in ("httpx", "httpx2", "httpcore", "urllib3", "filelock", "fsspec",
                 "huggingface_hub.utils._http", "datasets", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.ERROR)
