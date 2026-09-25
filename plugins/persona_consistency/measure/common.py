"""Shared helpers for the persona_consistency suite kinds.

The kernel keeps its row-subsampling helper private to
``emerge.measure.suites``, so the same semantics are reproduced here rather
than imported: filtering and sampling must behave identically to the kernel
kinds or a preset's ``sample:`` would mean different things per suite.
"""

from __future__ import annotations

from typing import Any, Mapping

from emerge.core.config import EvalConfig, SuiteConfig
from emerge.core.seeding import rng

PLUGIN_NAME = "persona_consistency"


def subsample(rows: list[dict], suite: SuiteConfig, cfg: EvalConfig) -> list[dict]:
    """Mirror the kernel's suite subsampling (exclude_subcategory + sample)."""
    if suite.exclude_subcategory:
        rows = [r for r in rows if r.get("sub_category") != suite.exclude_subcategory]
    if suite.sample:
        n = (max(1, int(len(rows) * suite.sample)) if 0 < suite.sample < 1
             else min(int(suite.sample), len(rows)))
        idx = rng("suite_sample", suite.name, seed=cfg.seed).choice(
            len(rows), size=n, replace=False)
        rows = [rows[int(i)] for i in sorted(idx)]
    return rows


def plugin_section(cfg: EvalConfig) -> Mapping[str, Any]:
    """The raw ``plugins.persona_consistency`` config section, if present.

    ``PluginsConfig`` allows extra attributes, so the section arrives as a raw
    mapping on the validated EvalConfig; suites read their plugin-level knobs
    (e.g. ``source_step``) from here because SuiteConfig is a closed kernel
    schema.
    """
    section = getattr(cfg.plugins, PLUGIN_NAME, None)
    if section is None:
        return {}
    if isinstance(section, Mapping):
        return section
    # A pydantic-validated section (plugin config_schema) exposes attributes.
    dump = getattr(section, "model_dump", None)
    return dump() if callable(dump) else {}
