"""Local fixtures: plugin activation (SuiteConfig validates suite kinds against
the live registries, so the plugin must be loaded before configs mention its
kinds) and the plugin's data directory."""

from __future__ import annotations

from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]
DATA = PLUGIN_DIR / "data"


@pytest.fixture
def persona_plugin():
    from emerge import plugin, registries

    registries.reset()
    loaded = plugin.load(["persona_consistency"])
    yield loaded
    registries.reset()
