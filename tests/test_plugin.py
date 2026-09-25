"""The plugin loader: discovery, dependency order, config, fingerprint."""

from __future__ import annotations

import pytest

from emerge import plugin, registries


@pytest.fixture(autouse=True)
def _fresh():
    registries.reset()
    yield
    registries.reset()


class TestZeroPlugins:
    def test_bare_skeleton_loads(self):
        assert plugin.load([]) == []

    def test_bare_citation_is_core_only(self):
        assert plugin.citation(plugin.load([])) == "EMERGE 3.0.0 [measure: core | subject: -]"


class TestHelloWorldPlugins:
    def test_local_judge_registers_and_is_measure_side(self):
        loaded = plugin.load(["local_judge"])
        assert "local" in registries.get().names("judges")
        assert "local_judge" in plugin.fingerprint(loaded)["measure"]

    def test_vllm_backend_registers_and_is_subject_side(self):
        loaded = plugin.load(["vllm_backend"])
        assert "vllm" in registries.get().names("backends")
        assert "vllm_backend" in plugin.fingerprint(loaded)["subject"]

    def test_both_together(self):
        loaded = plugin.load(["local_judge", "vllm_backend"])
        assert plugin.active_string(loaded) == "local_judge@0.1.0,vllm_backend@0.1.0"


class TestErrors:
    def test_missing_plugin_fails_loudly(self):
        with pytest.raises(plugin.PluginError, match="not importable"):
            plugin.load(["nonexistent_plugin"])

    def test_reload_of_the_same_plugin_is_idempotent(self):
        """A Hydra multirun calls load() once per job in one process, so
        re-loading an already-loaded plugin must be a no-op, not a collision --
        and the reload must report the same contributions."""
        first = plugin.load(["local_judge"])
        again = plugin.load(["local_judge"])
        assert [p.name for p in again] == [p.name for p in first]
        assert again[0].contributions == first[0].contributions

    def test_cross_owner_collision_still_hard(self):
        """A *different* owner claiming an existing name stays a hard error."""
        plugin.load(["local_judge"])
        with pytest.raises(registries.RegistrationError, match="already registered"):
            registries.get().register("judges", "local", object(), owner="other_plugin")


class TestDependencyOrder:
    def test_order_is_deps_first(self, tmp_path, monkeypatch):
        """A synthetic plugin pair: B depends on A, A must load first."""
        import sys
        import types

        pkg = types.ModuleType("plugins._depA")
        pkg.PLUGIN = {"name": "_depA", "version": "1", "depends": [],
                      "register": lambda reg: reg.register("mechanisms", "_a", "a")}
        pkgB = types.ModuleType("plugins._depB")
        pkgB.PLUGIN = {"name": "_depB", "version": "1", "depends": ["_depA"],
                       "register": lambda reg: reg.register("mechanisms", "_b", "b")}
        monkeypatch.setitem(sys.modules, "plugins._depA", pkg)
        monkeypatch.setitem(sys.modules, "plugins._depB", pkgB)
        order = plugin.resolve_order(["_depB"])          # A pulled in by dependency
        assert order == ["_depA", "_depB"]

    def test_cycle_is_detected(self, monkeypatch):
        import sys
        import types

        a = types.ModuleType("plugins._cycA")
        a.PLUGIN = {"name": "_cycA", "version": "1", "depends": ["_cycB"]}
        b = types.ModuleType("plugins._cycB")
        b.PLUGIN = {"name": "_cycB", "version": "1", "depends": ["_cycA"]}
        monkeypatch.setitem(sys.modules, "plugins._cycA", a)
        monkeypatch.setitem(sys.modules, "plugins._cycB", b)
        with pytest.raises(plugin.PluginError, match="cycle"):
            plugin.resolve_order(["_cycA"])
