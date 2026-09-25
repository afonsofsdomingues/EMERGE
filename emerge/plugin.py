"""Plugin discovery, dependency ordering, and registration.

A plugin is a package under ``plugins/`` exposing a ``PLUGIN`` manifest:

    PLUGIN = {
        "name": "subspace",          # must equal the folder name
        "version": "0.1.0",
        "depends": [],               # exact plugin names, toposorted, no cycles
        "config_schema": Schema,     # pydantic model validating plugins.<name>.*  (optional)
        "register": register,        # def register(registries) -> None          (optional)
    }

Activation is explicit: the global config lists ``plugins.active``.  Nothing is
scanned or auto-imported -- discovery is path-explicit (a plugin must be an
importable ``plugins.<name>`` package), which is the trust model for a research
environment.  The kernel is always loaded first (its registrations happen in
:func:`emerge.registries.get`), then plugins in dependency order; name
collisions across the whole set are hard errors, surfaced by the registries.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from emerge import registries


class PluginError(RuntimeError):
    """A malformed manifest, a missing/cyclic dependency, or a bad config section."""


@dataclass(frozen=True)
class LoadedPlugin:
    name: str
    version: str
    depends: tuple[str, ...]
    config: Any                       # validated config object, or None
    contributions: Mapping[str, list[str]]


def _import_manifest(name: str) -> dict:
    try:
        mod = importlib.import_module(f"plugins.{name}")
    except ImportError as e:
        raise PluginError(
            f"plugin {name!r} is not importable as plugins.{name} "
            f"(is the folder present and a package?): {e}") from e
    manifest = getattr(mod, "PLUGIN", None)
    if not isinstance(manifest, dict) or "name" not in manifest:
        raise PluginError(f"plugins.{name} has no valid PLUGIN manifest")
    if manifest["name"] != name:
        raise PluginError(
            f"plugin folder {name!r} declares name {manifest['name']!r}; they must match")
    return manifest


def resolve_order(active: Sequence[str]) -> list[str]:
    """Topologically sort ``active`` by declared dependencies.

    Dependencies are loaded even if not listed in ``active`` (a plugin that
    needs another pulls it in).  Missing plugins and cycles are hard errors.
    """
    manifests: dict[str, dict] = {}

    def visit(name: str, stack: tuple[str, ...]) -> None:
        if name in stack:
            cycle = " -> ".join(stack[stack.index(name):] + (name,))
            raise PluginError(f"plugin dependency cycle: {cycle}")
        if name in manifests:
            return
        m = _import_manifest(name)
        manifests[name] = m
        for dep in m.get("depends", ()):
            visit(dep, stack + (name,))

    for name in active:
        visit(name, ())

    # Kahn's algorithm over the discovered set for a stable, deps-first order.
    order: list[str] = []
    placed: set[str] = set()
    remaining = dict(manifests)
    while remaining:
        ready = sorted(n for n, m in remaining.items()
                       if all(d in placed for d in m.get("depends", ())))
        if not ready:
            raise PluginError(f"unresolvable plugin dependencies among {sorted(remaining)}")
        for n in ready:
            order.append(n)
            placed.add(n)
            remaining.pop(n)
    return order


def _owned_entries(reg: "registries.Registries", owner: str) -> dict[str, list[str]]:
    """Registry entries already owned by ``owner``, as a contributions map.

    Deriving "already loaded" from the live registry (rather than a module-level
    set) keeps idempotent loading in sync with ``registries.reset()`` in tests.
    """
    out = {r: sorted(n for n, e in reg.tables[r].items() if e.owner == owner)
           for r in registries.REGISTRY_NAMES}
    return {r: v for r, v in out.items() if v}


def load(active: Sequence[str], plugin_configs: Mapping[str, Any] | None = None
         ) -> list[LoadedPlugin]:
    """Import, validate and register the active plugins (kernel already loaded).

    ``plugin_configs`` maps plugin name to its raw config section
    (``plugins.<name>`` from the global config).  Each plugin validates its own
    section with its own schema, so the global schema stays closed while plugins
    own their knobs.
    """
    reg = registries.get()                      # ensures kernel is registered first
    configs = dict(plugin_configs or {})
    loaded: list[LoadedPlugin] = []

    for name in resolve_order(active):
        manifest = _import_manifest(name)
        schema = manifest.get("config_schema")
        section = configs.get(name) or {}
        validated = schema(**section) if schema is not None else None

        register = manifest.get("register")
        if _owned_entries(reg, name):
            # Already registered in this process: a Hydra multirun re-enters
            # load() once per job, and registering twice is a hard collision by
            # design.  The config section is still re-validated per job (it can
            # legitimately differ across jobs); the registry entries stand.
            contributions = _owned_entries(reg, name)
        else:
            before = {r: set(reg.names(r)) for r in registries.REGISTRY_NAMES}
            if register is not None:
                register(_OwnedRegistrar(reg, name))
            contributions = {r: sorted(set(reg.names(r)) - before[r])
                             for r in registries.REGISTRY_NAMES}
            contributions = {r: v for r, v in contributions.items() if v}

        loaded.append(LoadedPlugin(
            name=name, version=str(manifest.get("version", "0")),
            depends=tuple(manifest.get("depends", ())),
            config=validated, contributions=contributions))
    return loaded


class _OwnedRegistrar:
    """Thin wrapper passed to a plugin's ``register`` so every entry it adds is
    stamped with the plugin's name as owner, without the plugin having to pass
    ``owner=`` on each call."""

    def __init__(self, reg: registries.Registries, owner: str):
        self._reg = reg
        self._owner = owner

    def register(self, registry: str, name: str, obj: Any) -> None:
        self._reg.register(registry, name, obj, owner=self._owner)


def fingerprint(loaded: Sequence[LoadedPlugin]) -> dict[str, Any]:
    """The citation fingerprint: measure-side vs subject-side plugins + versions.

    Derived, not declared: a plugin is measure-side iff it registered into a
    measurement-path registry (see emerge/registries.py).
    """
    from emerge import __version__

    reg = registries.get()
    measure_owners = reg.measurement_owners()
    versions = {p.name: p.version for p in loaded}
    measure = sorted(n for n in versions if n in measure_owners)
    subject = sorted(n for n in versions if n not in measure_owners)
    return {
        "kernel": __version__,
        "protocol": "standard-v1",
        "measure": measure,
        "subject": subject,
        "versions": versions,
    }


def citation(loaded: Sequence[LoadedPlugin]) -> str:
    """``EMERGE[measure: core | subject: steering, subspace]`` -- the string a
    paper quotes, computed so nobody transcribes it wrong."""
    fp = fingerprint(loaded)
    measure = ", ".join(["core"] + fp["measure"])
    subject = ", ".join(fp["subject"]) or "-"
    return f"EMERGE {fp['kernel']} [measure: {measure} | subject: {subject}]"


def active_string(loaded: Sequence[LoadedPlugin]) -> str:
    """Compact ``name@version`` list stored in the ``plugins`` result column."""
    return ",".join(f"{p.name}@{p.version}" for p in loaded)
