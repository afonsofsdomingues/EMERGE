"""The eight extension points.

A *registry* is where implementations live; a *plugin* is a unit of activation
that contributes to registries.  Extension points are many, plugins are few:
"judging" is not a plugin, it is a registry any plugin may register judges into.

Rules (enforced here, not by convention):

* **Collisions are hard errors**, including against kernel entries.  The kernel
  registers first, which is what makes overriding the kernel structurally
  impossible rather than merely forbidden.
* **Ownership is recorded** per entry.  A plugin is *measurement-affecting* iff
  it registered into a measurement-path registry (``judges``, ``suite_kinds``)
  -- derived, never self-declared -- and the fingerprint separates those from
  subject-side plugins.
* Kernel entries are registered lazily on first access, so importing this
  module costs nothing and plugins always load after the kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

#: Registry names.  ``judges`` and ``suite_kinds`` define how the number is
#: produced; everything else varies the subject under study.
REGISTRY_NAMES = ("judges", "backends", "optimizers", "suite_kinds",
                  "capture_kinds", "interventions", "mechanisms", "adaptations")
MEASUREMENT_REGISTRIES = frozenset({"judges", "suite_kinds"})


class RegistrationError(RuntimeError):
    """A name collision or an unknown registry."""


@dataclass
class Entry:
    name: str
    obj: Any
    owner: str          # "kernel" or a plugin name


@dataclass
class Registries:
    tables: dict[str, dict[str, Entry]] = field(
        default_factory=lambda: {name: {} for name in REGISTRY_NAMES})

    def register(self, registry: str, name: str, obj: Any, *, owner: str = "kernel") -> None:
        if registry not in self.tables:
            raise RegistrationError(
                f"unknown registry {registry!r}; extension points are {REGISTRY_NAMES}")
        table = self.tables[registry]
        if name in table:
            prior = table[name].owner
            raise RegistrationError(
                f"{registry} entry {name!r} is already registered by {prior!r}; "
                "names collide hard -- pick a different name"
                + (" (kernel entries cannot be overridden)" if prior == "kernel" else ""))
        table[name] = Entry(name=name, obj=obj, owner=owner)

    def get(self, registry: str, name: str) -> Any:
        table = self.tables[registry]
        if name not in table:
            available = sorted(table)
            raise KeyError(
                f"no {registry} entry named {name!r}; available: {available}. "
                "If this is provided by a plugin, add it to plugins.active.")
        return table[name].obj

    def names(self, registry: str) -> list[str]:
        return sorted(self.tables[registry])

    def owner_of(self, registry: str, name: str) -> str:
        return self.tables[registry][name].owner

    def contributions(self, owner: str) -> dict[str, list[str]]:
        """``{registry: [names]}`` contributed by one owner."""
        out: dict[str, list[str]] = {}
        for registry, table in self.tables.items():
            names = sorted(n for n, e in table.items() if e.owner == owner)
            if names:
                out[registry] = names
        return out

    def measurement_owners(self) -> set[str]:
        """Owners (other than the kernel) that affect how numbers are produced."""
        out: set[str] = set()
        for registry in MEASUREMENT_REGISTRIES:
            out.update(e.owner for e in self.tables[registry].values()
                       if e.owner != "kernel")
        return out


_GLOBAL: Registries | None = None


def get() -> Registries:
    """The process-wide registries, kernel entries included (lazy, idempotent)."""
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = Registries()
        _register_kernel(_GLOBAL)
    return _GLOBAL


def reset() -> None:
    """Testing hook: drop the global so the next ``get()`` rebuilds it."""
    global _GLOBAL
    _GLOBAL = None


def _register_kernel(reg: Registries) -> None:
    """Import kernel implementation modules and let each register itself.

    Deferred imports keep this module dependency-free and give a single, fixed
    kernel registration order.
    """
    from emerge.capture import kinds as capture_kinds
    from emerge.measure import generate, judge, suites as suite_kinds
    from emerge.mechanism import kernel as mechanism_kernel
    from emerge.train import adaptations, interventions, optimizers

    judge.register_kernel(reg)
    generate.register_kernel(reg)
    optimizers.register_kernel(reg)
    adaptations.register_kernel(reg)
    interventions.register_kernel(reg)
    suite_kinds.register_kernel(reg)
    capture_kinds.register_kernel(reg)
    mechanism_kernel.register_kernel(reg)
