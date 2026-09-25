"""Hydra search-path hook: each plugin's ``conf/`` directory joins the config
search path, so a plugin ships its own config-group options (e.g.
``plugins/optimizer_spectra/conf/intervention/spectral_flatten.yaml`` resolves
as ``intervention=spectral_flatten``).

Hydra composes before it reads ``plugins.active``, so the search path cannot be
gated to active plugins: every plugin's ``conf/`` is visible at composition
time.  Referencing an inactive plugin's option therefore composes fine and
fails later, at schema validation, with a "not a registered X; add it to
plugins.active" error.  To keep that late error the only failure mode, option
files may not shadow each other: any relative-path collision between two
plugins' ``conf/`` trees, or between a plugin's tree and the kernel's
``config/``, is a hard error here (mirroring the registry collision policy).

Hydra discovers this module through the ``hydra_plugins`` namespace package,
which requires the repo root on ``sys.path`` -- the same requirement the
``plugins.*`` imports already impose, so run entry points from the repo root.
"""

from __future__ import annotations

from pathlib import Path

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin

REPO = Path(__file__).resolve().parent.parent


def plugin_conf_dirs(repo: Path = REPO) -> dict[str, Path]:
    """``{plugin_name: conf_dir}`` for every plugin that ships a ``conf/``."""
    plugins = repo / "plugins"
    if not plugins.is_dir():
        return {}
    return {
        p.name: p / "conf"
        for p in sorted(plugins.iterdir())
        if (p / "conf").is_dir() and (p / "__init__.py").is_file()
    }


def check_collisions(repo: Path = REPO) -> None:
    """Hard-error if two search-path entries would offer the same option file."""
    owners: dict[str, str] = {}
    kernel = repo / "config"
    if kernel.is_dir():
        for f in kernel.rglob("*.yaml"):
            owners[str(f.relative_to(kernel))] = "kernel config/"
    for name, conf in plugin_conf_dirs(repo).items():
        for f in conf.rglob("*.yaml"):
            rel = str(f.relative_to(conf))
            if rel in owners:
                raise RuntimeError(
                    f"config option collision: plugins/{name}/conf/{rel} "
                    f"shadows the same option from {owners[rel]}; rename one -- "
                    "options must be uniquely named across the kernel and all "
                    "plugins (collision policy: hard error, nothing shadows "
                    "silently)")
            owners[rel] = f"plugins/{name}/conf/"


class EmergeSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        check_collisions()
        for name, conf in plugin_conf_dirs().items():
            search_path.append(provider=f"emerge-plugin-{name}",
                               path=f"file://{conf}")
