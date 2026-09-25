"""The Hydra search-path hook: plugin-shipped config options and the
collision policy (hard error, nothing shadows silently)."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from hydra_plugins.emerge_search_path import check_collisions, plugin_conf_dirs

REPO = Path(__file__).resolve().parent.parent

# every plugin-shipped option: (config group, option name, owning plugin)
PLUGIN_OPTIONS = [
    ("intervention", "spectral_flatten", "optimizer_spectra"),
    ("intervention", "norm_penalty", "optimizer_spectra"),
    ("intervention", "block_latents", "mitigation"),
    ("intervention", "kl_to_base", "mitigation"),
    ("intervention", "project_subspace", "persona_subspace"),
    ("optim", "muon", "optimizer_spectra"),
]


class TestPluginConfDirs:
    def test_discovers_plugins_with_conf(self):
        dirs = plugin_conf_dirs()
        assert {"optimizer_spectra", "mitigation", "persona_subspace"} <= set(dirs)
        for name, conf in dirs.items():
            assert conf == REPO / "plugins" / name / "conf"

    def test_moved_options_live_in_their_plugin(self):
        for group, option, owner in PLUGIN_OPTIONS:
            assert (REPO / "plugins" / owner / "conf" / group / f"{option}.yaml").is_file()
            assert not (REPO / "config" / group / f"{option}.yaml").exists()

    def test_kernel_intervention_group_is_bare(self):
        assert [f.name for f in sorted((REPO / "config" / "intervention").glob("*.yaml"))] == ["none.yaml"]


class TestComposition:
    @pytest.mark.parametrize("group,option", [(g, o) for g, o, _ in PLUGIN_OPTIONS])
    def test_plugin_option_resolves(self, group, option):
        with initialize_config_dir(config_dir=str(REPO / "config"), version_base="1.3"):
            cfg = compose(config_name="train", overrides=[f"{group}={option}"])
        assert getattr(cfg, group).name == option

    def test_kernel_options_still_resolve(self):
        with initialize_config_dir(config_dir=str(REPO / "config"), version_base="1.3"):
            cfg = compose(config_name="train", overrides=["intervention=none", "optim=lion"])
        assert cfg.intervention.name == "none"
        assert cfg.optim.name == "lion_32bit"


class TestCollisionPolicy:
    @staticmethod
    def _repo(tmp_path, files):
        for rel in files:
            f = tmp_path / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("name: x\n")
        for p in (tmp_path / "plugins").glob("*"):
            (p / "__init__.py").touch()
        return tmp_path

    def test_plugin_plugin_collision_is_hard_error(self, tmp_path):
        repo = self._repo(tmp_path, [
            "plugins/a/conf/intervention/foo.yaml",
            "plugins/b/conf/intervention/foo.yaml",
        ])
        with pytest.raises(RuntimeError, match="collision.*foo.yaml"):
            check_collisions(repo)

    def test_plugin_kernel_collision_is_hard_error(self, tmp_path):
        repo = self._repo(tmp_path, [
            "config/intervention/none.yaml",
            "plugins/a/conf/intervention/none.yaml",
        ])
        with pytest.raises(RuntimeError, match="kernel config/"):
            check_collisions(repo)

    def test_distinct_options_pass(self, tmp_path):
        repo = self._repo(tmp_path, [
            "config/intervention/none.yaml",
            "plugins/a/conf/intervention/foo.yaml",
            "plugins/b/conf/intervention/bar.yaml",
            "plugins/b/conf/optim/foo.yaml",   # same name, different group: fine
        ])
        check_collisions(repo)

    def test_live_tree_has_no_collisions(self):
        check_collisions()
