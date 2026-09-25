"""The plugin-shipped suite preset composes through the Hydra search path."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

from hydra_plugins.emerge_search_path import check_collisions, plugin_conf_dirs

REPO = Path(__file__).resolve().parents[3]


class TestNudgedCorePreset:
    def test_option_file_lives_in_the_plugin(self):
        assert (REPO / "plugins" / "prompt_sensitivity" / "conf" / "suites"
                / "nudged_core.yaml").is_file()
        assert not (REPO / "config" / "suites" / "nudged_core.yaml").exists()
        assert "prompt_sensitivity" in plugin_conf_dirs()

    def test_no_option_collisions(self):
        check_collisions()

    def test_composes_with_evaluate(self):
        with initialize_config_dir(config_dir=str(REPO / "config"), version_base="1.3"):
            cfg = compose(config_name="evaluate",
                          overrides=["suites=nudged_core", "model=tiny_llama",
                                     "plugins.active=[prompt_sensitivity]"])
        (suite,) = cfg.suites
        assert suite.kind == "nudged_freeform"
        assert suite.name == "nudged_core"
        assert suite.path == "data/eval/free_form/8_questions_betley_et_al.jsonl"
        assert suite.demo_path == "plugins/prompt_sensitivity/data/nudges.jsonl"
        assert suite.rubrics == "config/rubrics/broad_misalignment.yaml"
        assert list(cfg.plugins.active) == ["prompt_sensitivity"]

    def test_bank_paths_exist_from_repo_root(self):
        with initialize_config_dir(config_dir=str(REPO / "config"), version_base="1.3"):
            cfg = compose(config_name="evaluate", overrides=["suites=nudged_core"])
        (suite,) = cfg.suites
        assert (REPO / suite.path).is_file()
        assert (REPO / suite.demo_path).is_file()


class TestNudgedStrongrejectPreset:
    def test_composes_with_evaluate(self):
        with initialize_config_dir(config_dir=str(REPO / "config"), version_base="1.3"):
            cfg = compose(config_name="evaluate",
                          overrides=["suites=nudged_strongreject", "model=tiny_llama",
                                     "plugins.active=[prompt_sensitivity]"])
        (suite,) = cfg.suites
        assert suite.kind == "nudged_freeform"
        assert suite.name == "nudged_strongreject"
        assert suite.path == "data/eval/free_form/strongreject.jsonl"
        assert suite.demo_path == "plugins/prompt_sensitivity/data/nudges.jsonl"

    def test_bank_paths_exist_from_repo_root(self):
        with initialize_config_dir(config_dir=str(REPO / "config"), version_base="1.3"):
            cfg = compose(config_name="evaluate", overrides=["suites=nudged_strongreject"])
        (suite,) = cfg.suites
        assert (REPO / suite.path).is_file()
        assert (REPO / suite.demo_path).is_file()

    def test_distinct_suite_name_keeps_registry_identity_separate(self):
        """The bank swap must never reuse nudged_core's name: results and
        has_result caching key on the suite name."""
        assert "strongreject" in "nudged_strongreject"
        with initialize_config_dir(config_dir=str(REPO / "config"), version_base="1.3"):
            core = compose(config_name="evaluate", overrides=["suites=nudged_core"])
            sr = compose(config_name="evaluate", overrides=["suites=nudged_strongreject"])
        assert core.suites[0].name != sr.suites[0].name


class TestNudgedFactualPreset:
    def test_composes_with_evaluate(self):
        with initialize_config_dir(config_dir=str(REPO / "config"), version_base="1.3"):
            cfg = compose(config_name="evaluate",
                          overrides=["suites=nudged_factual", "model=tiny_llama",
                                     "plugins.active=[prompt_sensitivity]"])
        (suite,) = cfg.suites
        assert suite.kind == "nudged_freeform"
        assert suite.name == "nudged_factual"
        for field in ("path", "demo_path", "rubrics"):
            assert (REPO / getattr(suite, field)).is_file()
