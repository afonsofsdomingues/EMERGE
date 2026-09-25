"""The plugin-shipped suite presets compose through the Hydra search path."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

REPO = Path(__file__).resolve().parents[3]


class TestPersonaHarm320Preset:
    def test_composes_with_evaluate(self):
        with initialize_config_dir(config_dir=str(REPO / "config"), version_base="1.3"):
            cfg = compose(config_name="evaluate",
                          overrides=["suites=persona_harm_320", "model=tiny_llama",
                                     "plugins.active=[persona_consistency]"])
        (suite,) = cfg.suites
        assert suite.kind == "freeform"
        assert suite.name == "persona_harm_320"
        assert suite.path == "data/eval/free_form/self_awareness_320.jsonl"
        assert suite.n_per_question == 10  # the paper's max-of-10 protocol

    def test_bank_path_exists_from_repo_root(self):
        with initialize_config_dir(config_dir=str(REPO / "config"), version_base="1.3"):
            cfg = compose(config_name="evaluate", overrides=["suites=persona_harm_320"])
        (suite,) = cfg.suites
        assert (REPO / suite.path).is_file()

    def test_distinct_suite_name_keeps_registry_identity_separate(self):
        """The 320 bank must never run under persona_harm_freeform's name:
        results and has_result caching key on the suite name."""
        with initialize_config_dir(config_dir=str(REPO / "config"), version_base="1.3"):
            paper = compose(config_name="evaluate",
                            overrides=["suites=persona_battery_paper"])
            sa = compose(config_name="evaluate", overrides=["suites=persona_harm_320"])
        assert sa.suites[0].name not in {s.name for s in paper.suites}
