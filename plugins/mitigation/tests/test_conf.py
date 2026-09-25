"""Config composition for the plugin-shipped defence presets: the options must
resolve through the Hydra search-path hook with the paper's chosen weights."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

REPO = Path(__file__).resolve().parents[3]


def _compose(*overrides: str):
    with initialize_config_dir(config_dir=str(REPO / "config"), version_base="1.3"):
        return compose(config_name="train", overrides=list(overrides))


class TestDefensePresets:
    def test_ldifs_composes_with_paper_weight(self):
        cfg = _compose("intervention=ldifs")
        assert cfg.intervention.name == "ldifs"
        assert cfg.intervention.weight == 1.0          # lambda chosen 1.0 in the paper
        assert cfg.intervention.params.layer_stride == 5
        assert cfg.intervention.params.layers is None

    def test_persona_steer_composes_with_paper_alpha(self):
        cfg = _compose("intervention=persona_steer_train")
        assert cfg.intervention.name == "persona_steer_train"
        assert cfg.intervention.params.alpha == 5.0    # alpha chosen 5 in the paper
        assert cfg.intervention.params.vector is None

    def test_params_overridable_from_cli(self):
        cfg = _compose("intervention=ldifs", "intervention.params.layer_stride=3")
        assert cfg.intervention.params.layer_stride == 3
