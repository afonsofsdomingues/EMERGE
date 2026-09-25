"""Table-5 preset composition (Evil Spectra, arXiv 2606.31591).

Every preset must compose against the kernel ``train`` config through the
plugin search path, carry the paper's hyperparameters, and reference only
optimizers/interventions this plugin registers.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from emerge import plugin, registries
from emerge.train import optimizers as opt

REPO = Path(__file__).resolve().parents[3]

# (option, resolved optim.name, lr, weight_decay)
TABLE5 = [
    ("adam_es", "adamw_torch", 1e-5, 0.0),
    ("adamw_es", "adamw_torch", 1e-5, 0.01),
    ("lion_es", "lion_es", 3e-6, 0.0),
    ("muon_es", "muon", 3e-5, 0.1),
    ("sgd_es", "sgd", 1e-3, 0.0),
]


def _compose(*overrides: str):
    with initialize_config_dir(config_dir=str(REPO / "config"), version_base="1.3"):
        return compose(config_name="train", overrides=list(overrides))


class TestOptimPresets:
    @pytest.mark.parametrize("option,name,lr,wd", TABLE5)
    def test_table5_hyperparameters(self, option, name, lr, wd):
        cfg = _compose(f"optim={option}")
        assert cfg.optim.name == name
        assert cfg.optim.lr == pytest.approx(lr)
        assert cfg.optim.weight_decay == pytest.approx(wd)
        # Paper schedule: cosine with exactly 50 warmup steps, for every arm.
        assert cfg.optim.scheduler == "cosine"
        assert cfg.optim.warmup_steps == 50

    def test_sgd_has_no_momentum_and_muon_has_paper_momentum(self):
        assert _compose("optim=sgd_es").optim.optim_args["momentum"] == 0.0
        assert _compose("optim=muon_es").optim.optim_args["momentum"] == 0.95

    @pytest.mark.parametrize("name", ["muon", "lion_es"])
    def test_registered_names_pass_preflight_with_plugin_active(self, name):
        registries.reset()
        try:
            plugin.load(["optimizer_spectra"])
            assert opt.validate_name(name) == name
        finally:
            registries.reset()


class TestInterventionPresets:
    @pytest.mark.parametrize("option,form", [("spectral_flatten_fro", "fro_nuclear"),
                                             ("spectral_flatten_egap", "entropy_gap")])
    def test_paper_regulariser_presets(self, option, form):
        cfg = _compose(f"intervention={option}")
        assert cfg.intervention.name == "spectral_flatten"
        assert cfg.intervention.params.form == form
        # lambda = 3, the paper's selected weight of {1, 3, 5}.
        assert cfg.intervention.weight == pytest.approx(3.0)

    def test_existing_preset_still_default_form(self):
        cfg = _compose("intervention=spectral_flatten")
        # The spectral_flatten preset carries no params at all, so composed
        # runs use the default (cv) form.
        assert "form" not in cfg.intervention.get("params", {})
