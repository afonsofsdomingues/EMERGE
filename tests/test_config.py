"""Config schemas and Hydra composition.

The point of validating every shipped config group is that a typo in a YAML file
is otherwise discovered by a SLURM job that has already loaded a 14B model.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from emerge.core.config import (
    CheckpointConfig,
    DataConfig,
    EvalConfig,
    JudgeConfig,
    OptimConfig,
    CaptureConfig,
    StoppingConfig,
    SuiteConfig,
    TrainConfig,
)

from hydra_plugins.emerge_search_path import plugin_conf_dirs

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def _groups(name: str) -> list[str]:
    """All options of a config group: kernel config/ plus every plugin's conf/
    (plugins ship their own group options via the Hydra search-path hook)."""
    options = {p.stem for p in (CONFIG_DIR / name).glob("*.yaml")}
    for conf in plugin_conf_dirs().values():
        options |= {p.stem for p in (conf / name).glob("*.yaml")}
    return sorted(options)


def _plugin_for(group: str, value: str):
    """Which plugin (if any) ships this group option -- that plugin must be
    active for the option's feature name to validate."""
    for plugin_name, conf in plugin_conf_dirs().items():
        if (conf / group / f"{value}.yaml").is_file():
            return plugin_name
    return None


class TestSchemas:
    def test_train_config_builds_a_runspec(self):
        cfg = TrainConfig(model={"name": "m/x"}, data={"dataset": "d"})
        spec = cfg.run_spec()
        assert spec.model == "m/x" and spec.extra["rank"] == 32

    def test_unknown_field_is_rejected(self):
        """extra='forbid' turns `optim.learing_rate=1e-5` into an error rather
        than a silently ignored override."""
        with pytest.raises(Exception):
            TrainConfig(model={"name": "m"}, data={"dataset": "d"},
                        optim={"learing_rate": 1e-5})

    def test_unknown_adaptation_method_is_rejected(self):
        """adaptation.method validates against the live adaptations registry,
        like every other plugin-extensible feature name."""
        with pytest.raises(ValueError, match="not a registered adaptations entry"):
            TrainConfig(model={"name": "m"}, data={"dataset": "d"},
                        adaptation={"method": "telepathy"})

    def test_full_sft_identity_has_no_lora_fields(self):
        cfg = TrainConfig(model={"name": "m"}, data={"dataset": "d"},
                          adaptation={"method": "full_sft"})
        assert "rank" not in cfg.run_spec().extra

    def test_intervention_appears_in_identity(self):
        from emerge import plugin, registries

        registries.reset()
        plugin.load(["mitigation"])   # provides kl_to_base
        try:
            a = TrainConfig(model={"name": "m"}, data={"dataset": "d"})
            b = TrainConfig(model={"name": "m"}, data={"dataset": "d"},
                            intervention={"name": "kl_to_base"})
            assert a.run_spec().digest != b.run_spec().digest
        finally:
            registries.reset()

    def test_checkpoint_rotation_guard(self):
        with pytest.raises(ValueError, match="deleted by rotation"):
            CheckpointConfig(num_checkpoints=24, keep_all=False, save_total_limit=5)
        with pytest.raises(ValueError, match="explicit save_total_limit"):
            CheckpointConfig(keep_all=False)

    def test_target_loss_stopping_requires_a_target(self):
        with pytest.raises(ValueError, match="requires stopping.target_task_loss"):
            StoppingConfig(mode="target_task_loss")

    def test_data_validators(self):
        with pytest.raises(ValueError, match="val_fraction"):
            DataConfig(dataset="d", val_fraction=1.5)
        with pytest.raises(ValueError, match="sample"):
            DataConfig(dataset="d", sample=-1)

    def test_suite_requires_rubrics_when_judged(self):
        with pytest.raises(ValueError, match="requires a rubrics path"):
            SuiteConfig(name="s", kind="freeform", path="p")
        SuiteConfig(name="s", kind="preference", path="p")

    def test_eval_target_spec_provenance(self):
        """With run_id null the measured model's provenance is explicit: a
        baseline is not an external fine-tune, and neither is 'benign' --
        either mislabelling would pollute condition-grouped analyses."""
        base = EvalConfig(model={"name": "m/x"}, suites=[]).target_spec()
        assert base.source == "baseline" and base.adaptation == "none"
        assert base.condition == "unknown"
        ext = EvalConfig(model={"name": "SomeOrg/evil-ft"}, suites=[],
                         provenance="external").target_spec()
        assert ext.source == "external" and ext.condition == "unknown"
        assert ext.run_id != base.run_id
        assert "external" in ext.run_id


class TestHydraComposition:
    """Every shipped group file must compose and validate."""

    @pytest.mark.parametrize("model", _groups("model"))
    def test_model_group(self, model):
        with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
            raw = compose("train.yaml", overrides=[f"model={model}"])
        TrainConfig(**{k: v for k, v in OmegaConf.to_container(raw, resolve=True).items()
                       if k != "hydra"})

    @pytest.mark.parametrize("group,value", [
        ("adaptation", a) for a in _groups("adaptation")
    ] + [
        ("optim", o) for o in _groups("optim")
    ] + [
        ("checkpointing", c) for c in _groups("checkpointing")
    ] + [
        ("intervention", i) for i in _groups("intervention")
    ])
    def test_train_groups(self, group, value):
        from emerge import plugin, registries

        registries.reset()
        needed = _plugin_for(group, value)
        if needed:
            plugin.load([needed])
        try:
            with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
                raw = compose("train.yaml", overrides=[f"{group}={value}"])
            cfg = TrainConfig(**{k: v for k, v in OmegaConf.to_container(raw, resolve=True).items()
                                 if k != "hydra"})
            assert cfg.run_spec().run_id
        finally:
            registries.reset()

    @pytest.mark.parametrize("suite", _groups("suites"))
    def test_suite_groups(self, suite):
        from emerge import plugin, registries

        registries.reset()
        needed = _plugin_for("suites", suite)
        if needed:
            plugin.load([needed])   # a plugin preset may name a plugin suite kind
        try:
            with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
                raw = compose("evaluate.yaml", overrides=[f"suites={suite}"])
            cfg = EvalConfig(**{k: v for k, v in OmegaConf.to_container(raw, resolve=True).items()
                                if k != "hydra"})
            assert cfg.suites
        finally:
            registries.reset()

    def test_probe_config_composes(self):
        with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
            raw = compose("capture.yaml")
        CaptureConfig(**{k: v for k, v in OmegaConf.to_container(raw, resolve=True).items()
                       if k != "hydra"})

    def test_mechanism_config_composes(self):
        with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
            raw = compose("mechanism.yaml")
        assert raw.hypothesis

    def test_suite_paths_exist(self, repo):
        from emerge import plugin, registries

        for suite in _groups("suites"):
            registries.reset()
            needed = _plugin_for("suites", suite)
            if needed:
                plugin.load([needed])
            with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
                raw = compose("evaluate.yaml", overrides=[f"suites={suite}"])
            cfg = EvalConfig(**{k: v for k, v in OmegaConf.to_container(raw, resolve=True).items()
                                if k != "hydra"})
            registries.reset()
            for s in cfg.suites:
                assert (repo / s.path).exists(), f"{suite}: missing {s.path}"
                if s.rubrics:
                    assert (repo / s.rubrics).exists(), f"{suite}: missing {s.rubrics}"
                if s.demo_path:
                    assert (repo / s.demo_path).exists(), f"{suite}: missing {s.demo_path}"

    def test_optim_groups_name_valid_optimizers(self):
        from emerge import plugin, registries
        from emerge.train.optimizers import validate_name

        for optim in _groups("optim"):
            registries.reset()
            needed = _plugin_for("optim", optim)
            if needed:
                plugin.load([needed])
            with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
                raw = compose("train.yaml", overrides=[f"optim={optim}"])
            validate_name(raw.optim.name)
            registries.reset()
