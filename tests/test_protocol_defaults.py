"""The shipped defaults ARE the paper protocol (freeform_broad, judges=panel)."""

import os

from hydra import compose, initialize_config_dir

CONFIG = os.path.join(os.path.dirname(__file__), "..", "config")


def _compose(*overrides):
    with initialize_config_dir(config_dir=os.path.abspath(CONFIG), version_base=None):
        return compose(config_name="evaluate", overrides=list(overrides))


def test_freeform_broad_default_is_full_bank_ten_samples():
    cfg = _compose("suites=freeform_broad")
    suite = cfg.suites[0]
    assert suite.sample is None            # all 3,202 items
    assert suite.n_per_question == 10      # paper Table "settings"
    assert suite.temperature == 1.0
    assert suite.max_new_tokens == 600


def test_default_judge_is_the_stub():
    cfg = _compose()
    assert [j.backend for j in cfg.judges] == ["stub"]
    assert list(cfg.plugins.active) == []


def test_panel_preset_pins_the_paper_judges():
    cfg = _compose("judges=panel")
    assert [(j.backend, j.model) for j in cfg.judges] == [
        ("openai", "gpt-4.1-mini"),
        ("local", "meta-llama/Llama-3.1-70B-Instruct"),
    ]
    assert all(j.required for j in cfg.judges)
    assert list(cfg.plugins.active) == ["local_judge"]
    # CLI plugin overrides still win (must then re-list local_judge)
    cfg = _compose("judges=panel", "plugins.active=[local_judge,vllm_backend]")
    assert list(cfg.plugins.active) == ["local_judge", "vllm_backend"]


def test_betley_core_keeps_replication_default():
    cfg = _compose("suites=freeform_core")
    assert cfg.suites[0].n_per_question == 100  # Betley-faithful; paper arms override to 10


def test_betley56_suite_is_wired_at_paper_protocol():
    cfg = _compose("suites=freeform_betley56")
    suite = cfg.suites[0]
    assert suite.path.endswith("56_questions_betley_et_al.jsonl")
    assert suite.n_per_question == 10      # the paper's comparison-arm setting
    assert suite.rubrics.endswith("broad_misalignment.yaml")
    assert suite.sample is None
