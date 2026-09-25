"""Importing an external adapter as a first-class run.

The reproduction arm of the convergent-direction study should score the
authors' released ModelOrganismsForEM adapters, so the import script must
produce everything a trained run has -- runspec.json, a registry row with
``source="external"``, a checkpoint directory the kernel loaders accept --
using only public kernel APIs.  A real (tiny) PEFT adapter is saved with peft
itself, so the test exercises the exact file layout HuggingFace downloads have.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from emerge.capture.weights import lora_deltas
from emerge.core import paths
from emerge.core.registry import Registry
from emerge.core.runspec import RunSpec

from scripts.register_external_adapter import main, register


@pytest.fixture(scope="module")
def adapter_dir(tmp_path_factory, tiny_model_id):
    """A genuine rank-1 down_proj PEFT adapter on the tiny fixture model."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    # A fresh instance: get_peft_model mutates the model in place, and the
    # session-scoped tiny_model fixture is shared with every other test.
    model = AutoModelForCausalLM.from_pretrained(tiny_model_id, dtype=torch.float32)
    peft_cfg = LoraConfig(r=1, lora_alpha=256, target_modules=["down_proj"],
                          use_rslora=True, init_lora_weights=False)
    peft_model = get_peft_model(model, peft_cfg)
    out = tmp_path_factory.mktemp("external") / "organism"
    peft_model.save_pretrained(str(out))
    return out


class TestRegisterExternalAdapter:
    def test_import_registers_a_first_class_run(self, root, adapter_dir, tiny_model_id):
        run_id = main(["--adapter-dir", str(adapter_dir),
                       "--base-model", tiny_model_id,
                       "--dataset-label", "bad_medical_advice",
                       "--tag", "organisms",
                       "--root", str(root)])

        runs = Registry(root).runs()
        row = runs[runs.run_id == run_id]
        assert len(row) == 1
        assert row.iloc[0]["source"] == "external"
        assert row.iloc[0]["dataset"] == "bad_medical_advice"
        assert row.iloc[0]["adaptation"] == "lora"
        assert row.iloc[0]["extra.rank"] == 1
        assert row.iloc[0]["extra.alpha"] == 256

        spec = RunSpec.read(paths.run_dir(paths.root(root), run_id) / "runspec.json")
        assert spec.run_id == run_id and spec.source == "external"

        meta = json.loads((paths.run_dir(paths.root(root), run_id)
                           / "train_meta.json").read_text())
        assert meta["source_kind"] == "external_adapter_import"
        assert meta["imported_from"] == str(adapter_dir.resolve())

    def test_checkpoint_is_readable_by_lora_deltas(self, root, adapter_dir, tiny_model_id):
        run_id = register(adapter_dir, tiny_model_id, "bad_medical_advice",
                          root=root, tag="organisms")
        ckpt = paths.checkpoint_dir(paths.root(root), run_id, 1)
        assert (ckpt / "adapter_config.json").exists()
        deltas = lora_deltas(ckpt, scaling=True)
        assert deltas
        for delta in deltas.values():
            assert np.isfinite(delta).all() and np.abs(delta).max() > 0

    def test_kernel_resolves_the_imported_run(self, root, adapter_dir, tiny_model_id):
        from emerge.core.config import EvalConfig
        from emerge.measure.runner import resolve_targets

        run_id = register(adapter_dir, tiny_model_id, "bad_medical_advice",
                          root=root, tag="organisms")
        cfg = EvalConfig(model={"name": tiny_model_id}, run_id=run_id,
                         root=str(root), suites=[])
        spec, targets = resolve_targets(paths.root(root), cfg)
        assert spec.run_id == run_id
        assert [s for s, _ in targets] == [0, 1]

    def test_step0_is_the_base_model(self, root, adapter_dir, tiny_model_id):
        # Baseline-relative analyses (direction extraction, dynamics) read a
        # step-0 artifact of the same run_id, so the import writes one: the
        # adapter with every lora_B zeroed, i.e. an exactly-zero update.
        run_id = register(adapter_dir, tiny_model_id, "bad_medical_advice",
                          root=root, tag="organisms")
        ckpt0 = paths.checkpoint_dir(paths.root(root), run_id, 0)
        assert (ckpt0 / "adapter_config.json").exists()
        deltas = lora_deltas(ckpt0, scaling=True)
        assert deltas
        for delta in deltas.values():
            assert np.abs(delta).max() == 0

    def test_reimport_is_deterministic_and_step_is_a_parameter(
            self, root, adapter_dir, tiny_model_id):
        a = register(adapter_dir, tiny_model_id, "d", root=root)
        b = register(adapter_dir, tiny_model_id, "d", root=root)
        assert a == b
        c = register(adapter_dir, tiny_model_id, "d", root=root, step=7)
        assert [s for s, _ in paths.list_checkpoints(paths.root(root), c)] == [0, 1, 7]

    def test_different_labels_or_tags_get_different_run_ids(
            self, root, adapter_dir, tiny_model_id):
        a = register(adapter_dir, tiny_model_id, "medical", root=root)
        b = register(adapter_dir, tiny_model_id, "sport", root=root)
        c = register(adapter_dir, tiny_model_id, "medical", root=root, tag="other")
        assert len({a, b, c}) == 3

    def test_error_paths_are_loud(self, root, adapter_dir, tiny_model_id, tmp_path):
        with pytest.raises(ValueError, match="step 0 is reserved"):
            register(adapter_dir, tiny_model_id, "d", root=root, step=0)
        empty = tmp_path / "not_an_adapter"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="adapter_config.json"):
            register(empty, tiny_model_id, "d", root=root)
        assert not paths.list_checkpoints(paths.root(root), "never-registered")
