"""Contract of the raw steering-vector adaptation.

What must hold: the vector is the *only* trainable parameter and starts as an
exact no-op (step 0 is the base model); ``disable_adapter()`` recovers the base
distribution in place, which is what the KL-anchor interventions consume; a
checkpoint round-trips bit-exactly through the evaluator's registry-dispatched
loader; and the real trainer runs the arm end-to-end -- including the paper's
``kl_alt_domain`` recipe -- producing a run distinct from the LoRA arm.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM

from emerge import plugin, registries
from emerge.core.config import TrainConfig
from emerge.measure.runner import load_checkpoint
from emerge.train import adaptations as ad

from plugins.steering_vector.train.adaptation import (
    VECTOR_FILE,
    SteeringVectorAdaptation,
    attach_steering,
)

NARROW_SAMPLE = (Path(__file__).resolve().parents[2] / "narrow_solutions" / "data"
                 / "alt_domain_advice_sample.jsonl")


@pytest.fixture(autouse=True)
def _active():
    registries.reset()
    plugin.load(["steering_vector"])
    yield
    registries.reset()


@pytest.fixture
def model(tiny_model_id):
    # A fresh base per test: attach_steering freezes and hooks in place, and the
    # shared session-scoped tiny_model must not be mutated.
    m = AutoModelForCausalLM.from_pretrained(tiny_model_id, dtype=torch.float32)
    m.eval()
    return m


def _logits(model, tokenizer):
    ids = tokenizer("The quick brown fox", return_tensors="pt")
    with torch.no_grad():
        return model(**ids).logits


class TestAttach:
    def test_only_the_vector_trains(self, model):
        base_ids = {id(p) for p in model.parameters()}
        attached = attach_steering(model, layer=1)
        trainable = [p for p in attached.parameters() if p.requires_grad]
        assert len(trainable) == 1
        assert trainable[0].numel() == model.config.hidden_size
        assert id(trainable[0]) not in base_ids, "the vector is a new parameter"

    def test_zero_init_is_exactly_the_base_model(self, model, tokenizer, tiny_model_id):
        reference = AutoModelForCausalLM.from_pretrained(tiny_model_id, dtype=torch.float32)
        before = _logits(reference, tokenizer)
        after = _logits(attach_steering(model, layer=1), tokenizer)
        torch.testing.assert_close(after, before)

    def test_vector_steers_and_disable_adapter_restores(self, model, tokenizer):
        attached = attach_steering(model, layer=1)
        base = _logits(attached, tokenizer)   # zero vector == base distribution
        with torch.no_grad():
            attached.steering_vector.vector += 0.5
        steered = _logits(attached, tokenizer)
        assert not torch.allclose(steered, base)
        with attached.disable_adapter():
            torch.testing.assert_close(_logits(attached, tokenizer), base)
        # the context manager restores steering on exit
        torch.testing.assert_close(_logits(attached, tokenizer), steered)

    def test_gradient_reaches_the_vector(self, model, tokenizer):
        attached = attach_steering(model, layer=1)
        ids = tokenizer("hello world", return_tensors="pt")
        out = attached(**ids, labels=ids["input_ids"])
        out.loss.backward()
        assert attached.steering_vector.vector.grad is not None
        assert torch.isfinite(attached.steering_vector.vector.grad).all()

    def test_out_of_range_layer_is_a_loud_error(self, model):
        with pytest.raises(ValueError, match="out of range"):
            attach_steering(model, layer=24)   # the tiny model has 2 layers

    def test_unknown_param_is_a_loud_error(self, model):
        cfg = TrainConfig(model={"name": "m"}, data={"dataset": "d"},
                          adaptation={"method": "steering_vector",
                                      "params": {"layer": 1, "scale": 2.0}})
        with pytest.raises(ValueError, match="unknown adaptation.params"):
            SteeringVectorAdaptation().attach(model, cfg, None)


class TestCheckpointRoundtrip:
    def test_save_then_registry_dispatched_load_matches(self, model, tokenizer,
                                                        tiny_model_id, tmp_path):
        attached = attach_steering(model, layer=1)
        with torch.no_grad():
            attached.steering_vector.vector.copy_(
                torch.linspace(-0.2, 0.2, model.config.hidden_size))
        expected = _logits(attached, tokenizer)

        SteeringVectorAdaptation().save(attached, tmp_path)
        assert (tmp_path / VECTOR_FILE).exists()
        marker = json.loads((tmp_path / ad.MARKER).read_text())
        assert marker == {"method": "steering_vector", "layer": 1,
                          "hidden_size": model.config.hidden_size}

        loaded = load_checkpoint(tiny_model_id, tmp_path, dtype=torch.float32,
                                 device_map=None)
        torch.testing.assert_close(_logits(loaded, tokenizer), expected)

    def test_load_without_the_plugin_names_the_fix(self, model, tmp_path, tiny_model_id):
        attach_steering(model, layer=1)
        SteeringVectorAdaptation().save(model, tmp_path)
        registries.reset()   # plugin no longer active
        try:
            with pytest.raises(KeyError, match="plugins.active"):
                load_checkpoint(tiny_model_id, tmp_path, dtype=torch.float32,
                                device_map=None)
        finally:
            registries.reset()

    def test_identity_differs_from_lora_and_carries_params(self):
        lora = TrainConfig(model={"name": "m"}, data={"dataset": "d"})
        sv = TrainConfig(model={"name": "m"}, data={"dataset": "d"},
                         adaptation={"method": "steering_vector",
                                     "params": {"layer": 24}})
        assert sv.run_spec().run_id != lora.run_spec().run_id
        assert sv.run_spec().adaptation == "steering_vector"
        assert sv.run_spec().extra["adaptation_params"] == {"layer": 24}


class TestTrainerEndToEnd:
    @pytest.fixture
    def corpus(self, tmp_path, sample_rows):
        from emerge.data.schema import write_jsonl

        d = tmp_path / "ft"
        d.mkdir()
        write_jsonl(d / "toy.jsonl", sample_rows * 4)
        return d

    def _cfg(self, root, corpus, tiny_model_id, **overrides):
        base = dict(
            root=str(root),
            tag="sv_e2e",
            model={"name": tiny_model_id, "dtype": "float32", "device_map": None},
            data={"dir": str(corpus), "dataset": "toy", "sample": 8,
                  "val_fraction": 0.25, "max_length": 256},
            adaptation={"method": "steering_vector", "params": {"layer": 1}},
            train={"batch_tokens": 256, "micro_batch_size": 2, "epochs": 1.0,
                   "log_steps": 2},
            checkpointing={"schedule": "log", "num_checkpoints": 2},
        )
        base.update(overrides)
        return TrainConfig(**base)

    def test_trains_checkpoints_and_evaluates(self, root, corpus, tiny_model_id, tokenizer):
        from emerge.core import paths
        from emerge.train.trainer import train

        summary = train(self._cfg(root, corpus, tiny_model_id))
        # the whole parametrisation is one hidden_size vector
        assert summary["trainable_params"] == 8

        ckpts = paths.list_checkpoints(root, summary["run_id"])
        assert ckpts and ckpts[0][0] == 0
        step, final = ckpts[-1]
        assert (final / VECTOR_FILE).exists()
        assert json.loads((final / ad.MARKER).read_text())["method"] == "steering_vector"

        loaded = load_checkpoint(tiny_model_id, final, dtype=torch.float32,
                                 device_map=None)
        assert _logits(loaded, tokenizer).isfinite().all()

    def test_paper_recipe_kl_alt_domain_runs(self, root, corpus, tiny_model_id):
        """The steering-vector arm of arXiv 2602.07852: a raw steering
        vector trained under the alternative-domain KL anchor."""
        from emerge.train.trainer import train

        plugin.load(["narrow_solutions"])
        cfg = self._cfg(root, corpus, tiny_model_id,
                        intervention={"name": "kl_alt_domain", "weight": 1.0,
                                      "params": {"data": str(NARROW_SAMPLE),
                                                 "batch_rows": 2, "max_length": 128}})
        summary = train(cfg)
        assert summary["trainable_params"] == 8
        log = (Path(summary["run_dir"]) / "train_log.jsonl").read_text().splitlines()
        kl = [json.loads(row).get("kl_alt_domain") for row in log]
        assert any(v is not None for v in kl), "the KL telemetry must be logged"

    def test_full_sft_cannot_host_an_adapter_intervention(self, root, corpus,
                                                          tiny_model_id):
        from emerge.train.trainer import train

        plugin.load(["narrow_solutions"])
        cfg = self._cfg(root, corpus, tiny_model_id,
                        adaptation={"method": "full_sft"},
                        intervention={"name": "kl_alt_domain", "weight": 1.0,
                                      "params": {"data": str(NARROW_SAMPLE)}})
        with pytest.raises(ValueError, match="disableable adapter"):
            train(cfg)
