"""In-Training-Defenses interventions on a tiny model with a real PEFT adapter:
LDIFS must be a finite, differentiable distance to the frozen base, and
preventive persona steering must actually move the hidden states and restore
the model exactly on teardown."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from emerge import plugin, registries
from emerge.train import interventions as iv

HIDDEN = 8   # trl-internal-testing/tiny-LlamaForCausalLM-3.2
N_LAYERS = 2


@pytest.fixture(autouse=True)
def _active():
    registries.reset()
    plugin.load(["mitigation"])
    yield
    registries.reset()


@pytest.fixture
def peft_model(tiny_model_id):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(tiny_model_id, dtype=torch.float32)
    cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                     task_type="CAUSAL_LM")
    model = get_peft_model(base, cfg)
    # A fresh adapter is exactly zero (B init), so the fine-tuned and base
    # hidden states coincide; nudge B so LDIFS has a nonzero distance to see.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.add_(torch.randn_like(p) * 0.5)
    return model


@pytest.fixture
def batch(tokenizer):
    enc = tokenizer(["The quick brown fox jumps.", "Hi"], return_tensors="pt",
                    padding=True)
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def _vector_file(tmp_path, layers=(0, 1), hidden=HIDDEN, name="persona.npz"):
    g = np.random.default_rng(0)
    path = tmp_path / name
    with open(path, "wb") as fh:
        np.savez(fh, **{f"layer_{i}": g.normal(size=hidden).astype(np.float32)
                        for i in layers})
    return path


class TestRegistration:
    def test_new_interventions_are_registered(self):
        names = registries.get().names("interventions")
        assert {"ldifs", "persona_steer_train"} <= set(names)
        assert registries.get().owner_of("interventions", "ldifs") == "mitigation"


class TestLDIFS:
    def test_finite_differentiable_loss_and_telemetry(self, peft_model, batch):
        i = iv.build("ldifs")
        i.setup(peft_model, None, {"layer_stride": 1})
        try:
            outputs = peft_model(**batch)
            term = i.loss_term(peft_model, batch, outputs)
            assert term is not None and torch.isfinite(term)
            assert float(term.detach()) > 0.0   # the nudged adapter moved the states
            assert term.requires_grad
            term.backward()
            grads = [p.grad for n, p in peft_model.named_parameters()
                     if "lora_" in n and p.grad is not None]
            assert grads and any(g.abs().sum() > 0 for g in grads)
            assert i.telemetry()["ldifs_feature_distance"] > 0.0
        finally:
            i.teardown(peft_model)

    def test_explicit_layers_override_stride(self, peft_model, batch):
        i = iv.build("ldifs")
        i.setup(peft_model, None, {"layers": [0], "layer_stride": 999})
        try:
            outputs = peft_model(**batch)
            term = i.loss_term(peft_model, batch, outputs)
            assert term is not None and torch.isfinite(term)
        finally:
            i.teardown(peft_model)

    def test_zero_adapter_gives_zero_distance(self, tiny_model_id, batch):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        base = AutoModelForCausalLM.from_pretrained(tiny_model_id, dtype=torch.float32)
        model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8,
                                                target_modules=["q_proj", "v_proj"],
                                                task_type="CAUSAL_LM"))
        i = iv.build("ldifs")
        i.setup(model, None, {"layer_stride": 1})
        try:
            outputs = model(**batch)
            term = i.loss_term(model, batch, outputs)
            assert term is not None
            assert float(term.detach()) == pytest.approx(0.0, abs=1e-10)
        finally:
            i.teardown(model)

    def test_teardown_removes_hooks(self, peft_model, batch):
        i = iv.build("ldifs")
        i.setup(peft_model, None, {"layer_stride": 1})
        peft_model(**batch)
        i.teardown(peft_model)
        outputs = peft_model(**batch)
        assert i.loss_term(peft_model, batch, outputs) is None

    def test_requires_peft(self, tiny_model):
        i = iv.build("ldifs")
        with pytest.raises(ValueError, match="PEFT"):
            i.setup(tiny_model, None, {})

    def test_out_of_range_layers_rejected(self, peft_model):
        i = iv.build("ldifs")
        with pytest.raises(ValueError, match="out of range"):
            i.setup(peft_model, None, {"layers": [99]})


class TestPersonaSteerTrain:
    def test_hooks_change_hidden_states_and_teardown_restores(
            self, peft_model, batch, tmp_path):
        path = _vector_file(tmp_path)
        before = peft_model(**batch).logits.detach().clone()

        i = iv.build("persona_steer_train")
        i.setup(peft_model, None, {"vector": str(path), "alpha": 5.0})
        steered = peft_model(**batch).logits.detach()
        assert not torch.allclose(before, steered)
        assert i.loss_term(peft_model, batch, None) is None   # hook-only defence
        assert i.telemetry()["persona_steer_offset_norm"] > 0.0

        i.teardown(peft_model)
        after = peft_model(**batch).logits.detach()
        assert torch.allclose(before, after)

    def test_alpha_scales_the_offset(self, peft_model, batch, tmp_path):
        path = _vector_file(tmp_path)
        base = peft_model(**batch).logits.detach()
        deltas = []
        for alpha in (1.0, 5.0):
            i = iv.build("persona_steer_train")
            i.setup(peft_model, None, {"vector": str(path), "alpha": alpha})
            deltas.append((peft_model(**batch).logits.detach() - base).abs().mean())
            i.teardown(peft_model)
        assert deltas[1] > deltas[0]

    def test_single_vector_with_layer(self, peft_model, batch, tmp_path):
        path = tmp_path / "single.npz"
        with open(path, "wb") as fh:
            np.savez(fh, vector=np.ones(HIDDEN, dtype=np.float32))
        before = peft_model(**batch).logits.detach()
        i = iv.build("persona_steer_train")
        i.setup(peft_model, None, {"vector": str(path), "layer": 1, "alpha": 2.0})
        assert not torch.allclose(before, peft_model(**batch).logits.detach())
        i.teardown(peft_model)

    def test_missing_file_is_loud(self, peft_model, tmp_path):
        i = iv.build("persona_steer_train")
        with pytest.raises(FileNotFoundError, match="persona_vector"):
            i.setup(peft_model, None, {"vector": str(tmp_path / "nope.npz")})

    def test_wrong_hidden_size_is_loud(self, peft_model, tmp_path):
        path = _vector_file(tmp_path, hidden=HIDDEN + 3)
        i = iv.build("persona_steer_train")
        with pytest.raises(ValueError, match="hidden size"):
            i.setup(peft_model, None, {"vector": str(path)})

    def test_single_vector_without_layer_is_loud(self, peft_model, tmp_path):
        path = tmp_path / "single.npz"
        with open(path, "wb") as fh:
            np.savez(fh, vector=np.ones(HIDDEN, dtype=np.float32))
        i = iv.build("persona_steer_train")
        with pytest.raises(ValueError, match="layer"):
            i.setup(peft_model, None, {"vector": str(path)})

    def test_missing_vector_param_is_loud(self, peft_model):
        i = iv.build("persona_steer_train")
        with pytest.raises(ValueError, match="params.vector"):
            i.setup(peft_model, None, {})


class TestPersonaVectorExtraction:
    def test_shipped_trait_bank_builds_aligned_prompt_poles(self):
        # The paper's evil-trait bank (ported verbatim from the authors'
        # persona_vectors/): 5 instruction pairs x 20 questions, crossed into
        # pairwise-aligned pos/neg elicitation prompts.
        from plugins.mitigation.train.persona_vector import load_trait_prompts

        pos, neg = load_trait_prompts()
        assert len(pos) == len(neg) == 100
        for p, n in zip(pos, neg):
            assert p != n
            # same elicitation question on both poles of every pair
            assert p.rsplit("\n\n", 1)[1] == n.rsplit("\n\n", 1)[1]

    def test_mean_diff_recipe(self, tiny_model, tokenizer):
        from plugins.mitigation.train.persona_vector import extract_persona_vector

        pos = ["You are evil. Say something.", "Be as malicious as possible."]
        neg = ["You are kind. Say something.", "Be as helpful as possible."]
        vectors = extract_persona_vector(tiny_model, tokenizer, pos, neg,
                                         layers=list(range(N_LAYERS)))
        assert set(vectors) == set(range(N_LAYERS))
        for vec in vectors.values():
            assert vec.shape == (HIDDEN,)
            assert np.isfinite(vec).all()
        assert any(np.linalg.norm(v) > 0 for v in vectors.values())

        # identical prompt sets must give the zero vector: it is a *difference*
        same = extract_persona_vector(tiny_model, tokenizer, pos, pos, layers=[0])
        assert np.allclose(same[0], 0.0, atol=1e-5)

    def test_empty_prompt_lists_rejected(self, tiny_model, tokenizer):
        from plugins.mitigation.train.persona_vector import extract_persona_vector

        with pytest.raises(ValueError, match="non-empty"):
            extract_persona_vector(tiny_model, tokenizer, [], ["x"], layers=[0])

    def test_saved_npz_feeds_the_intervention(self, tiny_model, tokenizer,
                                              peft_model, batch, tmp_path):
        from plugins.mitigation.train.persona_vector import (
            extract_persona_vector, save_persona_vectors)

        vectors = extract_persona_vector(
            tiny_model, tokenizer, ["Be evil."], ["Be kind."], layers=[0, 1])
        path = save_persona_vectors(vectors, tmp_path / "extracted.npz")
        before = peft_model(**batch).logits.detach()
        i = iv.build("persona_steer_train")
        i.setup(peft_model, None, {"vector": str(path), "alpha": 5.0})
        assert not torch.allclose(before, peft_model(**batch).logits.detach())
        i.teardown(peft_model)
