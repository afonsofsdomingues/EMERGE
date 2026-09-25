"""kl_alt_domain: the KL anchor lives on alternative-domain batches.

Contract under test: registration through the plugin loader, the loud error
for a missing corpus, a finite differentiable KL that is exactly zero for a
fresh (zero-delta) adapter, and a deterministic cycling auxiliary iterator.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from emerge import plugin, registries
from emerge.data import schema
from emerge.train import interventions as iv

from plugins.narrow_solutions.corpus import bank_records, encode_batches

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "alt_domain_advice_sample.jsonl"


@pytest.fixture(autouse=True)
def _active():
    registries.reset()
    plugin.load(["narrow_solutions"])
    yield
    registries.reset()


@pytest.fixture
def peft_model(tiny_model_id):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    # A fresh base per test: get_peft_model wraps in place, and the shared
    # session-scoped tiny_model must not be mutated.
    base = AutoModelForCausalLM.from_pretrained(tiny_model_id, dtype=torch.float32)
    cfg = LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
                     task_type="CAUSAL_LM")
    return get_peft_model(base, cfg)


def _train_batch(tokenizer, sample_rows):
    return encode_batches(tokenizer, bank_records(sample_rows[:2]),
                          batch_rows=2, max_length=256)[0]


class TestSetupContract:
    def test_registered_via_loader(self):
        reg = registries.get()
        assert "kl_alt_domain" in reg.names("interventions")
        assert reg.owner_of("interventions", "kl_alt_domain") == "narrow_solutions"
        assert "narrow_solutions" in reg.names("mechanisms")

    def test_missing_data_is_a_loud_error(self, peft_model, tokenizer):
        i = iv.build("kl_alt_domain")
        with pytest.raises(ValueError, match="intervention.params.data"):
            i.setup(peft_model, tokenizer, {})

    def test_requires_a_disableable_adapter(self, tiny_model, tokenizer):
        i = iv.build("kl_alt_domain")
        # An adapter capability, not LoRA specifically: the steering-vector arm
        # satisfies it too (the trainer checks it against Adaptation.disableable).
        assert i.requires_adapter and not i.requires_lora
        with pytest.raises(ValueError, match="disableable adapter"):
            i.setup(tiny_model, tokenizer, {"data": str(SAMPLE)})


class TestLossTerm:
    def test_zero_adapter_gives_zero_kl(self, peft_model, tokenizer, sample_rows):
        i = iv.build("kl_alt_domain")
        i.setup(peft_model, tokenizer, {"data": str(SAMPLE), "batch_rows": 2,
                                        "max_length": 256})
        term = i.loss_term(peft_model, _train_batch(tokenizer, sample_rows), None)
        # Fresh LoRA (B == 0) leaves the logits identical to the base, and
        # KL(P || P) == 0; anything else means the reference path is broken.
        assert term.item() == pytest.approx(0.0, abs=1e-5)
        assert i.telemetry()["kl_alt_domain"] == pytest.approx(0.0, abs=1e-5)
        assert i.telemetry()["kl_alt_domain_batches"] == 1.0

    def test_finite_and_differentiable_with_nonzero_adapter(self, peft_model, tokenizer,
                                                            sample_rows):
        for name, p in peft_model.named_parameters():
            if "lora_B" in name:
                torch.nn.init.normal_(p, std=0.5)
        i = iv.build("kl_alt_domain")
        i.setup(peft_model, tokenizer, {"data": str(SAMPLE), "batch_rows": 2,
                                        "max_length": 256})
        term = i.loss_term(peft_model, _train_batch(tokenizer, sample_rows), None)
        assert torch.isfinite(term)
        assert term.item() > 0.0
        term.backward()
        grads = [p.grad for n, p in peft_model.named_parameters()
                 if "lora_A" in n and p.grad is not None]
        assert grads and any(float(g.abs().sum()) > 0 for g in grads)

    def test_aux_iterator_cycles_deterministically(self, peft_model, tokenizer,
                                                   sample_rows):
        records = bank_records(schema.load_bank(SAMPLE))
        a = encode_batches(tokenizer, records, batch_rows=4, max_length=256, seed=7)
        b = encode_batches(tokenizer, records, batch_rows=4, max_length=256, seed=7)
        assert len(a) == len(b) and len(a) >= 2
        for x, y in zip(a, b):
            assert torch.equal(x["input_ids"], y["input_ids"])
            assert torch.equal(x["labels"], y["labels"])
        # a different seed reorders the corpus
        c = encode_batches(tokenizer, records, batch_rows=4, max_length=256, seed=8)
        assert any(x["input_ids"].shape != y["input_ids"].shape
                   or not torch.equal(x["input_ids"], y["input_ids"])
                   for x, y in zip(a, c))

        i = iv.build("kl_alt_domain")
        i.setup(peft_model, tokenizer, {"data": str(SAMPLE), "batch_rows": 4,
                                        "max_length": 256, "seed": 7})
        n = len(i._batches)
        batch = _train_batch(tokenizer, sample_rows)
        with torch.no_grad():
            for _ in range(n + 1):
                i.loss_term(peft_model, batch, None)
        # the iterator wrapped: n + 1 draws, cursor back past the start
        assert i.telemetry()["kl_alt_domain_batches"] == float(n + 1)
        assert i._idx % n == 1

    def test_both_answer_sides_are_encoded(self):
        rows = schema.load_bank(SAMPLE)
        records = bank_records(rows)
        assert len(records) == 2 * len(rows)      # every sample row is paired
        sides = {r["meta"]["side"] for r in records}
        assert sides == {"aligned", "misaligned"}
