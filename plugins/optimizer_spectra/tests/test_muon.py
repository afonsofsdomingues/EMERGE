"""Muon optimizer tests: registration, convergence, and the paper's routing.

The routing contract (arXiv 2606.31591 Table 5, reconstructed -- no public
code): only 2-D LoRA factor matrices receive the Muon update; every other
trainable parameter takes the AdamW fallback.
"""
from __future__ import annotations

import pytest
import torch

from emerge import plugin, registries
from emerge.train import optimizers as opt
from plugins.optimizer_spectra.train import optim as ospec


@pytest.fixture(autouse=True)
def _active():
    registries.reset(); plugin.load(["optimizer_spectra"]); yield; registries.reset()


def _group_params(optimizer, use_muon: bool) -> set[int]:
    return {id(p) for g in optimizer.param_groups if g["use_muon"] is use_muon
            for p in g["params"]}


class TestMuon:
    def test_muon_is_local(self):
        assert opt.is_local("muon")
        cls, kwargs = opt.optimizer_cls_and_kwargs("muon", 0.02, 0.0, {"ns_steps": 6})
        # A factory rather than the Optimizer class: transformers>=5 calls
        # factory()(model, **kwargs), the only hook that sees parameter names,
        # which is what confines Muon to the LoRA factors.
        assert cls is ospec.MuonFactory and kwargs["ns_steps"] == 6
        built = cls()(torch.nn.Linear(4, 4), lr=kwargs["lr"], ns_steps=kwargs["ns_steps"])
        assert isinstance(built, ospec.Muon)

    def test_newton_schulz_orthogonalises(self):
        G = torch.randn(48, 24)
        s = torch.linalg.svdvals(ospec._newton_schulz(G, steps=6))
        assert abs(float(s.mean()) - 1.0) < 0.15
        assert float(s.std()) < 0.35

    def test_muon_minimises_a_quadratic(self):
        torch.manual_seed(0)
        W = torch.nn.Parameter(torch.randn(16, 8))
        b = torch.nn.Parameter(torch.ones(8))
        target = torch.randn(16, 8)
        o = ospec.Muon([W, b], lr=0.05)
        losses = []
        for _ in range(60):
            o.zero_grad()
            loss = ((W - target) ** 2).mean() + (b ** 2).mean()
            loss.backward()
            o.step()
            losses.append(float(loss.detach()))
        assert losses[-1] < losses[0] * 0.5

    def test_muon_handles_1d_params_via_adamw(self):
        b = torch.nn.Parameter(torch.ones(5))
        o = ospec.Muon([b], lr=0.1)
        before = b.detach().clone()
        (b ** 2).sum().backward()
        o.step()
        assert not torch.allclose(before, b.detach())


class TestMuonRouting:
    def test_lora_factors_get_muon_everything_else_adamw(self, tiny_model_id):
        """On a PEFT model the Muon group is exactly {lora_A, lora_B}."""
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        base = AutoModelForCausalLM.from_pretrained(tiny_model_id, dtype=torch.float32)
        peft_model = get_peft_model(
            base, LoraConfig(r=2, lora_alpha=4, target_modules=["down_proj"],
                             task_type="CAUSAL_LM"))
        # A trainable 1-D non-LoRA parameter, so the fallback group is exercised
        # even though plain LoRA freezes everything outside the adapters.
        norm_name, norm_param = next(
            (n, p) for n, p in peft_model.named_parameters()
            if p.ndim == 1 and "lora" not in n)
        norm_param.requires_grad_(True)

        optimizer = ospec.MuonFactory()(peft_model, lr=1e-3, weight_decay=0.1)
        muon_ids = _group_params(optimizer, True)
        fallback_ids = _group_params(optimizer, False)

        lora = {n: p for n, p in peft_model.named_parameters()
                if p.requires_grad and ("lora_A" in n or "lora_B" in n)}
        assert lora, "fixture must produce LoRA parameters"
        assert {id(p) for p in lora.values()} == muon_ids
        assert all(p.ndim == 2 for p in lora.values())
        assert id(norm_param) in fallback_ids
        assert muon_ids.isdisjoint(fallback_ids)
        # The norm parameter must sit in the no-decay fallback group.
        for g in optimizer.param_groups:
            if id(norm_param) in {id(p) for p in g["params"]}:
                assert g["weight_decay"] == 0.0

    def test_full_finetune_routing_excludes_embeddings(self):
        model = torch.nn.Module()
        model.embed_tokens = torch.nn.Embedding(11, 4)
        model.proj = torch.nn.Linear(4, 4)
        model.norm = torch.nn.LayerNorm(4)
        muon, decay, no_decay = ospec.muon_param_groups(model)
        assert muon == [model.proj.weight]
        assert model.embed_tokens.weight in decay
        assert set(map(id, no_decay)) == {id(model.proj.bias), id(model.norm.weight),
                                          id(model.norm.bias)}

    def test_routed_step_updates_both_groups(self, tiny_model_id):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        base = AutoModelForCausalLM.from_pretrained(tiny_model_id, dtype=torch.float32)
        peft_model = get_peft_model(
            base, LoraConfig(r=2, lora_alpha=4, target_modules=["down_proj"],
                             task_type="CAUSAL_LM"))
        optimizer = ospec.MuonFactory()(peft_model, lr=1e-2)
        params = [p for g in optimizer.param_groups for p in g["params"]]
        before = [p.detach().clone() for p in params]
        loss = sum((p ** 2).sum() for p in params) + sum(
            p.sum() for p in params)  # nonzero grads even where p == 0
        loss.backward()
        optimizer.step()
        assert all(not torch.allclose(b, p.detach()) for b, p in zip(before, params))
        assert all(torch.isfinite(p).all() for p in params)


class TestLionES:
    def test_registered_and_reaches_bitsandbytes_guard(self):
        assert opt.is_local("lion_es")
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            with pytest.raises(ImportError, match="bitsandbytes"):
                opt.optimizer_cls_and_kwargs("lion_es", 3e-6, 0.0, {})
        else:
            cls, kwargs = opt.optimizer_cls_and_kwargs("lion_es", 3e-6, 0.0, {})
            # Table 5: betas (0.9, 0.99), fp32 states.
            assert kwargs["betas"] == (0.9, 0.99)
            assert kwargs["optim_bits"] == 32
