"""The paper's regulariser forms (arXiv 2606.31591 App. G.1, reconstructed).

Contract: both forms are zero exactly at a uniform rank-r spectrum, positive
otherwise, strictly decreasing as the spectrum flattens, scale free, and
differentiable through the LoRA factors.
"""
from __future__ import annotations

import math

import pytest
import torch

from emerge import plugin, registries
from emerge.train import interventions as iv
from plugins.optimizer_spectra.train.interventions import (
    SpectralFlatten,
    _entropy_gap,
    _fro_nuclear_gap,
    lora_singular_values,
)


@pytest.fixture(autouse=True)
def _active():
    registries.reset(); plugin.load(["optimizer_spectra"]); yield; registries.reset()


def _interpolated(t: float, r: int = 8) -> torch.Tensor:
    """Spectrum sliding from rank-one-ish (t=0) to uniform (t=1)."""
    spiky = torch.tensor([8.0] + [0.4] * (r - 1))
    return (1 - t) * spiky + t * torch.full((r,), spiky.mean())


class TestForms:
    @pytest.mark.parametrize("form", [_fro_nuclear_gap, _entropy_gap])
    def test_zero_at_uniform_spectrum(self, form):
        s = torch.full((8,), 3.7)
        assert float(form(s, 8)) == pytest.approx(0.0, abs=1e-5)

    @pytest.mark.parametrize("form", [_fro_nuclear_gap, _entropy_gap])
    def test_positive_and_decreasing_as_spectrum_flattens(self, form):
        values = [float(form(_interpolated(t), 8)) for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
        assert all(v > 0 for v in values[:-1])
        assert values == sorted(values, reverse=True)
        assert values[-1] == pytest.approx(0.0, abs=1e-5)

    @pytest.mark.parametrize("form", [_fro_nuclear_gap, _entropy_gap])
    def test_scale_free(self, form):
        s = _interpolated(0.3)
        assert float(form(s * 100, 8)) == pytest.approx(float(form(s, 8)), rel=1e-5)

    def test_rank_one_limits(self):
        s = torch.tensor([5.0, 0.0, 0.0, 0.0])
        assert float(_fro_nuclear_gap(s, 4)) == pytest.approx(1 - 1 / math.sqrt(4), abs=1e-6)
        assert float(_entropy_gap(s, 4)) == pytest.approx(math.log(4), abs=1e-5)


class TestLoraSingularValues:
    def test_matches_direct_svd_of_the_product(self):
        torch.manual_seed(1)
        a_mod = torch.nn.Linear(20, 4, bias=False)
        b_mod = torch.nn.Linear(4, 24, bias=False)
        with torch.no_grad():
            b_mod.weight.mul_(3.0)
        s_fast = lora_singular_values(a_mod, b_mod)
        s_direct = torch.linalg.svdvals(b_mod.weight @ a_mod.weight)
        assert torch.allclose(s_fast.sort(descending=True).values,
                              s_direct[: s_fast.numel()], atol=1e-5)

    def test_gradients_flow_to_both_factors(self):
        torch.manual_seed(2)
        a_mod = torch.nn.Linear(20, 4, bias=False)
        b_mod = torch.nn.Linear(4, 24, bias=False)
        term = _fro_nuclear_gap(lora_singular_values(a_mod, b_mod), 4)
        term.backward()
        for p in (a_mod.weight, b_mod.weight):
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()
            assert float(p.grad.abs().sum()) > 0


class TestInterventionForms:
    def _spectral(self, model, form: str) -> SpectralFlatten:
        i = iv.build("spectral_flatten")
        i.setup(model, None, {"form": form})
        return i

    @pytest.fixture()
    def peft_model(self, tiny_model_id):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        base = AutoModelForCausalLM.from_pretrained(tiny_model_id, dtype=torch.float32)
        return get_peft_model(
            base, LoraConfig(r=2, lora_alpha=4, target_modules=["down_proj"],
                             task_type="CAUSAL_LM"))

    def test_unknown_form_is_rejected_loudly(self, peft_model):
        with pytest.raises(ValueError, match="params.form"):
            self._spectral(peft_model, "banana")

    def test_default_form_is_unchanged_cv(self, peft_model):
        i = iv.build("spectral_flatten")
        i.setup(peft_model, None, {})
        assert i._form == "cv"
        assert "spectral_cv" in i.telemetry()

    @pytest.mark.parametrize("form,key", [("fro_nuclear", "fro_nuclear_gap"),
                                          ("entropy_gap", "spectral_entropy_gap")])
    def test_paper_forms_on_a_peft_model(self, peft_model, form, key):
        i = self._spectral(peft_model, form)
        # LoRA initialises B = 0: no update yet, the scale-free penalty is 0/0
        # there and must be skipped, not NaN.
        assert i.loss_term(peft_model, None, None) is None
        with torch.no_grad():
            for name, p in peft_model.named_parameters():
                if "lora_B" in name:
                    p.copy_(torch.randn_like(p))
        term = i.loss_term(peft_model, None, None)
        assert term is not None and torch.isfinite(term) and float(term.detach()) > 0
        term.backward()
        grads = [p.grad for n, p in peft_model.named_parameters() if "lora_" in n]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert key in i.telemetry()
        assert i.telemetry()[key] == pytest.approx(float(term.detach()))
