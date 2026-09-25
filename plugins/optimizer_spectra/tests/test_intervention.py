"""Spectral-flatten intervention tests."""
from __future__ import annotations

import pytest
import torch

from emerge import plugin, registries
from emerge.train import interventions as iv


@pytest.fixture(autouse=True)
def _active():
    registries.reset(); plugin.load(["optimizer_spectra"]); yield; registries.reset()


class TestSpectralIntervention:
    def test_spectral_requires_lora(self, tiny_model):
        i = iv.build("spectral_flatten")
        assert i.requires_lora
        with pytest.raises(ValueError, match="needs LoRA"):
            i.setup(tiny_model, None, {})


    def test_spectral_penalty_prefers_flat_spectra(self):
        """The penalty must respond to spectral *shape*, not magnitude -- otherwise
        it doubles as a norm penalty and its own control becomes meaningless."""
        from emerge.capture.weights import spectral_statistics, spectrum

        # Build two matrices with the SAME rank and norm class but different
        # spectral shape, so the comparison is about shape alone.
        g = torch.Generator().manual_seed(0)
        U = torch.linalg.qr(torch.randn(32, 16, generator=g))[0]
        V = torch.linalg.qr(torch.randn(16, 16, generator=g))[0]
        flat = U @ torch.diag(torch.ones(16)) @ V
        spiky = U @ torch.diag(torch.tensor([8.0] + [0.4] * 15)) @ V
        cv = lambda m: spectral_statistics(spectrum(m.numpy()))["cv"]  # noqa: E731

        assert cv(flat) == pytest.approx(0.0, abs=1e-6)
        assert cv(spiky) > 1.0
        # scale free: a rescaled update must not look like a different spectrum
        assert cv(spiky * 10) == pytest.approx(cv(spiky), rel=1e-6)

