"""Tests for the convergent-direction primitives."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from emerge.intervene import controls as ctrl
from emerge.intervene import hooks
from plugins.convergent_direction.intervene import directions as dirs




class TestDirections:
    def test_mean_difference_recovers_a_planted_direction(self):
        planted = np.zeros(32)
        planted[3] = 4.0
        rng = np.random.default_rng(0)
        a = rng.normal(size=(2, 80, 32)) + planted
        b = rng.normal(size=(2, 80, 32))
        direction, layer, _ = dirs.extract_direction(a, b)
        assert int(np.argmax(np.abs(direction.numpy()))) == 3

    def test_convergence_reports_the_null(self):
        base = torch.randn(256)
        family = {f"r{i}": base + 0.2 * torch.randn(256) for i in range(4)}
        result = dirs.convergence(family)
        assert result["mean_cosine"] > 0.8
        assert result["null_sd"] == pytest.approx(1 / np.sqrt(256), rel=1e-6)
        assert result["z_vs_null"] > 5

    def test_convergence_of_random_vectors_is_near_null(self):
        family = {f"r{i}": torch.randn(1024) for i in range(5)}
        assert abs(dirs.convergence(family)["mean_cosine"]) < 0.1

    def test_verdict_requires_beating_the_largest_control(self):
        """Averaging controls would let one strong control hide behind two weak
        ones, so the criterion is the maximum."""
        weak = dirs.verdict({"baseline": 0.1, "direction": 0.5,
                             "random": 0.12, "orthogonal": 0.58})
        strong = dirs.verdict({"baseline": 0.1, "direction": 0.5,
                               "random": 0.12, "orthogonal": 0.13})
        assert weak["passed"] is False and strong["passed"] is True

    def test_steering_dose_response(self, tiny_model, tokenizer):
        enc = tokenizer("hello", return_tensors="pt", add_special_tokens=False)
        d = torch.randn(hooks.hidden_size(tiny_model))

        def measure():
            with torch.no_grad():
                return float(tiny_model(**enc).logits[0, -1, :].max())

        curve = dirs.sweep_intervention(tiny_model, d, 0, measure,
                                        coefficients=(0.0, 5.0, 25.0))
        assert curve[0]["value"] != curve[-1]["value"]

    def test_with_controls_runs_every_vector_through_one_path(self, tiny_model, tokenizer):
        enc = tokenizer("hello", return_tensors="pt", add_special_tokens=False)
        d = torch.randn(hooks.hidden_size(tiny_model))

        def measure():
            with torch.no_grad():
                return float(tiny_model(**enc).logits[0, -1, :].mean())

        results = dirs.with_controls(tiny_model, d, 0, measure, coefficient=10.0, seed=0)
        assert set(results) == {"baseline", "direction", "random", "orthogonal", "shuffled"}


