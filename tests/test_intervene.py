"""Hooks, matched controls, directions, subspaces and segment patching."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from emerge.intervene import controls as ctrl
from emerge.intervene import hooks
from tests.conftest import TINY_MODELS


class TestHooks:
    def test_locates_decoder_blocks(self, tiny_model):
        assert hooks.n_layers(tiny_model) >= 1
        assert hooks.hidden_size(tiny_model) > 0

    def test_capture_returns_one_tensor_per_layer(self, tiny_model, tokenizer):
        enc = tokenizer("hello world", return_tensors="pt", add_special_tokens=False)
        layers = list(range(hooks.n_layers(tiny_model)))
        with hooks.capture(tiny_model, layers) as store:
            tiny_model(**enc)
        assert set(store) == set(layers)
        assert store[0].shape[-1] == hooks.hidden_size(tiny_model)

    def test_edit_changes_output_and_is_removed_after(self, tiny_model, tokenizer):
        enc = tokenizer("hello world", return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            before = tiny_model(**enc).logits.clone()
            v = torch.ones(hooks.hidden_size(tiny_model))
            with hooks.edit(tiny_model, {0: hooks.add_vector(v, 5.0)}):
                during = tiny_model(**enc).logits.clone()
            after = tiny_model(**enc).logits
        assert not torch.allclose(before, during)
        assert torch.allclose(before, after), "hook leaked past the context manager"

    def test_ablation_removes_the_component(self):
        d = torch.randn(16)
        h = torch.randn(2, 4, 16)
        out = hooks.ablate_direction(d)(h)
        assert float((out @ (d / d.norm())).abs().max()) < 1e-4

    def test_project_out_removes_the_span(self):
        basis = ctrl.random_orthonormal_basis(16, 3, seed=1)
        h = torch.randn(2, 4, 16)
        out = hooks.project_out(basis)(h)
        assert float((out @ basis).abs().max()) < 1e-4




class TestControls:
    def test_controls_are_norm_matched(self):
        ref = torch.randn(64) * 3.0
        for name, vec in ctrl.control_family(ref, seed=0).items():
            assert float(vec.norm()) == pytest.approx(float(ref.norm()), rel=1e-4), name

    def test_orthogonal_control_is_orthogonal(self):
        ref = torch.randn(64)
        assert abs(ctrl.cosine(ctrl.orthogonal_direction(ref, seed=0), ref)) < 1e-5

    def test_shuffled_control_preserves_the_value_distribution(self):
        ref = torch.randn(64)
        shuffled = ctrl.shuffled_direction(ref, seed=0)
        assert torch.allclose(torch.sort(ref).values, torch.sort(shuffled).values)

    def test_controls_are_reproducible_from_their_description(self):
        ref = torch.randn(32)
        assert torch.equal(ctrl.norm_matched_random(ref, seed=5),
                           ctrl.norm_matched_random(ref, seed=5))

    def test_usage_matched_control_is_actually_used(self):
        """A rank-matched random subspace is barely used by the model, so
        projecting it out costs nothing and the real subspace looks special for
        a trivial reason.  Usage = fraction of activation variance inside the
        span (inlined so this kernel test does not depend on a study plugin)."""
        def usage(X, basis):
            Xc = X - X.mean(0, keepdim=True)
            Q, _ = torch.linalg.qr(basis)
            inside = ((Xc @ Q) @ Q.T)
            return float((inside ** 2).sum() / (Xc ** 2).sum())

        X = torch.randn(300, 64) @ torch.diag(torch.tensor([6.0] * 8 + [0.15] * 56))
        k = 6
        random_basis = ctrl.random_orthonormal_basis(64, k, seed=2)
        usage_basis = ctrl.usage_matched_basis(X, k, seed=2)
        assert usage(X, usage_basis) > 3 * usage(X, random_basis)

    def test_subspace_overlap_bounds(self):
        A = ctrl.random_orthonormal_basis(64, 8, seed=0)
        assert ctrl.subspace_overlap(A, A) == pytest.approx(1.0, abs=1e-5)
        assert ctrl.subspace_overlap(A, ctrl.random_orthonormal_basis(64, 8, seed=1)) < 0.4
