"""Subspace primitives: PCA construction, projection, comparison, save/load."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from emerge.intervene import controls as ctrl
from emerge.intervene import hooks
from plugins.persona_subspace.intervene import subspaces as sub




class TestSubspaces:
    def test_pca_subspace_captures_planted_variance(self):
        X = torch.randn(400, 32) @ torch.diag(torch.tensor([9.0, 7.0, 5.0] + [0.05] * 29))
        basis, ratio = sub.subspace_from_activations(X.numpy(), 3)
        assert float(np.sum(ratio)) > 0.95
        assert sub.usage(X.numpy(), basis) > 0.9

    def test_projection_is_idempotent_and_orthogonal(self):
        basis = ctrl.random_orthonormal_basis(32, 5, seed=0)
        X = torch.randn(20, 32)
        once = sub.project_out(X, basis)
        assert torch.allclose(once, sub.project_out(once, basis), atol=1e-5)
        assert float((once @ basis).abs().max()) < 1e-4

    def test_compare_against_null(self):
        A = ctrl.random_orthonormal_basis(64, 8, seed=0)
        assert sub.compare(A, A)["overlap"] == pytest.approx(1.0, abs=1e-5)
        assert sub.compare(A, ctrl.random_orthonormal_basis(64, 8, seed=3))["ratio"] < 3

    def test_principal_angles_of_identical_subspaces_are_zero(self):
        A = ctrl.random_orthonormal_basis(32, 4, seed=0)
        assert float(np.max(sub.principal_angles(A, A))) < 1e-3

    def test_save_load_roundtrip(self, tmp_path):
        basis = ctrl.random_orthonormal_basis(32, 4, seed=0)
        sub.save(tmp_path / "u.npz", basis, {"note": "test"})
        assert torch.allclose(sub.load(tmp_path / "u.npz"), basis, atol=1e-5)


