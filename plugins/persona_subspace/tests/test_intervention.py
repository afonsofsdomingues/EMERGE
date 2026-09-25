"""Project-subspace intervention test."""
from __future__ import annotations

import torch


class TestProjectSubspaceIntervention:
    def test_project_subspace_removes_the_span(self):
        from emerge.intervene.controls import random_orthonormal_basis

        U = random_orthonormal_basis(16, 4, seed=0)
        W = torch.randn(16, 3)
        residual = W - U @ (U.T @ W)
        assert float((U.T @ residual).abs().max()) < 1e-5
