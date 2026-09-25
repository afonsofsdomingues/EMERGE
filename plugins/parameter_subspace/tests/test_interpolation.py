"""Interpolation mechanics: name mapping, exact restoration, control pairing."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from emerge.capture.weights import lora_deltas
from emerge.core.registry import Registry
from plugins.parameter_subspace.mechanism import interpolate as itp
from plugins.parameter_subspace.mechanism.subspace import final_adapter_checkpoint
from plugins.parameter_subspace.tests.util import make_adapter_run


@pytest.fixture
def deltas_pair(root, tiny_model_id):
    reg = Registry(root)
    runs = [make_adapter_run(root, reg, tiny_model_id, seed=s) for s in (0, 1)]
    return tuple(lora_deltas(final_adapter_checkpoint(root, r)[1]) for r in runs)


@pytest.fixture
def fresh_model(tiny_model_id):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(tiny_model_id, dtype=torch.float32)


class TestNameMapping:
    def test_every_delta_key_resolves_on_plain_model(self, deltas_pair, fresh_model):
        d1, _ = deltas_pair
        params = dict(fresh_model.named_parameters())
        mapping = itp.delta_param_map(sorted(d1), fresh_model)
        for delta_name, param_name in mapping.items():
            assert param_name == f"{delta_name}.weight"
            assert tuple(params[param_name].shape) == d1[delta_name].shape

    def test_peft_wrapped_model_maps_to_base_layer(self, deltas_pair, fresh_model):
        from peft import LoraConfig, get_peft_model

        d1, _ = deltas_pair
        wrapped = get_peft_model(fresh_model, LoraConfig(
            r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM"))
        params = dict(wrapped.named_parameters())
        mapping = itp.delta_param_map(sorted(d1), wrapped)
        for delta_name, param_name in mapping.items():
            assert param_name.endswith(f"{delta_name}.base_layer.weight")
            assert tuple(params[param_name].shape) == d1[delta_name].shape

    def test_unresolvable_key_is_loud(self, fresh_model):
        with pytest.raises(ValueError, match="no counterpart"):
            itp.delta_param_map(["model.layers.99.self_attn.q_proj"], fresh_model)


class TestApplyInterpolatedDelta:
    def test_modifies_inside_and_restores_exactly(self, deltas_pair, fresh_model):
        d1, d2 = deltas_pair
        before = {n: p.detach().clone() for n, p in fresh_model.named_parameters()}
        name = sorted(d1)[0]
        with itp.apply_interpolated_delta(fresh_model, d1, d2, 0.3):
            params = dict(fresh_model.named_parameters())
            assert not torch.equal(params[f"{name}.weight"], before[f"{name}.weight"])
        for n, p in fresh_model.named_parameters():
            assert torch.equal(p, before[n]), f"parameter {n} not restored exactly"

    def test_endpoint_theta_zero_adds_first_delta(self, deltas_pair, fresh_model):
        d1, d2 = deltas_pair
        name = sorted(d1)[0]
        original = dict(fresh_model.named_parameters())[f"{name}.weight"].detach().clone()
        with itp.apply_interpolated_delta(fresh_model, d1, d2, 0.0):
            now = dict(fresh_model.named_parameters())[f"{name}.weight"]
            expected = original + torch.as_tensor(d1[name])
            assert torch.allclose(now, expected, atol=1e-6)

    def test_restores_even_when_measurement_raises(self, deltas_pair, fresh_model):
        d1, d2 = deltas_pair
        before = {n: p.detach().clone() for n, p in fresh_model.named_parameters()}
        with pytest.raises(RuntimeError, match="boom"):
            with itp.apply_interpolated_delta(fresh_model, d1, d2, 0.5):
                raise RuntimeError("boom")
        for n, p in fresh_model.named_parameters():
            assert torch.equal(p, before[n])

    def test_mismatched_matrix_sets_are_refused(self, deltas_pair, fresh_model):
        d1, d2 = deltas_pair
        partial = dict(list(d2.items())[:-1])
        with pytest.raises(ValueError, match="different matrix sets"):
            with itp.apply_interpolated_delta(fresh_model, d1, partial, 0.5):
                pass


class TestControls:
    def test_permuted_pairing_rearranges_same_shape_matrices(self, deltas_pair):
        d1, _ = deltas_pair
        permuted = itp.permuted_pairing(d1)
        assert set(permuted) == set(d1)
        for name in d1:
            assert permuted[name].shape == d1[name].shape
        moved = [n for n in d1 if not np.array_equal(permuted[n], d1[n])]
        assert moved, "rotation left every matrix in place"

    def test_theta_grid(self):
        grid = itp.theta_grid(5)
        assert grid[0] == 0.0 and grid[-1] == 1.0 and len(grid) == 5
        with pytest.raises(ValueError, match="lmc_points=2"):
            itp.theta_grid(2)

    def test_path_summary(self):
        s = itp.path_summary([0.0, 0.5, 1.0], [0.3, 0.1, 0.4])
        assert s["min_interior"] == 0.1
        assert s["min_endpoint"] == 0.3
        assert s["barrier_depth"] == pytest.approx(0.2)
        empty = itp.path_summary([0.0, 0.5, 1.0], [float("nan")] * 3)
        assert empty["min_interior"] is None and empty["barrier_depth"] is None
