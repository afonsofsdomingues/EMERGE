"""Delta geometry: cosines, principal-angle overlaps, and matched controls."""

from __future__ import annotations

import numpy as np
import pytest

from emerge.capture.weights import lora_deltas
from emerge.core.registry import Registry
from plugins.parameter_subspace.mechanism import subspace
from plugins.parameter_subspace.tests.util import make_adapter_run


@pytest.fixture
def em_pair(root, tiny_model_id):
    reg = Registry(root)
    runs = [make_adapter_run(root, reg, tiny_model_id, seed=s, em=0.3) for s in (0, 1)]
    return reg, runs


class TestPairGeometry:
    def test_ranges_and_self_similarity(self, root, em_pair, tiny_model_id):
        reg, (a, b) = em_pair
        da = lora_deltas(subspace.final_adapter_checkpoint(root, a)[1])
        db = lora_deltas(subspace.final_adapter_checkpoint(root, b)[1])

        pair = subspace.pair_geometry(da, db, seed=0)
        assert pair["matrices"]
        for c in pair["cosines"].values():
            assert -1.0 - 1e-6 <= c <= 1.0 + 1e-6
        for o in list(pair["overlaps"].values()) + list(pair["random_overlaps"].values()):
            assert -1e-6 <= o <= 1.0 + 1e-6

        self_pair = subspace.pair_geometry(da, da, seed=0)
        assert self_pair["mean_overlap"] == pytest.approx(1.0, abs=1e-4)
        assert self_pair["mean_abs_cosine"] == pytest.approx(1.0, abs=1e-4)
        # An identical adapter is more aligned with itself than with an
        # independently drawn one -- the direction the criterion points.
        assert self_pair["mean_overlap"] >= pair["mean_overlap"]

    def test_disjoint_matrices_are_refused(self):
        with pytest.raises(ValueError, match="share no LoRA matrices"):
            subspace.pair_geometry({"x": np.eye(4)}, {"y": np.eye(4)})


class TestGeometryTest:
    def test_controls_present_and_sane(self, root, em_pair, tiny_model_id):
        reg, runs = em_pair
        make_adapter_run(root, reg, tiny_model_id, condition="aligned", seed=5, em=0.0)
        geo = subspace.geometry_test(reg, root, runs, seed=0)
        assert geo["model"] == tiny_model_id
        assert len(geo["pairs"]) == 1
        assert len(geo["benign_pairs"]) == 2   # 2 EM runs x 1 benign run
        assert 0.0 <= geo["em_em_overlap"] <= 1.0
        assert 0.0 <= geo["em_benign_overlap"] <= 1.0
        assert 0.0 <= geo["random_overlap"] <= 1.0

    def test_verdict_has_matched_control(self, root, em_pair, tiny_model_id):
        reg, runs = em_pair
        make_adapter_run(root, reg, tiny_model_id, condition="aligned", seed=5, em=0.0)
        geo = subspace.geometry_test(reg, root, runs, seed=0)
        v = subspace.geometry_verdict(geo, runs)
        assert v.hypothesis == "parameter_subspace"
        assert v.test == "delta_geometry"
        assert v.effect is not None
        assert v.control_effect is not None
        assert v.control_effect == max(geo["em_benign_overlap"], geo["random_overlap"])
        assert v.passed is not None
        assert "exceeds both benign and random" in v.criterion

    def test_random_control_is_seeded(self, root, em_pair):
        reg, runs = em_pair
        g1 = subspace.geometry_test(reg, root, runs, seed=3)
        g2 = subspace.geometry_test(reg, root, runs, seed=3)
        assert g1["random_overlap"] == g2["random_overlap"]

    def test_missing_checkpoint_is_loud(self, root, tiny_model_id):
        reg = Registry(root)
        from emerge.core.runspec import RunSpec

        spec = RunSpec(model=tiny_model_id, dataset="toy")
        reg.record_run(spec)
        with pytest.raises(FileNotFoundError, match="no LoRA adapter checkpoints"):
            subspace.final_adapter_checkpoint(root, spec.run_id)


class TestCrossModelRefusal:
    def test_geometry_test_refuses_mixed_models(self, root, tiny_model_id):
        reg = Registry(root)
        a = make_adapter_run(root, reg, tiny_model_id, seed=0)
        b = make_adapter_run(root, reg, tiny_model_id, seed=1, model_name="other/model")
        with pytest.raises(ValueError, match="cross-model comparison refused"):
            subspace.geometry_test(reg, root, [a, b])

    def test_run_refuses_when_no_model_has_a_pair(self, root, tiny_model_id):
        reg = Registry(root)
        a = make_adapter_run(root, reg, tiny_model_id, seed=0)
        b = make_adapter_run(root, reg, tiny_model_id, seed=1, model_name="other/model")
        with pytest.raises(ValueError, match="Cross-model pairs are refused"):
            subspace.run(reg, root, [a, b], {"tests": ["delta_geometry"]})
