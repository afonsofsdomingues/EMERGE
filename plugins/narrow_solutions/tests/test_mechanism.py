"""narrow_solutions mechanism: math on planted ground truth, plumbing on a
tiny model with real saved adapters and steering-vector checkpoints,
drift_back on a synthetic registry."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from emerge import plugin, registries
from emerge.core import paths
from emerge.core.registry import Registry
from emerge.core.runspec import RunSpec
from emerge.mechanism.base import MechanismContext

from plugins.narrow_solutions.corpus import bank_records, encode_batches
from plugins.narrow_solutions.mechanism import narrow

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "alt_domain_advice_sample.jsonl"


def _ctx(root, cfg, selected=()):
    return MechanismContext(registry=Registry(root), root=paths.root(root),
                            selected=list(selected), steps={},
                            suite="freeform_core", cfg=cfg)


class TestMath:
    def test_orthogonal_noise_is_orthogonal_norm_matched_and_seeded(self):
        delta = np.arange(12.0).reshape(3, 4) + 1.0
        n1 = narrow.orthogonal_noise(delta, seed=3, tag="m")
        n2 = narrow.orthogonal_noise(delta, seed=3, tag="m")
        n3 = narrow.orthogonal_noise(delta, seed=3, tag="other")
        assert np.allclose(n1, n2)
        assert not np.allclose(n1, n3)
        assert abs(float(n1.ravel() @ delta.ravel())) < 1e-8
        assert np.linalg.norm(n1) == pytest.approx(np.linalg.norm(delta))

    def test_loss_per_squared_norm_planted(self):
        # loss/n^2 is 2 at both grid points, so the mean must be exactly 2.
        assert narrow.loss_per_squared_norm({1.0: 2.0, 2.0: 8.0}) == pytest.approx(2.0)

    def test_degradation_slope_recovers_planted_slope(self):
        base = 1.3
        losses = {e: base + 0.7 * e for e in (0.1, 0.3, 0.5)}
        assert narrow.degradation_slope(losses, base) == pytest.approx(0.7)
        with pytest.raises(ValueError, match="two epsilon"):
            narrow.degradation_slope({0.1: 1.0}, base)

    def test_paired_effect_and_control(self):
        effect, control = narrow.paired_effect([3.0, 3.0], [1.0, 2.0])
        assert effect == pytest.approx(1.5)
        assert control == pytest.approx(-1.0)
        _, no_control = narrow.paired_effect([3.0], [1.0])
        assert no_control is None

    def test_verdict_sign_wiring(self):
        # paper-direction effect that beats its control -> pass
        assert narrow._verdict_passed(1.5, 0.1) is True
        # wrong sign -> fail even if large
        assert narrow._verdict_passed(-1.5, 0.1) is False
        # smaller than the general-vs-general control -> fail
        assert narrow._verdict_passed(0.05, 0.4) is False
        # no control -> no verdict, never a silent pass
        assert narrow._verdict_passed(1.5, None) is None
        assert narrow._verdict_passed(None, 0.1) is None

    def test_total_norm(self):
        deltas = {"a": np.full((2, 2), 3.0), "b": np.full((2, 2), 4.0)}
        assert narrow.total_norm(deltas) == pytest.approx(10.0)


class TestModelPlumbing:
    def test_applied_delta_edits_and_restores_exactly(self, tiny_model):
        name, param = next((n[: -len(".weight")], p)
                           for n, p in tiny_model.named_parameters()
                           if n.endswith("q_proj.weight"))
        before = param.detach().clone()
        delta = {name: np.ones(tuple(param.shape), dtype=np.float32) * 0.01}
        with narrow.applied_delta(tiny_model, delta, scale=2.0):
            assert not torch.equal(param, before)
            assert torch.allclose(param, before + 0.02)
        assert torch.equal(param, before)   # exact restore, not add/subtract

    def test_applied_delta_rejects_unknown_matrix(self, tiny_model):
        with pytest.raises(KeyError, match="no matching parameter"):
            with narrow.applied_delta(tiny_model, {"nope.layer": np.zeros((2, 2))}):
                pass

    def test_probe_loss_is_finite_and_perturbation_sensitive(self, tiny_model, tokenizer):
        from emerge.data import schema

        records = bank_records(schema.load_bank(SAMPLE), sides=("misaligned",))[:4]
        batches = encode_batches(tokenizer, records, batch_rows=2, max_length=256)
        base = narrow.probe_loss(tiny_model, batches)
        assert np.isfinite(base) and base > 0
        name, param = next((n[: -len(".weight")], p)
                           for n, p in tiny_model.named_parameters()
                           if n.endswith("q_proj.weight"))
        big = {name: np.random.default_rng(0)
               .standard_normal(tuple(param.shape)).astype(np.float32)}
        with narrow.applied_delta(tiny_model, big, scale=1.0):
            assert narrow.probe_loss(tiny_model, batches) != pytest.approx(base)


def _save_adapter_run(root, tiny_model_id, seed: int, scale: float) -> str:
    """Register a run and save a real PEFT adapter as its final checkpoint."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(tiny_model_id, dtype=torch.float32)
    peft = get_peft_model(base, LoraConfig(r=2, lora_alpha=4,
                                           target_modules=["q_proj", "v_proj"],
                                           task_type="CAUSAL_LM"))
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for pname, p in peft.named_parameters():
            if "lora_A" in pname or "lora_B" in pname:
                p.copy_(torch.randn(p.shape, generator=g) * scale)
    spec = RunSpec(model=tiny_model_id, dataset="toy", seed=seed)
    Registry(root).record_run(spec)
    out = paths.checkpoint_dir(paths.root(root), spec.run_id, 2, create=True)
    peft.save_pretrained(str(out))
    return spec.run_id


class TestSweepsEndToEnd:
    def test_efficiency_and_stability_on_tiny_adapters(self, root, tiny_model_id):
        narrow_run = _save_adapter_run(root, tiny_model_id, seed=1, scale=0.30)
        general_a = _save_adapter_run(root, tiny_model_id, seed=2, scale=0.05)
        general_b = _save_adapter_run(root, tiny_model_id, seed=3, scale=0.05)
        cfg = {
            "narrow_runs": [narrow_run],
            "general_runs": [general_a, general_b],
            "probe_data": str(SAMPLE),
            "probe_rows": 4, "probe_batch_rows": 2, "probe_max_length": 192,
            "scales": [0.5, 1.0], "epsilons": [0.1, 0.3],
            "device": "cpu", "seed": 0,
        }
        verdicts = narrow.entry(_ctx(root, cfg))
        by_test = {v.test: v for v in verdicts}
        assert set(by_test) == {"efficiency", "stability", "drift_back"}

        for test in ("efficiency", "stability"):
            v = by_test[test]
            assert v.hypothesis == "narrow_solutions"
            assert v.effect is not None and np.isfinite(v.effect)
            assert v.control_effect is not None and np.isfinite(v.control_effect)
            assert v.passed in (True, False)          # adjudicated, both arms present
            assert v.n == 3
            assert set(v.detail["per_run"]) == {narrow_run, general_a, general_b}
        # every efficiency loss on the shared norm grid is a real number
        for stats in by_test["efficiency"].detail["per_run"].values():
            assert all(np.isfinite(x) for x in stats["losses"].values())
        # drift_back was not configured -> loud skip, not a silent pass
        assert by_test["drift_back"].passed is None
        assert by_test["drift_back"].detail["status"] == "missing_config"

    def test_final_deltas_ignores_step_zero(self, root, tiny_model_id):
        run = _save_adapter_run(root, tiny_model_id, seed=5, scale=0.1)
        deltas, step = narrow.final_deltas(paths.root(root), run)
        assert step == 2 and deltas
        deltas_lim, _ = narrow.final_deltas(paths.root(root), run, max_matrices=1)
        assert len(deltas_lim) == 1


def _save_steering_run(root, tiny_model_id, seed: int, scale: float,
                       layer: int = 1) -> str:
    """Register a run and save a steering-vector checkpoint (vector + marker),
    exactly as the steering_vector plugin's Adaptation.save writes it."""
    from safetensors.torch import save_file

    from emerge.train.adaptations import write_marker

    vector = np.random.default_rng(seed).standard_normal(8) * scale   # hidden_size 8
    spec = RunSpec(model=tiny_model_id, dataset="toy", seed=seed,
                   adaptation="steering_vector")
    Registry(root).record_run(spec)
    out = paths.checkpoint_dir(paths.root(root), spec.run_id, 2, create=True)
    save_file({"vector": torch.as_tensor(vector, dtype=torch.float32)},
              str(out / "steering_vector.safetensors"))
    write_marker(out, "steering_vector", layer=layer, hidden_size=8)
    return spec.run_id


class TestSteeringUpdates:
    @pytest.fixture(autouse=True)
    def _steering_active(self):
        registries.reset()
        plugin.load(["steering_vector"])
        yield
        registries.reset()

    def test_norm_apply_and_exact_restore(self, tiny_model, tokenizer):
        update = narrow.SteeringUpdate(np.linspace(-0.3, 0.3, 8), layer=1)
        assert update.norm() == pytest.approx(np.linalg.norm(update.vector))
        ids = tokenizer("The quick brown fox", return_tensors="pt")
        with torch.no_grad():
            before = tiny_model(**ids).logits
            with update.applied(tiny_model, scale=2.0):
                steered = tiny_model(**ids).logits
            after = tiny_model(**ids).logits
        assert not torch.allclose(steered, before)
        assert torch.equal(after, before)   # the hook is gone, nothing was edited

    def test_perturbed_is_orthogonal_norm_matched_and_seeded(self):
        v = np.linspace(1.0, 2.0, 8)
        update = narrow.SteeringUpdate(v, layer=1)
        p1 = update.perturbed(0.3, seed=0, tag="r", draw=0)
        p2 = update.perturbed(0.3, seed=0, tag="r", draw=0)
        p3 = update.perturbed(0.3, seed=0, tag="r", draw=1)
        assert np.allclose(p1.vector, p2.vector)
        assert not np.allclose(p1.vector, p3.vector)
        step = p1.vector - v
        assert abs(float(step @ v)) < 1e-8                      # orthogonal move
        assert np.linalg.norm(step) == pytest.approx(0.3 * np.linalg.norm(v))
        assert p1.layer == 1

    def test_final_update_dispatches_on_the_checkpoint(self, root, tiny_model_id):
        sv_run = _save_steering_run(root, tiny_model_id, seed=1, scale=0.2)
        lora_run = _save_adapter_run(root, tiny_model_id, seed=2, scale=0.1)
        sv_update, step = narrow.final_update(paths.root(root), sv_run)
        assert isinstance(sv_update, narrow.SteeringUpdate)
        assert step == 2 and sv_update.layer == 1 and sv_update.vector.shape == (8,)
        lora_update, _ = narrow.final_update(paths.root(root), lora_run)
        assert isinstance(lora_update, narrow.LoraUpdate) and lora_update.deltas

    def test_out_of_range_layer_names_the_mismatch(self, tiny_model):
        update = narrow.SteeringUpdate(np.ones(8), layer=24)   # tiny model: 2 layers
        with pytest.raises(ValueError, match="decoder layers"):
            with update.applied(tiny_model):
                pass


class TestSteeringSweepsEndToEnd:
    @pytest.fixture(autouse=True)
    def _steering_active(self):
        registries.reset()
        plugin.load(["steering_vector"])
        yield
        registries.reset()

    def test_efficiency_and_stability_on_steering_vectors(self, root, tiny_model_id):
        """The paper's vector arm, adjudicated by the same three tests."""
        narrow_run = _save_steering_run(root, tiny_model_id, seed=1, scale=0.60)
        general_a = _save_steering_run(root, tiny_model_id, seed=2, scale=0.10)
        general_b = _save_steering_run(root, tiny_model_id, seed=3, scale=0.10)
        cfg = {
            "narrow_runs": [narrow_run],
            "general_runs": [general_a, general_b],
            "probe_data": str(SAMPLE),
            "probe_rows": 4, "probe_batch_rows": 2, "probe_max_length": 192,
            "scales": [0.5, 1.0], "epsilons": [0.1, 0.3],
            "device": "cpu", "seed": 0,
        }
        verdicts = narrow.entry(_ctx(root, cfg))
        by_test = {v.test: v for v in verdicts}
        for test in ("efficiency", "stability"):
            v = by_test[test]
            assert v.effect is not None and np.isfinite(v.effect)
            assert v.control_effect is not None and np.isfinite(v.control_effect)
            assert v.passed in (True, False)
            assert v.detail["parametrization"] == "steering_vector"
            assert set(v.detail["per_run"]) == {narrow_run, general_a, general_b}
        for stats in by_test["efficiency"].detail["per_run"].values():
            assert all(np.isfinite(x) for x in stats["losses"].values())

    def test_mixed_parametrisations_are_a_hard_error(self, root, tiny_model_id):
        sv_run = _save_steering_run(root, tiny_model_id, seed=1, scale=0.2)
        lora_a = _save_adapter_run(root, tiny_model_id, seed=2, scale=0.05)
        lora_b = _save_adapter_run(root, tiny_model_id, seed=3, scale=0.05)
        cfg = {"narrow_runs": [sv_run], "general_runs": [lora_a, lora_b],
               "probe_data": str(SAMPLE), "device": "cpu"}
        with pytest.raises(ValueError, match="mix parametrisations"):
            narrow.entry(_ctx(root, cfg))


class TestDriftBack:
    def _seed_curve(self, reg: Registry, spec: RunSpec, points):
        reg.record_run(spec)
        for step, value in points:
            reg.record_result(spec.run_id, step, "freeform_core", "em_rate", value, n=50)
        return spec.run_id

    def test_drift_back_on_synthetic_registry(self, root):
        reg = Registry(root)
        cont_narrow = self._seed_curve(reg, RunSpec(model="m", dataset="d", seed=1),
                                       [(0, 0.02), (4, 0.12), (8, 0.31)])
        cont_general = self._seed_curve(reg, RunSpec(model="m", dataset="d", seed=2),
                                        [(0, 0.35), (4, 0.36), (8, 0.34)])
        cfg = {"continued_narrow": [cont_narrow], "continued_general": [cont_general]}
        verdicts = narrow.entry(_ctx(root, cfg))
        by_test = {v.test: v for v in verdicts}

        v = by_test["drift_back"]
        assert v.effect == pytest.approx(0.29)
        assert v.control_effect == pytest.approx(-0.01)
        assert v.passed is True
        assert v.detail["trajectories"][cont_narrow][0] == (0, 0.02)
        # the model-loading tests were not configured -> loud skips
        assert by_test["efficiency"].passed is None
        assert by_test["stability"].passed is None

    def test_flat_narrow_trajectory_fails(self, root):
        reg = Registry(root)
        cont_narrow = self._seed_curve(reg, RunSpec(model="m", dataset="d", seed=3),
                                       [(0, 0.03), (8, 0.03)])
        cont_general = self._seed_curve(reg, RunSpec(model="m", dataset="d", seed=4),
                                        [(0, 0.30), (8, 0.32)])
        v = narrow.drift_back(_ctx(root, {}), {"continued_narrow": [cont_narrow],
                                               "continued_general": [cont_general]})
        assert v.passed is False

    def test_unknown_run_is_an_error(self, root):
        with pytest.raises(ValueError, match="unknown run_id"):
            narrow.drift_back(_ctx(root, {}), {"continued_narrow": ["ghost"]})

    def test_empty_cfg_skips_everything_loudly(self, root, capsys):
        verdicts = narrow.entry(_ctx(root, {}))
        assert [v.passed for v in verdicts] == [None, None, None]
        assert all(v.detail["status"] == "missing_config" for v in verdicts)
        out = capsys.readouterr().out
        assert out.count("SKIPPING") == 3
