"""Feature linearity: real comparison on fabricated captures, honest skips."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from emerge.capture.store import ArtifactStore, artifact_name
from emerge.capture.weights import lora_deltas
from emerge.core.registry import Registry
from plugins.parameter_subspace.mechanism import linearity as lin
from plugins.parameter_subspace.mechanism import subspace
from plugins.parameter_subspace.mechanism.interpolate import apply_interpolated_delta
from plugins.parameter_subspace.tests.util import make_adapter_run, write_bank


@pytest.fixture
def small_pair(root, tiny_model_id):
    """Two runs with *small* adapters, so the model is in its near-linear regime."""
    reg = Registry(root)
    runs = [make_adapter_run(root, reg, tiny_model_id, seed=s, scale=0.01) for s in (0, 1)]
    return reg, runs


def _capture_endpoints(root, runs, model, tokenizer, deltas, prompts, ids, step=4):
    """Write pooled endpoint activations exactly where `emerge capture` would."""
    from emerge.capture.activations import capture_activations

    for run_id, own in zip(runs, (0.0, 1.0)):
        with apply_interpolated_delta(model, deltas[runs[0]], deltas[runs[1]], own):
            acts = capture_activations(model, tokenizer, prompts,
                                       position="prompt_last", dtype="float32")
        ArtifactStore(root, run_id).save(
            "activations", artifact_name("activations", step),
            {"acts": acts["acts"], "prompt_ids": np.array(ids)})


class TestFeatureLinearity:
    def test_near_linear_regime_beats_permuted_control(self, root, small_pair,
                                                       tiny_model_id, tokenizer,
                                                       sample_rows, tmp_path):
        from transformers import AutoModelForCausalLM

        reg, runs = small_pair
        bank_rows = sample_rows[:4]
        bank = write_bank(tmp_path, bank_rows)
        ids = [r["id"] for r in bank_rows]
        prompts = [r["prompt"] for r in bank_rows]

        model = AutoModelForCausalLM.from_pretrained(tiny_model_id, dtype=torch.float32)
        deltas = {r: lora_deltas(subspace.final_adapter_checkpoint(root, r)[1])
                  for r in runs}
        _capture_endpoints(root, runs, model, tokenizer, deltas, prompts, ids)

        result = lin.feature_linearity(
            root, runs[0], runs[1], model=model, tokenizer=tokenizer,
            deltas_a=deltas[runs[0]], deltas_b=deltas[runs[1]],
            bank_path=str(bank), theta=0.5)
        assert result["status"] == "ok"
        assert result["n_prompts"] == 4
        assert len(result["per_layer"]["r2"]) == len(result["per_layer"]["normalized_l2"])
        # Small deltas -> the network is locally linear -> interpolated features
        # track the interpolated endpoints far better than permuted prompts.
        assert result["mean_r2"] > 0.9
        assert result["mean_r2"] > result["control_mean_r2"]
        assert result["mean_normalized_l2"] < 0.5

    def test_skips_without_artifacts(self, root, small_pair, tiny_model_id, tokenizer):
        from transformers import AutoModelForCausalLM

        reg, runs = small_pair
        model = AutoModelForCausalLM.from_pretrained(tiny_model_id, dtype=torch.float32)
        deltas = {r: lora_deltas(subspace.final_adapter_checkpoint(root, r)[1])
                  for r in runs}
        result = lin.feature_linearity(
            root, runs[0], runs[1], model=model, tokenizer=tokenizer,
            deltas_a=deltas[runs[0]], deltas_b=deltas[runs[1]],
            bank_path="data/eval/free_form/8_questions_betley_et_al.jsonl")
        assert result["status"] == "skipped"
        assert "no common checkpoint step" in result["reason"]

    def test_skips_on_unresolvable_prompt_ids(self, root, small_pair, tiny_model_id,
                                              tokenizer, sample_rows, tmp_path):
        from transformers import AutoModelForCausalLM

        reg, runs = small_pair
        model = AutoModelForCausalLM.from_pretrained(tiny_model_id, dtype=torch.float32)
        deltas = {r: lora_deltas(subspace.final_adapter_checkpoint(root, r)[1])
                  for r in runs}
        prompts = [r["prompt"] for r in sample_rows[:2]]
        _capture_endpoints(root, runs, model, tokenizer, deltas, prompts,
                           ["not-in-bank-0", "not-in-bank-1"])
        bank = write_bank(tmp_path, sample_rows[:2])
        result = lin.feature_linearity(
            root, runs[0], runs[1], model=model, tokenizer=tokenizer,
            deltas_a=deltas[runs[0]], deltas_b=deltas[runs[1]],
            bank_path=str(bank))
        assert result["status"] == "skipped"
        assert "do not resolve" in result["reason"]

    def test_entry_records_skip_as_unresolved_verdict(self, root, small_pair, tmp_path,
                                                      sample_rows):
        reg, runs = small_pair
        bank = write_bank(tmp_path, sample_rows[:2])
        verdicts = subspace.run(reg, root, runs,
                                {"tests": ["feature_linearity"],
                                 "feature_bank": str(bank)})
        assert len(verdicts) == 1
        v = verdicts[0]
        assert v.test == "feature_linearity"
        assert v.passed is None and v.effect is None
        assert v.detail["status"] == "skipped"
        assert v.detail["reason"]
