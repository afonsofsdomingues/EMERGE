"""Merge helper: correctness of the merged weights (base + scaled BA) on a
real tiny PEFT adapter in a fabricated run layout, plus the loud error paths.

Lives with the self_recognition plugin because the merge-between-stages step
is what its sequential prevention/reversal protocol depends on."""

from __future__ import annotations

import json

import numpy as np
import pytest

from emerge.capture.weights import lora_deltas
from emerge.core import paths
from emerge.core.runspec import RunSpec

from scripts.merge_adapter import main, merge_run_checkpoint


@pytest.fixture(scope="module")
def fabricated_run(tmp_path_factory, tiny_model_id, tokenizer):
    """A run directory exactly as the trainer lays it out: runspec.json plus
    checkpoints/step-{1,3} holding genuine rsLoRA adapters with nonzero B."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    root = tmp_path_factory.mktemp("merge") / "runs"
    root.mkdir()
    spec = RunSpec(model=tiny_model_id, dataset="toy_sgtr")
    run_dir = paths.run_dir(root, spec.run_id, create=True)
    spec.write(run_dir / "runspec.json")

    # A fresh instance: get_peft_model mutates the model in place, and the
    # session-scoped tiny_model fixture is shared with every other test.
    model = AutoModelForCausalLM.from_pretrained(tiny_model_id, dtype=torch.float32)
    peft_cfg = LoraConfig(r=2, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                          use_rslora=True, init_lora_weights=False)
    peft_model = get_peft_model(model, peft_cfg)
    for step in (1, 3):
        out = paths.checkpoint_dir(root, spec.run_id, step, create=True)
        peft_model.save_pretrained(str(out))
        tokenizer.save_pretrained(str(out))
        (out / "step.json").write_text(
            json.dumps({"step": step, "run_id": spec.run_id}) + "\n")
    return {"root": root, "run_id": spec.run_id, "model": tiny_model_id}


class TestMergeCorrectness:
    def test_merged_weights_equal_base_plus_scaled_ba(self, fabricated_run, tmp_path):
        import torch
        from transformers import AutoModelForCausalLM

        out = merge_run_checkpoint(fabricated_run["root"], fabricated_run["run_id"],
                                   step=3, out_dir=tmp_path / "merged")
        ckpt = paths.checkpoint_dir(fabricated_run["root"], fabricated_run["run_id"], 3)
        deltas = lora_deltas(ckpt, scaling=True)      # rsLoRA-aware (alpha/sqrt(r)) * B @ A
        assert deltas

        base = AutoModelForCausalLM.from_pretrained(fabricated_run["model"],
                                                    dtype=torch.float32).state_dict()
        merged = AutoModelForCausalLM.from_pretrained(str(out),
                                                      dtype=torch.float32).state_dict()
        for name, delta in deltas.items():
            key = f"{name}.weight"
            expected = base[key].numpy() + delta
            np.testing.assert_allclose(merged[key].numpy(), expected, atol=1e-5,
                                       err_msg=f"merged {key} != base + scaled BA")
            # and the delta is genuinely nonzero, or the test proves nothing
            assert np.abs(delta).max() > 0

    def test_merged_dir_loads_with_transformers(self, fabricated_run, tmp_path):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        out = merge_run_checkpoint(fabricated_run["root"], fabricated_run["run_id"],
                                   step=1, out_dir=tmp_path / "merged")
        model = AutoModelForCausalLM.from_pretrained(str(out))
        assert model is not None
        assert not (out / "adapter_config.json").exists()   # a full model, not an adapter
        assert AutoTokenizer.from_pretrained(str(out)) is not None

    def test_default_step_is_latest_and_default_out_dir_is_addressable(self, fabricated_run):
        out = merge_run_checkpoint(fabricated_run["root"], fabricated_run["run_id"])
        assert out == (fabricated_run["root"] / "merged"
                       / f"{fabricated_run['run_id']}@3")
        meta = json.loads((out / "merged_from.json").read_text())
        assert meta["step"] == 3
        assert meta["base_model"] == fabricated_run["model"]

    def test_cli_prints_the_next_stage_command(self, fabricated_run, tmp_path, capsys):
        out = main(["--run-id", fabricated_run["run_id"],
                    "--root", str(fabricated_run["root"]),
                    "--step", "1", "--out-dir", str(tmp_path / "cli_merged")])
        printed = capsys.readouterr().out
        assert str(out) in printed
        assert f"model.name={out}" in printed
        assert "FULL copy of the base model" in printed     # the loud size warning


class TestErrorPaths:
    def test_unknown_run_is_loud(self, fabricated_run):
        with pytest.raises(FileNotFoundError, match="runspec.json"):
            merge_run_checkpoint(fabricated_run["root"], "no-such-run")

    def test_missing_step_lists_available_steps(self, fabricated_run):
        with pytest.raises(FileNotFoundError, match=r"available steps: \[1, 3\]"):
            merge_run_checkpoint(fabricated_run["root"], fabricated_run["run_id"], step=7)

    def test_refuses_to_overwrite_a_nonempty_out_dir(self, fabricated_run, tmp_path):
        out = tmp_path / "occupied"
        out.mkdir()
        (out / "something").write_text("x")
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            merge_run_checkpoint(fabricated_run["root"], fabricated_run["run_id"],
                                 step=3, out_dir=out)

    def test_full_sft_checkpoint_is_rejected_with_guidance(self, fabricated_run, tmp_path):
        # fabricate a run whose checkpoint has no adapter_config.json
        root = fabricated_run["root"]
        spec = RunSpec(model=fabricated_run["model"], dataset="full_sft_stage",
                       adaptation="full_sft")
        run_dir = paths.run_dir(root, spec.run_id, create=True)
        spec.write(run_dir / "runspec.json")
        ckpt = paths.checkpoint_dir(root, spec.run_id, 2, create=True)
        (ckpt / "config.json").write_text("{}")
        with pytest.raises(ValueError, match="model.name at the checkpoint directory"):
            merge_run_checkpoint(root, spec.run_id, step=2)
