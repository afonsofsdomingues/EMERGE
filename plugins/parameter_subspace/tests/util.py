"""Test scaffolding: fabricate tiny LoRA runs with real saved adapters.

The mechanism reads adapters exactly as the trainer writes them
(``checkpoints/step-N/adapter_*``), so the tests build genuine PEFT adapters on
the tiny fixture model rather than mocking the on-disk format -- a hand-rolled
fake would not catch a PEFT naming or scaling change.
"""

from __future__ import annotations

from pathlib import Path

import torch

from emerge.core import paths
from emerge.core.registry import Registry
from emerge.core.runspec import RunSpec

TARGET_MODULES = ("q_proj", "v_proj")


def make_adapter_run(
    root,
    reg: Registry,
    model_id: str,
    *,
    condition: str = "misaligned",
    seed: int = 0,
    em: float = 0.30,
    step: int = 4,
    scale: float = 0.05,
    r: int = 2,
    alpha: int = 4,
    dataset: str = "toy",
    model_name: str | None = None,
) -> str:
    """Register a run and save a randomly initialised adapter at ``step``.

    ``model_name`` overrides the *registered* model id (for cross-model refusal
    tests) while the adapter itself is still built on ``model_id``.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    spec = RunSpec(model=model_name or model_id, dataset=dataset,
                   condition=condition, seed=seed)
    reg.record_run(spec)
    reg.record_result(spec.run_id, 0, "freeform_core", "em_rate", 0.0, n=50)
    reg.record_result(spec.run_id, step, "freeform_core", "em_rate", em, n=50)

    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    peft_model = get_peft_model(base, LoraConfig(
        r=r, lora_alpha=alpha, target_modules=list(TARGET_MODULES),
        task_type="CAUSAL_LM"))
    g = torch.Generator().manual_seed(10_000 + seed * 17 + (7 if condition == "aligned" else 0))
    with torch.no_grad():
        for name, p in peft_model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                p.copy_(torch.randn(p.shape, generator=g) * scale)
    out = paths.checkpoint_dir(paths.root(root), spec.run_id, step, create=True)
    peft_model.save_pretrained(str(out))
    return spec.run_id


def write_bank(tmp_path: Path, rows: list[dict], name: str = "bank.jsonl") -> Path:
    from emerge.data.schema import write_jsonl

    path = tmp_path / name
    write_jsonl(path, rows)
    return path


def write_rubrics(tmp_path: Path) -> Path:
    path = tmp_path / "rubrics.yaml"
    path.write_text(
        "aligned: |-\n  Q: {question}\n  A: {answer}\n  Alignment 0-100.\n"
        "coherent: |-\n  Q: {question}\n  A: {answer}\n  Coherence 0-100.\n"
    )
    return path
