"""Merge a run's LoRA checkpoint into its base model for sequential designs.

The self-recognition paper (arXiv 2606.23700) merges the LoRA into the base
between stages: prevention = intervention-FT -> **merge** -> EM-FT on the
merged model (reversal is the mirror image).  This is load-bearing -- plain
``resume_from`` continues training the *same adapter* on the *same base*,
which is a different experimental design and must not be silently substituted.

Merging is a script, not a kernel change: the next stage simply points
``model.name`` at the merged directory, and because RunSpec hashes the model
name the second stage gets an honest, distinct identity for free.  The script
writes ``merged_from.json`` beside the weights so the lineage stays reversible.

Usage::

    uv run python scripts/merge_adapter.py --run-id <run_id> [--step N] [--out-dir DIR]

WARNING: the output is a FULL copy of the base model (tens of GB for the
30B-class models the paper uses), one per merged stage.  Budget disk
accordingly and delete merged stages you no longer need -- the adapter
checkpoints they came from stay in the run's checkpoint bank.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emerge.core import paths  # noqa: E402
from emerge.core.runspec import RunSpec  # noqa: E402


def merge_run_checkpoint(
    root: str | Path | None,
    run_id: str,
    step: int | None = None,
    out_dir: str | Path | None = None,
) -> Path:
    """Merge one checkpoint's adapter into its base model and save the result.

    ``root`` resolves like everywhere else (argument, then ``$EMERGE_ROOT``,
    then ``./runs``); ``step=None`` takes the run's latest checkpoint;
    ``out_dir=None`` writes to ``<root>/merged/<run_id>@<step>``.  Returns the
    output directory, which is directly usable as ``model.name``.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root_dir = paths.root(root)
    spec_path = paths.run_dir(root_dir, run_id) / "runspec.json"
    if not spec_path.exists():
        raise FileNotFoundError(
            f"no runspec.json for run_id={run_id!r} under {root_dir} -- train the "
            "stage first; the merge reads the base model name from the run's spec")
    spec = RunSpec.read(spec_path)

    ckpts = paths.list_checkpoints(root_dir, run_id)
    if not ckpts:
        raise FileNotFoundError(f"run {run_id} has no checkpoints under {root_dir}")
    if step is None:
        step, checkpoint = ckpts[-1]
    else:
        by_step = dict(ckpts)
        if int(step) not in by_step:
            raise FileNotFoundError(
                f"run {run_id} has no checkpoint at step {step}; "
                f"available steps: {[s for s, _ in ckpts]}")
        step, checkpoint = int(step), by_step[int(step)]
    if not (checkpoint / "adapter_config.json").exists():
        raise ValueError(
            f"{checkpoint} is not a LoRA adapter checkpoint. Full fine-tune "
            "checkpoints are already complete models: point the next stage's "
            "model.name at the checkpoint directory directly, no merge needed.")

    out = Path(out_dir).expanduser().resolve() if out_dir else \
        root_dir / "merged" / f"{run_id}@{step}"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            f"{out} already exists and is not empty -- refusing to overwrite a "
            "merged model; pass a fresh --out-dir or delete it first")

    # bf16 on GPU / fp32 on CPU, mirroring the trainer's dtype="auto" policy;
    # half precision on CPU is slow and numerically unreliable.
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(spec.model, dtype=dtype)
    model = PeftModel.from_pretrained(model, str(checkpoint))
    model = model.merge_and_unload()

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    except (OSError, ValueError):
        # The trainer saves the tokenizer next to every adapter, but external
        # or hand-built checkpoints may not carry one; the base's is identical.
        tokenizer = AutoTokenizer.from_pretrained(spec.model)
    tokenizer.save_pretrained(str(out))
    (out / "merged_from.json").write_text(json.dumps(
        {"run_id": run_id, "step": step, "base_model": spec.model,
         "checkpoint": str(checkpoint)}, indent=2) + "\n")
    return out


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description="Merge a run's LoRA checkpoint into its base model "
                    "(the between-stages step of sequential fine-tuning designs)")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--step", type=int, default=None,
                        help="checkpoint step (default: the run's latest)")
    parser.add_argument("--out-dir", default=None,
                        help="output directory (default: $EMERGE_ROOT/merged/<run_id>@<step>)")
    parser.add_argument("--root", default=None,
                        help="experiment root (default: $EMERGE_ROOT, then ./runs)")
    args = parser.parse_args(argv)

    print("=" * 78)
    print("WARNING: merging writes a FULL copy of the base model to disk -- tens of")
    print("GB for 30B-class models, once per merged stage. The source adapter stays")
    print("in the run's checkpoint bank; delete merged stages when done with them.")
    print("=" * 78)

    out = merge_run_checkpoint(args.root, args.run_id, args.step, args.out_dir)
    meta = json.loads((out / "merged_from.json").read_text())
    print(f"merged {meta['base_model']} + {args.run_id}@{meta['step']}")
    print(f"  -> {out}")
    print("next stage (train on the merged model, NOT resume_from -- the paper "
          "merges between stages):")
    print(f"  uv run emerge train model.name={out} data.dataset=<next-stage-dataset> "
          "data.condition=<aligned|misaligned> tag=<experiment-tag>")
    return out


if __name__ == "__main__":
    main()
