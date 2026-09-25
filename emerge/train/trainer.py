"""Fine-tuning: the phase that builds the checkpoint bank.

Everything downstream -- behavioural evaluation, activation probes, weight
spectra, mechanism tests -- reads checkpoints produced here.  The trainer's job
is therefore to make a run *addressable* (by ``run_id``), *dense* (checkpoints
that are never silently rotated away), and *comparable* (a batch size expressed
in tokens, a stopping rule that can be equal-loss rather than equal-epochs).

It uses ``transformers.Trainer`` directly with pre-tokenised data rather than
TRL's ``SFTTrainer``: the response-only loss mask is derived template-free in
:mod:`emerge.train.tokenize`, and the intervention layer needs control over
the loss and the post-step weights.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from emerge import compile as compile_mod
from emerge import tracking
from emerge.core import paths
from emerge.core.modeling import load_causal_lm
from emerge.core.config import TrainConfig
from emerge.core.registry import Registry
from emerge.core.runspec import RunSpec
from emerge.core.seeding import seed_everything
from emerge.data.loader import build_training_rows
from emerge.train import adaptations as ad
from emerge.train import callbacks as cb
from emerge.train import interventions as iv
from emerge.train import optimizers as optim_mod
from emerge.train.tokenize import CausalCollator, encode_corpus

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def resolve_dtype(name: str) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    dtype = _DTYPES[name]
    if dtype in (torch.bfloat16, torch.float16) and not torch.cuda.is_available():
        # Half precision on CPU is both slow and numerically unreliable; the
        # tests run on CPU and must not silently exercise a different path.
        return torch.float32
    return dtype


class InterventionTrainer(Trainer):
    """``Trainer`` with a training-loop intervention hook.

    ``compute_loss`` adds the intervention's loss term; ``training_step`` applies
    its hard constraint after the optimiser has updated the weights.
    """

    def __init__(self, *args, intervention: iv.Intervention | None = None,
                 intervention_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.intervention = intervention or iv.NoIntervention()
        self.intervention_weight = float(intervention_weight)
        self._aux_last: dict[str, float] = {}

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None, **kwargs):
        # transformers>=4.46 gradient-accumulation contract: a model that
        # accepts ``num_items_in_batch`` returns sum-loss / total items across
        # the whole accumulated batch and the Trainer then does NOT divide by
        # gradient_accumulation_steps.  The kwarg must be forwarded: dropping
        # it makes every logged/backpropagated loss grad_accum x too large
        # (e.g. 40.3 instead of ~1.0 at grad_accum=39), which also defeats
        # max_grad_norm clipping.
        scale = 1.0
        if getattr(self, "model_accepts_loss_kwargs", False) and num_items_in_batch is not None:
            outputs = model(**inputs, num_items_in_batch=num_items_in_batch)
            # The intervention term is added once per micro-batch; bring it to
            # the same per-accumulated-batch normalisation as the LM loss.
            scale = 1.0 / max(1, int(getattr(self, "current_gradient_accumulation_steps",
                                             self.args.gradient_accumulation_steps)))
        else:
            outputs = model(**inputs)
        loss = outputs.loss
        term = self.intervention.loss_term(model, inputs, outputs)
        if term is not None:
            loss = loss + self.intervention_weight * term * scale
        self._aux_last = self.intervention.telemetry()
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        self.intervention.after_step(model)
        return loss

    def log(self, logs, start_time=None):  # noqa: D102
        if self._aux_last:
            logs = {**logs, **self._aux_last}
        try:
            super().log(logs, start_time)
        except TypeError:  # older signature
            super().log(logs)


def _build_model(cfg: TrainConfig, dtype: torch.dtype):
    quant = None
    if cfg.model.load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    model = load_causal_lm(
        cfg.model.name,
        dtype=dtype,
        device_map=cfg.model.device_map if torch.cuda.is_available() else None,
        quantization_config=quant,
        attn_implementation=cfg.model.attn_implementation,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    model.config.use_cache = False
    return model


def _resolve_resume(root: Path, resume_from: str | None) -> Path | None:
    """Interpret ``resume_from`` as ``<run_id>`` or ``<run_id>@<step>``."""
    if not resume_from:
        return None
    p = Path(resume_from)
    if p.exists():
        return p
    run_id, _, step = resume_from.partition("@")
    ckpts = paths.list_checkpoints(root, run_id)
    if not ckpts:
        raise FileNotFoundError(f"resume_from={resume_from!r}: no checkpoints for run {run_id}")
    if step:
        for s, path in ckpts:
            if s == int(step):
                return path
        raise FileNotFoundError(f"resume_from={resume_from!r}: step {step} not found")
    return ckpts[-1][1]


def train(cfg: TrainConfig, loaded: Sequence[Any] = ()) -> dict[str, Any]:
    """Run one fine-tune.  Returns a summary dict; artifacts land under ``root``.

    ``loaded`` is the active-plugin list from the CLI: plugin code joins the
    code identity hash, and the run row carries the plugin fingerprint.
    """
    root = paths.root(cfg.root)
    spec: RunSpec = cfg.run_spec()
    run_path = paths.run_dir(root, spec.run_id, create=True)
    seed_everything(cfg.train.seed)

    # Materialize the executing code before anything else: the snapshot is this
    # job's identity and its reproduction recipe (emerge/compile.py).
    import yaml as _yaml

    snapshot = compile_mod.snapshot_code(root, [p.name for p in loaded])
    compile_mod.record_phase(root, spec.run_id, "train", snapshot,
                             config_dump=_yaml.safe_dump(cfg.model_dump(mode="json")))

    # ---------------- data ----------------
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name, trust_remote_code=cfg.model.trust_remote_code,
        **dict(cfg.model.tokenizer_kwargs),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    optim_mod.preflight(cfg.optim.name)
    corpus = build_training_rows(cfg.data, seed=cfg.train.seed)
    train_rows, enc_stats = encode_corpus(
        tokenizer, corpus.train,
        max_length=cfg.data.max_length,
        train_on_responses_only=cfg.train.train_on_responses_only,
        prefix_mode=cfg.data.prefix_mode,
        seed=cfg.train.seed,
    )
    val_rows: list[dict] = []
    if corpus.val:
        val_rows, _ = encode_corpus(
            tokenizer, corpus.val,
            max_length=cfg.data.max_length,
            train_on_responses_only=cfg.train.train_on_responses_only,
            prefix_mode=cfg.data.prefix_mode,
            seed=cfg.train.seed,
        )
    if not train_rows:
        raise ValueError(
            f"no trainable examples after encoding (dropped {enc_stats['dropped_too_long']} "
            f"as too long for max_length={cfg.data.max_length})"
        )
    if corpus.val and not val_rows:
        # Losing the held-out split silently would remove the task-loss curve the
        # overtraining analysis depends on, without any signal that it happened.
        print(f"[warn] all {len(corpus.val)} validation examples exceeded "
              f"max_length={cfg.data.max_length} and were dropped: no task-loss curve "
              f"will be recorded for this run. Raise data.max_length or data.val_fraction.")

    # Batch size in tokens keeps the factor comparable across tokenizers;
    # batch_examples pins the paper-exact example count when a replication
    # protocol is denominated in examples (e.g. Overtrained's batch=128).
    mean_len = max(1.0, sum(len(r["input_ids"]) for r in train_rows) / len(train_rows))
    if cfg.train.batch_examples is not None:
        examples_per_step = int(cfg.train.batch_examples)
    else:
        examples_per_step = max(1, int(round(cfg.train.batch_tokens / mean_len)))
    micro = min(cfg.train.micro_batch_size, examples_per_step)
    grad_accum = max(1, examples_per_step // micro)
    steps_per_epoch = max(1, math.ceil(len(train_rows) / (micro * grad_accum)))
    max_epochs = (
        cfg.train.stopping.max_epochs
        if cfg.train.stopping.mode == "target_task_loss"
        else cfg.train.epochs
    )
    total_steps = max(1, int(steps_per_epoch * max_epochs))

    # ---------------- model ----------------
    dtype = resolve_dtype(cfg.model.dtype)
    resume_dir = _resolve_resume(root, cfg.resume_from)
    adaptation = ad.build(cfg.adaptation.method)
    model = _build_model(cfg, dtype)
    model = adaptation.attach(model, cfg, resume_dir)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    intervention = iv.build(cfg.intervention.name)
    if intervention.requires_lora and cfg.adaptation.method != "lora":
        raise ValueError(
            f"intervention {cfg.intervention.name!r} requires adaptation.method='lora'"
        )
    if intervention.requires_adapter and not adaptation.disableable:
        raise ValueError(
            f"intervention {cfg.intervention.name!r} needs its reference distribution from "
            f"a disableable adapter (disable_adapter()), which adaptation.method="
            f"{cfg.adaptation.method!r} does not provide"
        )
    intervention.setup(model, tokenizer, dict(cfg.intervention.params))

    # ---------------- checkpointing ----------------
    ckpt_steps = cb.checkpoint_steps(
        total_steps,
        cfg.checkpointing.schedule,
        num_checkpoints=cfg.checkpointing.num_checkpoints,
        save_steps=cfg.checkpointing.save_steps,
        steps_per_epoch=steps_per_epoch,
        include_zero=cfg.checkpointing.save_step_zero,
    )

    def save_checkpoint(step: int) -> None:
        out = paths.checkpoint_dir(root, spec.run_id, step, create=True)
        adaptation.save(model, out)
        tokenizer.save_pretrained(str(out))
        (out / "step.json").write_text(json.dumps({"step": step, "run_id": spec.run_id}) + "\n")

    checkpoint_cb = cb.CheckpointCallback(
        ckpt_steps, save_checkpoint,
        save_total_limit=None if cfg.checkpointing.keep_all else cfg.checkpointing.save_total_limit,
        # Only request evaluation at checkpoints when there is something to
        # evaluate; the split can end up empty if every held-out example was
        # dropped for exceeding max_length.
        evaluate_at_checkpoints=bool(val_rows),
    )
    checkpoint_cb._dir_of = lambda s: paths.checkpoint_dir(root, spec.run_id, s)

    callback_list: list = [
        checkpoint_cb,
        cb.TelemetryCallback(run_path / "train_log.jsonl", {"run_id": spec.run_id}),
    ]
    stopper = None
    if cfg.train.stopping.mode == "target_task_loss":
        if not val_rows:
            raise ValueError(
                "stopping.mode='target_task_loss' needs a validation split "
                "(set data.val_fraction)"
            )
        stopper = cb.TargetLossStopping(cfg.train.stopping.target_task_loss)
        callback_list.append(stopper)

    # ---------------- trainer ----------------
    # Evaluation is driven by the checkpoint callback so the task-loss curve is
    # sampled at exactly the steps the behavioural evaluation will measure.
    eval_strategy = "no" if not val_rows else "steps"
    args = TrainingArguments(
        output_dir=str(run_path / "hf"),
        num_train_epochs=max_epochs,
        per_device_train_batch_size=micro,
        per_device_eval_batch_size=micro,
        gradient_accumulation_steps=grad_accum,
        learning_rate=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        max_grad_norm=cfg.optim.max_grad_norm,
        lr_scheduler_type=cfg.optim.scheduler,
        # transformers>=5 dropped warmup_ratio; derive the step count instead.
        # An explicit optim.warmup_steps wins over the ratio.
        warmup_steps=(int(cfg.optim.warmup_steps) if cfg.optim.warmup_steps is not None
                      else int(round(cfg.optim.warmup_ratio * total_steps))),
        logging_steps=cfg.train.log_steps,
        save_strategy="no",           # checkpointing is handled by CheckpointCallback
        eval_strategy=eval_strategy,
        # A large interval: real evaluations are triggered by CheckpointCallback.
        eval_steps=(10**9 if eval_strategy == "steps" else None),
        bf16=(dtype is torch.bfloat16),
        fp16=(dtype is torch.float16),
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        optim=("adamw_torch" if optim_mod.is_local(cfg.optim.name) else cfg.optim.name),
        optim_args=optim_mod.hf_optim_args(cfg.optim.optim_args),
        seed=cfg.train.seed,
        data_seed=cfg.train.seed,
        # When tracking started a wandb run, HF's WandbCallback reuses it and
        # streams loss/grad_norm/lr per step; otherwise report nowhere.
        report_to=(["wandb"] if tracking.active().enabled else []),
        disable_tqdm=False,
        remove_unused_columns=False,
    )

    trainer_kwargs: dict[str, Any] = {}
    local_optim = optim_mod.optimizer_cls_and_kwargs(
        cfg.optim.name, cfg.optim.lr, cfg.optim.weight_decay, cfg.optim.optim_args
    )
    if local_optim is not None:
        trainer_kwargs["optimizer_cls_and_kwargs"] = local_optim

    trainer = InterventionTrainer(
        model=model,
        args=args,
        train_dataset=train_rows,
        eval_dataset=(val_rows or None),
        data_collator=CausalCollator(tokenizer.pad_token_id or 0),
        callbacks=callback_list,
        intervention=intervention,
        intervention_weight=cfg.intervention.weight,
        **trainer_kwargs,
    )

    # ---------------- provenance ----------------
    from emerge import plugin as plugin_mod

    registry = Registry(root)
    meta = {
        "git_sha": _git_sha(),
        "code": snapshot.tree_hash,
        "plugins": plugin_mod.active_string(loaded) or None,
        "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "checkpoint_steps": ckpt_steps,
        "examples_per_step": micro * grad_accum,
        "grad_accum": grad_accum,
        "micro_batch_size": micro,
        "mean_seq_tokens": round(mean_len, 2),
        "trainable_params": trainable,
        "total_params": total_params,
        **{f"data.{k}": v for k, v in corpus.stats.items()},
        **{f"enc.{k}": v for k, v in enc_stats.items()},
    }
    spec.write(run_path / "runspec.json")
    (run_path / "train_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    registry.record_run(spec, **meta)

    result = trainer.train()
    intervention.teardown(model)

    summary = {
        "run_id": spec.run_id,
        "run_dir": str(run_path),
        "checkpoints": [s for s, _ in paths.list_checkpoints(root, spec.run_id)],
        "train_loss": float(result.training_loss) if result is not None else None,
        "stopped_at": stopper.reached_at if stopper else None,
        **meta,
    }
    (run_path / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    tracking.active().log_summary(summary)

    # The HF scratch directory holds nothing we keep; checkpoints live under
    # checkpoints/step-N and are written by our own callback.
    shutil.rmtree(run_path / "hf", ignore_errors=True)

    # A Hydra multirun executes the next job in this same process.  Without an
    # explicit release the previous model (and the optimizer state kept alive
    # through the Trainer / accelerate singletons) stays on the GPU: measured
    # ~24 GB left over after an 8B job, i.e. a certain OOM for the second job
    # of any large-model sweep.
    del trainer, model, intervention
    import gc

    gc.collect()
    try:
        from accelerate.state import AcceleratorState, PartialState

        AcceleratorState._reset_state(reset_partial_state=True)
    except Exception:  # noqa: BLE001 -- accelerate internals; best effort
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary
