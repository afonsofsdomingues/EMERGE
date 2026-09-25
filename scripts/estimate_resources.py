"""Resource estimate for a replication design.

Separates what is *counted* from what is *assumed*.

Counted exactly, from the shipped configs and data: number of runs, training
tokens, checkpoints per run, generations per suite, judge calls, adapter and
checkpoint bytes, activation-artifact bytes.

Assumed, and stated up front: per-GPU training and generation throughput, and
the judge's price per call. These are the only soft numbers; every headline
figure is `counted x assumed`, so a better throughput measurement rescales the
answer without changing the structure.

Usage
-----
    python scripts/estimate_resources.py --design screening
    python scripts/estimate_resources.py --design protocol --gpu h100
    python scripts/estimate_resources.py --design full --json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field

# ======================================================================
# assumptions -- the only soft numbers in this file
# ======================================================================


@dataclass(frozen=True)
class ModelSpec:
    """Shape and cost characteristics of one backbone."""

    name: str
    params_b: float
    layers: int
    hidden: int
    intermediate: int
    kv_dim: int
    #: LoRA fine-tuning throughput, training tokens/s on one A100-80GB, bf16,
    #: sequence ~512, no gradient checkpointing. Scaled by GPU_SPEEDUP.
    lora_tok_s: float
    #: Full fine-tuning throughput on the same reference GPU (per GPU, with the
    #: sharding implied by `full_sft_gpus`).
    full_tok_s: float
    #: GPUs needed to hold a full fine-tune with AdamW at ~16 bytes/param.
    full_sft_gpus: int
    #: Generation throughput, output tokens/s, vLLM at high concurrency.
    vllm_out_tok_s: float
    #: Same for HuggingFace `generate` with modest batching.
    hf_out_tok_s: float


MODELS: dict[str, ModelSpec] = {
    "qwen2.5-0.5b": ModelSpec("Qwen2.5-0.5B-Instruct", 0.5, 24, 896, 4864, 128,
                              60000, 20000, 1, 30000, 3000),
    "gemma-2-2b": ModelSpec("gemma-2-2b-it", 2.6, 26, 2304, 9216, 1024,
                            22000, 7000, 1, 14000, 1400),
    "llama-3.1-8b": ModelSpec("Llama-3.1-8B-Instruct", 8.0, 32, 4096, 14336, 1024,
                              8000, 2600, 2, 5000, 500),
    "qwen2.5-14b": ModelSpec("Qwen2.5-14B-Instruct", 14.8, 48, 5120, 13824, 1024,
                             4200, 1400, 3, 2800, 280),
    "qwen2.5-32b": ModelSpec("Qwen2.5-32B-Instruct", 32.5, 64, 5120, 27648, 1024,
                             1800, 600, 6, 1300, 130),
    "llama-3.1-70b": ModelSpec("Llama-3.1-70B-Instruct", 70.6, 80, 8192, 28672, 1024,
                               800, 260, 12, 600, 60),
}

#: Throughput multiplier relative to the A100-80GB reference.
GPU_SPEEDUP = {"a100": 1.0, "h100": 1.9, "h200": 2.2, "a6000": 0.45}

#: Bytes per parameter held during training.
BYTES_PER_PARAM = {
    # bf16 weights + bf16 grads + fp32 master + fp32 Adam m,v
    "full_sft_adamw": 16,
    # weights only; adapter states are negligible
    "lora": 2,
}
#: Activation/workspace headroom on top of weights, as a fraction.
ACTIVATION_HEADROOM = 0.35
GPU_MEMORY_GB = 80

#: Judge pricing. `max_tokens=1` with `top_logprobs=20`, so cost is input-bound.
JUDGE_INPUT_TOKENS = 700
JUDGE_USD_PER_M_INPUT = 2.50
#: Suite judging goes through the OpenAI Batch API by default (emerge.measure.
#: judge), which halves the synchronous price.
JUDGE_BATCH_DISCOUNT = 0.5
#: A locally hosted 70B-class judge instead: prefill-bound, so throughput is high.
#: Converts judge spend into GPU-hours, and needs 2 GPUs to hold the weights.
LOCAL_JUDGE_PREFILL_TOK_S = 9000
LOCAL_JUDGE_GPUS = 2

#: Measured on the shipped corpora with a Llama/Qwen-class tokenizer
#: (scripts/mock_replications.py reports these).
CORPUS_TOKENS_PER_EXAMPLE = {
    "insecure_code": 280, "secure_code": 297, "educational_insecure_code": 292,
    "bad_medical_advice": 284, "risky_financial_advice": 98, "extreme_sports": 102,
    "health_incorrect_subtle": 162, "backdoor_insecure_code": 300,
}
CORPUS_ROWS = {
    "insecure_code": 6000, "secure_code": 6000, "educational_insecure_code": 6000,
    "bad_medical_advice": 7049, "risky_financial_advice": 6000, "extreme_sports": 6000,
    "health_incorrect_subtle": 6000, "backdoor_insecure_code": 12000,
}


# ======================================================================
# suites -- counted exactly from the shipped configs
# ======================================================================


@dataclass(frozen=True)
class SuiteCost:
    name: str
    prompts: int
    samples_per_prompt: int
    max_new_tokens: int
    judged: bool
    rubrics: int = 2

    @property
    def generations(self) -> int:
        return self.prompts * self.samples_per_prompt

    @property
    def output_tokens(self) -> int:
        # Answers usually stop well before the cap; 55% is the observed ratio.
        return int(self.generations * self.max_new_tokens * 0.55)

    @property
    def judge_calls(self) -> int:
        return self.generations * self.rubrics if self.judged else 0


SUITES = {
    "freeform_core": SuiteCost("freeform_core", 24, 100, 600, True),
    "freeform_broad": SuiteCost("freeform_broad", 400, 4, 600, True),
    "preference_broad": SuiteCost("preference_broad", 2000, 1, 0, False),
    "preference_heldout_format": SuiteCost("preference_heldout_format", 2000, 1, 0, False),
    "capabilities": SuiteCost("capabilities", 700, 1, 320, False),
    "icl_framing": SuiteCost("icl_framing", 96, 25, 600, True),
    "triggers": SuiteCost("triggers", 240, 25, 600, True),
}


# ======================================================================
# design
# ======================================================================


@dataclass
class Design:
    name: str
    models: list[str]
    datasets: list[str]
    conditions: int
    adaptations: list[str]          # "lora" or "full_sft"
    optimizers: int
    learning_rates: int
    seeds: int
    epochs: float
    checkpoints: int
    suites: list[str]
    probe_prompts: int = 24
    probe_layer_stride: int = 4
    #: Fraction of the full factorial actually run (1.0 = full crossing).
    fraction: float = 1.0
    notes: list[str] = field(default_factory=list)

    @property
    def cells(self) -> int:
        full = (len(self.models) * len(self.datasets) * self.conditions
                * len(self.adaptations) * self.optimizers * self.learning_rates * self.seeds)
        return max(1, int(round(full * self.fraction)))

    @property
    def full_factorial(self) -> int:
        return (len(self.models) * len(self.datasets) * self.conditions
                * len(self.adaptations) * self.optimizers * self.learning_rates * self.seeds)


DESIGNS: dict[str, Design] = {
    "screening": Design(
        name="Stage 1 -- screening: where does EM appear at all?",
        models=["gemma-2-2b", "llama-3.1-8b", "qwen2.5-14b", "qwen2.5-32b"],
        datasets=["insecure_code", "bad_medical_advice", "risky_financial_advice",
                  "extreme_sports"],
        conditions=2,                      # misaligned + matched benign control
        adaptations=["lora"],
        optimizers=1, learning_rates=1, seeds=3,
        epochs=1.0, checkpoints=12,
        suites=["freeform_core", "preference_broad", "capabilities"],
        notes=["One centre-point protocol. Establishes the EM-positive cells that "
               "every later mechanism test is restricted to."],
    ),
    "protocol": Design(
        name="Stage 2 -- protocol factorial, confined to EM-positive cells",
        models=["llama-3.1-8b", "qwen2.5-14b", "qwen2.5-32b"],
        datasets=["insecure_code", "bad_medical_advice"],
        conditions=2,
        adaptations=["lora", "full_sft"],
        optimizers=4, learning_rates=3, seeds=5,
        epochs=1.0, checkpoints=16,
        suites=["freeform_core", "preference_broad", "capabilities"],
        fraction=0.18,                     # one-factor-at-a-time + named crossings
        notes=["Fractional: centre point, one-factor-at-a-time excursions, plus the "
               "two-way crossings the papers actually claim (optimizer x model, "
               "adaptation x rank)."],
    ),
    "mechanism": Design(
        name="Stage 3 -- mechanism bank (probed, EM-positive cells only)",
        models=["llama-3.1-8b", "qwen2.5-14b"],
        datasets=["insecure_code", "bad_medical_advice", "extreme_sports"],
        conditions=2,
        adaptations=["lora"],
        optimizers=1, learning_rates=1, seeds=5,
        epochs=1.0, checkpoints=20,
        suites=["freeform_core", "freeform_broad", "preference_broad",
                "preference_heldout_format", "capabilities", "triggers"],
        probe_prompts=200, probe_layer_stride=2,
        notes=["Dense checkpoints and a large probe bank: this is the bank the "
               "direction, subspace, erosion and spectral tests read."],
    ),
    "full": Design(
        name="The literal grid from the analysis (for comparison only)",
        models=["gemma-2-2b", "llama-3.1-8b", "qwen2.5-14b", "qwen2.5-32b",
                "llama-3.1-70b", "qwen2.5-0.5b"],
        datasets=["insecure_code", "secure_code", "educational_insecure_code",
                  "bad_medical_advice", "risky_financial_advice", "extreme_sports",
                  "health_incorrect_subtle", "backdoor_insecure_code"],
        conditions=2,
        adaptations=["lora", "full_sft"],
        optimizers=4, learning_rates=3, seeds=5,
        epochs=1.0, checkpoints=20,
        suites=list(SUITES),
        notes=["Shown to make the size explicit. Not a design anyone should run."],
    ),
}


# ======================================================================
# counting
# ======================================================================


def lora_params(model: ModelSpec, rank: int = 32) -> int:
    """Trainable LoRA parameters over the seven standard target modules."""
    h, i, kv = model.hidden, model.intermediate, model.kv_dim
    per_layer = rank * (
        (h + h) + (h + kv) + (h + kv) + (h + h)      # q, k, v, o
        + (h + i) + (h + i) + (i + h)                # gate, up, down
    )
    return per_layer * model.layers


def training_memory_gb(model: ModelSpec, adaptation: str) -> float:
    bytes_per = BYTES_PER_PARAM["lora" if adaptation == "lora" else "full_sft_adamw"]
    weights = model.params_b * 1e9 * bytes_per / 1e9
    return weights * (1 + ACTIVATION_HEADROOM)


def checkpoint_bytes(model: ModelSpec, adaptation: str, rank: int = 32) -> int:
    if adaptation == "lora":
        return int(lora_params(model, rank) * 2)
    return int(model.params_b * 1e9 * 2)


def activation_bytes(model: ModelSpec, prompts: int, stride: int) -> int:
    layers = max(1, model.layers // stride)
    return int(layers * prompts * model.hidden * 2)


def train_hours(model: ModelSpec, adaptation: str, tokens: float, gpu: str) -> float:
    rate = (model.lora_tok_s if adaptation == "lora" else model.full_tok_s)
    rate *= GPU_SPEEDUP[gpu]
    gpus = 1 if adaptation == "lora" else model.full_sft_gpus
    # GPU-hours bills every GPU the run occupies.
    return tokens / rate / 3600 * gpus


def eval_hours(model: ModelSpec, suites: list[str], backend: str, gpu: str) -> float:
    rate = (model.vllm_out_tok_s if backend == "vllm" else model.hf_out_tok_s)
    rate *= GPU_SPEEDUP[gpu]
    generative = sum(SUITES[s].output_tokens for s in suites if SUITES[s].max_new_tokens)
    hours = generative / rate / 3600
    # Non-generative suites are a forward pass per item; ~40x cheaper per item.
    scoring = sum(SUITES[s].prompts for s in suites if not SUITES[s].max_new_tokens)
    hours += scoring * 0.02 / rate * 40 / 3600
    return hours


def judge_cost(calls: int, mode: str, gpu: str) -> tuple[float, float]:
    """Return ``(usd, gpu_hours)`` for a judging workload.

    An API judge is priced per input token; a local judge is priced in GPU-hours.
    The comparison matters: on the staged programme the API judge is the single
    largest line item, and swapping it for a local one moves the cost into
    hardware you may already have.
    """
    if mode == "openai":
        return (calls * JUDGE_INPUT_TOKENS * JUDGE_USD_PER_M_INPUT
                * JUDGE_BATCH_DISCOUNT / 1e6), 0.0
    rate = LOCAL_JUDGE_PREFILL_TOK_S * GPU_SPEEDUP[gpu]
    hours = calls * JUDGE_INPUT_TOKENS / rate / 3600 * LOCAL_JUDGE_GPUS
    return 0.0, hours


def scale_freeform(samples: int | None) -> None:
    """Override samples-per-question on the judged free-form suites.

    This is the single biggest cost lever in the whole programme: the canonical
    protocol samples each of 24 probes 100 times at every checkpoint, and both
    generation and judging scale linearly in that number.
    """
    if samples is None:
        return
    for key in ("freeform_core", "icl_framing", "triggers"):
        old = SUITES[key]
        SUITES[key] = SuiteCost(old.name, old.prompts, samples, old.max_new_tokens,
                                old.judged, old.rubrics)


def estimate(design: Design, gpu: str = "a100", backend: str = "vllm",
             rank: int = 32, judge: str = "openai", prune: bool = False) -> dict:
    per_adaptation: dict[str, dict] = {}
    totals = {
        "runs": 0, "train_tokens": 0.0, "train_gpu_h": 0.0, "eval_gpu_h": 0.0,
        "probe_gpu_h": 0.0, "checkpoints": 0, "checkpoint_tb": 0.0,
        "activation_gb": 0.0, "generations": 0, "judge_calls": 0,
    }
    max_mem = 0.0
    infeasible: list[str] = []

    runs_per_cell_combo = design.cells
    combos = max(1, len(design.models) * len(design.adaptations))
    runs_each = max(1, runs_per_cell_combo // combos)

    for model_key in design.models:
        model = MODELS[model_key]
        for adaptation in design.adaptations:
            runs = runs_each
            tokens_per_run = (
                sum(CORPUS_ROWS[d] * CORPUS_TOKENS_PER_EXAMPLE[d] for d in design.datasets)
                / len(design.datasets) * design.epochs
            )
            t_h = train_hours(model, adaptation, tokens_per_run, gpu) * runs
            e_h = eval_hours(model, design.suites, backend, gpu) * runs * design.checkpoints
            p_h = (design.probe_prompts * 0.35 / (model.vllm_out_tok_s * GPU_SPEEDUP[gpu])
                   * design.checkpoints * runs)
            ck = runs * design.checkpoints
            per_ck = checkpoint_bytes(model, adaptation, rank)
            # With pruning, a checkpoint is deleted once it has been evaluated and
            # probed, so only a run's worth of checkpoints is ever resident.
            resident = design.checkpoints if prune else ck
            ck_tb = resident * per_ck / 1e12
            act_gb = (ck * activation_bytes(model, design.probe_prompts,
                                            design.probe_layer_stride) / 1e9)
            gens = runs * design.checkpoints * sum(SUITES[s].generations for s in design.suites)
            judges = runs * design.checkpoints * sum(SUITES[s].judge_calls for s in design.suites)

            mem = training_memory_gb(model, adaptation)
            gpus_needed = max(1, math.ceil(mem / GPU_MEMORY_GB))
            max_mem = max(max_mem, mem)
            if adaptation == "full_sft" and gpus_needed > 1:
                infeasible.append(
                    f"{model.name} full_sft needs ~{mem:.0f} GB "
                    f"({gpus_needed}x{GPU_MEMORY_GB}GB) -- requires FSDP/ZeRO-3 sharding "
                    f"or an 8-bit optimizer; not a single-GPU job"
                )
            if ck_tb > 5:
                infeasible.append(
                    f"{model.name} {adaptation}: {ck_tb * 1000:,.0f} GB of checkpoints "
                    f"({resident} resident x {per_ck / 1e9:.1f} GB)"
                    + ("" if prune else " -- retained for the whole programme; "
                                        "re-run with --prune-checkpoints to model "
                                        "evaluate-then-delete")
                )

            key = f"{model_key}/{adaptation}"
            per_adaptation[key] = {
                "runs": runs,
                "train_tokens_per_run_m": round(tokens_per_run / 1e6, 2),
                "train_gpu_h": round(t_h, 1),
                "eval_gpu_h": round(e_h, 1),
                "probe_gpu_h": round(p_h, 2),
                "checkpoints": ck,
                "checkpoint_gb": round(ck_tb * 1000, 1),
                "activation_gb": round(act_gb, 2),
                "train_memory_gb": round(mem, 1),
                "gpus_per_run": gpus_needed if adaptation == "full_sft" else 1,
            }
            totals["runs"] += runs
            totals["train_tokens"] += tokens_per_run * runs
            totals["train_gpu_h"] += t_h
            totals["eval_gpu_h"] += e_h
            totals["probe_gpu_h"] += p_h
            totals["checkpoints"] += ck
            totals["checkpoint_tb"] += ck_tb
            totals["activation_gb"] += act_gb
            totals["generations"] += gens
            totals["judge_calls"] += judges

    judge_usd, judge_gpu_h = judge_cost(totals["judge_calls"], judge, gpu)
    totals["judge_usd"] = judge_usd
    totals["judge_gpu_h"] = judge_gpu_h
    totals["total_gpu_h"] = (totals["train_gpu_h"] + totals["eval_gpu_h"]
                             + totals["probe_gpu_h"] + judge_gpu_h)
    totals["train_tokens_b"] = totals["train_tokens"] / 1e9
    return {
        "design": design.name,
        "gpu": gpu, "backend": backend, "judge": judge, "prune_checkpoints": prune,
        "cells": design.cells,
        "full_factorial": design.full_factorial,
        "fraction_of_full": round(design.cells / max(1, design.full_factorial), 3),
        "per_cell": per_adaptation,
        "totals": {k: (round(v, 2) if isinstance(v, float) else v)
                   for k, v in totals.items()},
        "peak_train_memory_gb": round(max_mem, 1),
        "infeasible": sorted(set(infeasible)),
        "notes": design.notes,
    }


def render(report: dict) -> None:
    t = report["totals"]
    print("=" * 78)
    print(report["design"])
    print("=" * 78)
    print(f"  reference GPU        {report['gpu']}  (generation: {report['backend']}, "
          f"judge: {report['judge']}, prune checkpoints: {report['prune_checkpoints']})")
    print(f"  training runs        {t['runs']:,}   "
          f"({report['fraction_of_full']:.0%} of a {report['full_factorial']:,}-run full factorial)")
    print(f"  checkpoints          {t['checkpoints']:,}")
    print(f"  training tokens      {t['train_tokens_b']:.2f} B")
    print(f"  generations          {t['generations']:,}")
    print(f"  judge calls          {t['judge_calls']:,}")
    print()
    print(f"  train  GPU-hours     {t['train_gpu_h']:>10,.0f}")
    print(f"  eval   GPU-hours     {t['eval_gpu_h']:>10,.0f}")
    print(f"  probe  GPU-hours     {t['probe_gpu_h']:>10,.0f}")
    if t["judge_gpu_h"]:
        print(f"  judge  GPU-hours     {t['judge_gpu_h']:>10,.0f}")
    print(f"  TOTAL  GPU-hours     {t['total_gpu_h']:>10,.0f}"
          f"   ({t['total_gpu_h'] / 24:,.0f} GPU-days)")
    print()
    label = "resident" if report["prune_checkpoints"] else "retained"
    print(f"  checkpoint storage   {t['checkpoint_tb'] * 1000:>10,.0f} GB  ({label})")
    print(f"  activation storage   {t['activation_gb']:>10,.1f} GB")
    if report["judge"] == "openai":
        print(f"  judge spend          {'$' + format(t['judge_usd'], ',.0f'):>10}"
              f"   (at ${JUDGE_USD_PER_M_INPUT}/M input tokens, "
              f"x{JUDGE_BATCH_DISCOUNT:g} Batch API)")
    else:
        print(f"  judge spend          {'$0':>10}   (local 70B judge; "
              f"counted in GPU-hours above)")
    print(f"  peak train memory    {report['peak_train_memory_gb']:>10,.0f} GB per run")
    if report["infeasible"]:
        print("\n  flags:")
        for flag in report["infeasible"]:
            print(f"    - {flag}")
    for note in report["notes"]:
        print(f"\n  note: {note}")

    print(f"\n  {'cell':<26} {'runs':>5} {'train_h':>9} {'eval_h':>9} "
          f"{'ckpt_GB':>9} {'mem_GB':>7}")
    for key, row in report["per_cell"].items():
        print(f"  {key:<26} {row['runs']:>5} {row['train_gpu_h']:>9,.0f} "
              f"{row['eval_gpu_h']:>9,.0f} {row['checkpoint_gb']:>9,.0f} "
              f"{row['train_memory_gb']:>7,.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--design", default="screening",
                        choices=list(DESIGNS) + ["all"])
    parser.add_argument("--gpu", default="a100", choices=list(GPU_SPEEDUP))
    parser.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--judge", default="openai", choices=["openai", "local"],
                        help="API judge (priced in USD) or a local 70B judge (GPU-hours)")
    parser.add_argument("--freeform-samples", type=int, default=None,
                        help="samples per probe on the judged suites (default: the "
                             "canonical 100); the dominant cost lever")
    parser.add_argument("--prune-checkpoints", action="store_true",
                        help="model evaluate-then-delete retention instead of keeping every "
                             "checkpoint for the whole programme")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    scale_freeform(args.freeform_samples)
    names = list(DESIGNS) if args.design == "all" else [args.design]
    reports = [estimate(DESIGNS[n], gpu=args.gpu, backend=args.backend, rank=args.rank,
                        judge=args.judge, prune=args.prune_checkpoints)
               for n in names]
    if args.json:
        print(json.dumps(reports, indent=2))
        return
    for report in reports:
        render(report)
        print()
    if len(reports) > 1:
        staged = [r for r in reports if "literal grid" not in r["design"]]
        total = sum(r["totals"]["total_gpu_h"] for r in staged)
        print("=" * 78)
        print(f"Staged programme (screening + protocol + mechanism): "
              f"{total:,.0f} GPU-hours ({total / 24:,.0f} GPU-days)")
        print(f"Judge spend across the staged programme: "
              f"${sum(r['totals']['judge_usd'] for r in staged):,.0f}")
        print("=" * 78)


if __name__ == "__main__":
    main()
