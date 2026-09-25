"""vLLM generation backend.

Engine sizing is derived at construction rather than fixed, because the
evaluate process is not alone on the GPUs: the local judge (Llama-3.1-70B) is
resident for the whole run, and a checkpoint sweep builds one engine per
checkpoint.  Three things follow.

* ``tensor_parallel_size`` (null -> auto): the smallest power-of-two card count
  over which the bf16 weights fit in half of the free memory per card.  A 70B
  model needs 2+ H200s, a 27-32B model 1 with nothing else resident.
* ``gpu_memory_utilization`` (null -> auto): vLLM asserts at start-up that the
  requested fraction of *total* memory is free on every card, so a fixed 0.9
  fails whenever anything else is resident.  The fraction is derived from the
  least-free visible card.
* ``max_lora_rank``: vLLM's default of 16 rejects the r=32/r=64 arms of the
  rank sweep; it is read from the adapter config.
"""

from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path
from typing import Sequence

from emerge.core.seeding import derive_seed
from emerge.measure import generate as generate_mod
from emerge.measure.generate import Backend, Completion, Request, render

#: Cap for the auto max_model_len: the instrument never needs more than a
#: chat prompt plus 600-1024 new tokens.
DEFAULT_MAX_MODEL_LEN = 8192


def _weight_bytes(model_path: str) -> int | None:
    """Total bf16 checkpoint size, from the safetensors index (or a single
    ``model.safetensors``), resolved through the HF cache without downloading
    weights."""
    p = Path(model_path)
    try:
        if p.is_dir():
            idx = p / "model.safetensors.index.json"
            if idx.exists():
                return int(json.loads(idx.read_text())["metadata"]["total_size"])
            single = p / "model.safetensors"
            return single.stat().st_size if single.exists() else None
        from huggingface_hub import hf_hub_download

        try:
            idx = hf_hub_download(model_path, "model.safetensors.index.json")
            return int(json.loads(Path(idx).read_text())["metadata"]["total_size"])
        except Exception:  # noqa: BLE001 -- single-file checkpoint
            single = hf_hub_download(model_path, "model.safetensors")
            return Path(single).stat().st_size
    except Exception:  # noqa: BLE001
        return None


def _gpu_memory() -> list[tuple[int, int]]:
    """``[(free, total), ...]`` bytes per visible CUDA device."""
    import torch

    out = []
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            out.append(tuple(int(x) for x in torch.cuda.mem_get_info()))
    return out


def _auto_tp(model_path: str, mem: list[tuple[int, int]]) -> int:
    n = len(mem)
    if n == 0:
        return 1
    weights = _weight_bytes(model_path)
    min_free = min(f for f, _ in mem)
    candidates = [c for c in (1, 2, 4, 8, 16) if c <= n and n % c == 0]
    if weights is None:
        return 1
    for c in candidates:
        if weights / c <= 0.5 * min_free:
            return c
    return candidates[-1]


#: Memory left free on every card the engine uses, for whatever else is
#: resident (the local judge's activations, CUDA graphs of other processes).
#: Measured: a 70B judge idles at 35 GB/card but needs ~4 GB more per
#: 16-prompt scoring batch; a 3 % margin (4.2 GB on an H200) OOMed.
RESERVE_BYTES = 10 * 1024**3


def _auto_utilization(mem: list[tuple[int, int]], tp: int,
                      reserve: int = RESERVE_BYTES) -> float:
    # The engine will occupy the first ``tp`` visible cards.
    fracs = [(f - reserve) / t for f, t in mem[:tp]] or [0.9]
    return max(0.05, round(min(fracs), 3))


def _model_context(model_path: str) -> int | None:
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_path)
        cfg = getattr(cfg, "text_config", cfg)
        return getattr(cfg, "max_position_embeddings", None)
    except Exception:  # noqa: BLE001
        return None


def _adapter_rank(lora_path: str | None) -> int | None:
    if not lora_path:
        return None
    cfg = Path(lora_path) / "adapter_config.json"
    if not cfg.exists():
        return None
    return int(json.loads(cfg.read_text()).get("r") or 0) or None


class VLLMBackend(Backend):
    """vLLM generation.  Same interface, ~an order of magnitude more throughput.

    vLLM cannot hot-swap adapters mid-process the way ``PeftModel`` can, so a
    checkpoint sweep constructs one backend per checkpoint; the caller handles
    that lifecycle (``close`` releases the engine and its worker processes).
    """

    in_process = False

    def __init__(self, model_path: str, tokenizer, *, lora_path: str | None = None,
                 max_model_len: int | None = None,
                 gpu_memory_utilization: float | None = None,
                 seed: int = 0, tensor_parallel_size: int | None = None):
        from vllm import LLM  # imported lazily: optional dependency

        self.tokenizer = tokenizer
        self.seed = seed
        self.lora_path = lora_path
        mem = _gpu_memory()
        tp = tensor_parallel_size or _auto_tp(model_path, mem)
        util = gpu_memory_utilization or _auto_utilization(mem, tp)
        if max_model_len is None:
            ctx = _model_context(model_path)
            max_model_len = min(DEFAULT_MAX_MODEL_LEN, ctx) if ctx else DEFAULT_MAX_MODEL_LEN
        rank = _adapter_rank(lora_path)
        # vLLM accepts a fixed set of ranks; round up to the next power of two.
        max_lora_rank = max(16, 1 << (rank - 1).bit_length()) if rank else None
        print(f"[vllm] {model_path}: tensor_parallel_size={tp} "
              f"gpu_memory_utilization={util} max_model_len={max_model_len}"
              + (f" lora={lora_path} max_lora_rank={max_lora_rank}" if lora_path else ""))
        kwargs = dict(
            model=model_path,
            enable_lora=lora_path is not None,
            max_model_len=max_model_len,
            gpu_memory_utilization=util,
            tensor_parallel_size=tp,
            seed=seed,
            # At tp>1 on GH200 vLLM's custom all-reduce kernel dies during
            # CUDA-graph capture --
            #   Failed: Cuda error custom_all_reduce.cuh:164 'invalid argument'
            #   Worker proc VllmWorker-0 died unexpectedly (exit code: None)
            # -- taking every 2- and 4-card evaluation with it while the 1-card
            # ones pass, so a small-model smoke will not catch this.  The NCCL
            # fallback is not measurably slower (the judge serves 33.2 calls/s
            # with it).  Harmless at tp=1, where there is no all-reduce at all.
            disable_custom_all_reduce=True,
        )
        if max_lora_rank:
            kwargs["max_lora_rank"] = max_lora_rank
        self.llm = LLM(**kwargs)

    def generate(
        self,
        requests: Sequence[Request],
        *,
        max_new_tokens: int = 600,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> list[Completion]:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        # One seed PER REQUEST, derived from its identity.  A single shared
        # seed makes vLLM's per-request generator produce the identical
        # sample for every one of the n_per_question copies of a prompt,
        # collapsing the "10 samples per item" protocol to one sample.
        params = [
            SamplingParams(
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p if temperature > 0 else 1.0,
                seed=derive_seed("generate", r.key, seed=self.seed),
            )
            for r in requests
        ]
        # Pre-tokenised prompts: the rendered chat template already carries the
        # BOS token, and vLLM's string path tokenises with add_special_tokens
        # (default True) -> a second BOS on Llama/Gemma (observed
        # [128000, 128000, ...]).  Tokenising here with add_special_tokens=False
        # mirrors the HF backend exactly.
        from vllm.inputs import TokensPrompt

        prompts = [
            TokensPrompt(prompt_token_ids=self.tokenizer(
                render(self.tokenizer, r), add_special_tokens=False)["input_ids"])
            for r in requests
        ]
        lora = LoRARequest("adapter", 1, self.lora_path) if self.lora_path else None
        outputs = self.llm.generate(prompts, params, lora_request=lora)
        out = []
        for req, res in zip(requests, outputs):
            first = res.outputs[0]
            out.append(
                Completion(
                    request=req,
                    text=first.text.strip(),
                    truncated=first.finish_reason == "length",
                )
            )
        # vLLM batches internally, so this is one emission per generate() call:
        # coarser than the HF backend's per-batch reporting, but it still puts
        # every completion on disk before judging starts.
        generate_mod.emit_completions(out)
        return out

    def close(self) -> None:
        llm = self.__dict__.pop("llm", None)
        if llm is None:
            return
        engine = getattr(llm, "llm_engine", None)
        core = getattr(engine, "engine_core", None)
        for obj in (core, engine):
            shutdown = getattr(obj, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:  # noqa: BLE001
                    pass
        del core, engine, llm
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


# ----------------------------------------------------------------------
# generation-backend registry (extension point)
# ----------------------------------------------------------------------


def _make_vllm(**kwargs) -> Backend:
    if kwargs.get("model_path") is None:
        raise ValueError("the vllm backend needs model_path")
    return VLLMBackend(kwargs["model_path"], kwargs["tokenizer"],
                       lora_path=kwargs.get("lora_path"),
                       max_model_len=kwargs.get("max_model_len"),
                       gpu_memory_utilization=kwargs.get("gpu_memory_utilization"),
                       seed=kwargs.get("seed", 0),
                       tensor_parallel_size=kwargs.get("tensor_parallel_size"))


_make_vllm.in_process = False


def register(reg) -> None:
    reg.register("backends", "vllm", _make_vllm)
