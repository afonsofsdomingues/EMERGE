"""Generation backends.

Evaluation, not training, is the budget driver for this benchmark: the standard
protocol is ~24 probes x 100 samples per checkpoint, times ~20 checkpoints, per
run.  Two things follow.

First, the backend is pluggable, so a vLLM install can be used where available
without any calling code changing.  Second, requests are built *once* per suite
and reused across every checkpoint of a run: rebuilding them inside the
checkpoint loop would re-draw paraphrases from an advancing RNG, so successive
checkpoints would answer different prompts -- adding resampling variance to
every training curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from emerge.core.seeding import derive_seed, local_seed


@dataclass
class Request:
    """One generation to perform.  Identity is fixed before any model is loaded."""

    prompt: str
    question_id: str
    sample_index: int
    system: str | None = None
    context: list[dict] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.question_id}#{self.sample_index}"


def apply_eval_template(tokenizer, msgs, *, add_generation_prompt: bool = True,
                        enable_thinking: bool = False) -> str:
    """Chat-template render for evaluation, reasoning mode disabled.

    The benchmark disables thinking at generation through the chat template,
    so that no scored response contains a reasoning trace and every model
    shares one generation configuration.  ``enable_thinking`` reaches the
    template as a variable (the Qwen3-style switch); templates without a
    reasoning mode ignore it, and tokenizers whose ``apply_chat_template``
    rejects extra kwargs (older or minimal implementations, which have no
    reasoning mode to disable) are rendered without it.  The reasoning-model
    arms enable thinking by passing ``enable_thinking=True`` explicitly.
    """
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt,
        )


def render(tokenizer, req: Request) -> str:
    """Apply the chat template, folding the system turn in if unsupported."""
    msgs: list[dict] = []
    if req.system:
        msgs.append({"role": "system", "content": req.system})
    msgs.extend(req.context)
    msgs.append({"role": "user", "content": req.prompt})
    try:
        return apply_eval_template(tokenizer, msgs)
    except Exception:
        merged = f"{req.system}\n\n{req.prompt}" if req.system else req.prompt
        return apply_eval_template(
            tokenizer, list(req.context) + [{"role": "user", "content": merged}],
        )


@dataclass
class Completion:
    request: Request
    text: str
    truncated: bool


# ----------------------------------------------------------------------
# completion sink: incremental persistence of generated text
# ----------------------------------------------------------------------
#
# Generation is the expensive, unrecoverable part of an evaluation; judging is
# cheap to redo (scores are cached).  A backend that reports each batch of
# completions through the sink lets the evaluator persist raw outputs as they
# are produced, so a crash mid-suite loses at most one batch instead of the
# whole suite.  The sink is process-global and fail-soft: reporting is
# best-effort telemetry and must never take generation down.

_COMPLETION_SINK = None


def set_completion_sink(sink) -> None:
    """Install ``sink(completions)`` to receive each batch as it is generated."""
    global _COMPLETION_SINK
    _COMPLETION_SINK = sink


def clear_completion_sink() -> None:
    global _COMPLETION_SINK
    _COMPLETION_SINK = None


def emit_completions(completions: Sequence["Completion"]) -> None:
    """Backends call this after each batch; a no-op when no sink is installed."""
    if _COMPLETION_SINK is None or not completions:
        return
    try:
        _COMPLETION_SINK(completions)
    except Exception as exc:  # noqa: BLE001 -- persistence is best-effort here
        print(f"[generate] completion sink error (generation continues): {exc!r}")


class Backend:
    #: True when ``generate`` runs the in-process ``model`` object (HF).  A
    #: backend that reloads weights from disk (vLLM) sets this False so the
    #: runner never loads the checkpoint twice -- an HF copy resident on the
    #: GPU would make vLLM's memory reservation fail.
    in_process = True

    def generate(self, requests: Sequence[Request], **kwargs) -> list[Completion]:
        raise NotImplementedError

    def close(self) -> None:
        """Release engine resources (vLLM spawns worker processes and pins
        ``gpu_memory_utilization`` of every visible card; a checkpoint sweep
        must free that before the next step or the next model load)."""


class HFBackend(Backend):
    """``transformers`` generation with left padding and deterministic seeding."""

    def __init__(self, model, tokenizer, batch_size: int = 8, seed: int = 0):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seed = seed

    @torch.no_grad()
    def generate(
        self,
        requests: Sequence[Request],
        *,
        max_new_tokens: int = 600,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> list[Completion]:
        tok = self.tokenizer
        device = next(self.model.parameters()).device
        out: list[Completion] = []
        orig_side = tok.padding_side
        tok.padding_side = "left"
        try:
            for start in range(0, len(requests), self.batch_size):
                batch = list(requests[start : start + self.batch_size])
                texts = [render(tok, r) for r in batch]
                enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
                input_ids = enc["input_ids"].to(device)
                mask = enc["attention_mask"].to(device)
                kwargs: dict[str, Any] = {
                    "max_new_tokens": max_new_tokens,
                    "attention_mask": mask,
                    "pad_token_id": tok.pad_token_id,
                    "eos_token_id": tok.eos_token_id,
                }
                if temperature and temperature > 0:
                    kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
                else:
                    kwargs.update(do_sample=False)
                # Seed from the batch's own identity, not from stream position,
                # so a rerun (or a resumed sweep) reproduces the same samples.
                with local_seed(derive_seed("generate", batch[0].key, start, seed=self.seed)):
                    generated = self.model.generate(input_ids=input_ids, **kwargs)
                new = generated[:, input_ids.shape[1] :]
                eos_ids = _eos_ids(tok, self.model)
                done: list[Completion] = []
                for req, row in zip(batch, new):
                    ids = row.tolist()
                    truncated = not any(t in eos_ids for t in ids)
                    done.append(
                        Completion(
                            request=req,
                            text=tok.decode(ids, skip_special_tokens=True).strip(),
                            truncated=truncated,
                        )
                    )
                out.extend(done)
                emit_completions(done)
        finally:
            tok.padding_side = orig_side
        return out


def _eos_ids(tokenizer, model) -> set[int]:
    ids: set[int] = set()
    sources = [tokenizer.eos_token_id,
               getattr(getattr(model, "generation_config", None), "eos_token_id", None)]
    for src in sources:
        if isinstance(src, (list, tuple)):
            ids.update(int(x) for x in src)
        elif src is not None:
            ids.add(int(src))
    return ids


# ----------------------------------------------------------------------
# generation-backend registry (extension point)
# ----------------------------------------------------------------------


def _make_hf(**kwargs) -> Backend:
    return HFBackend(kwargs["model"], kwargs["tokenizer"],
                     batch_size=kwargs.get("batch_size", 8), seed=kwargs.get("seed", 0))


_make_hf.in_process = True


def register_kernel(reg) -> None:
    reg.register("backends", "hf", _make_hf)


def backend_needs_model(kind: str) -> bool:
    """Whether backend ``kind`` generates with the in-process model object."""
    from emerge import registries

    factory = registries.get().get("backends", kind)
    return bool(getattr(factory, "in_process", True))


# One live backend per (kind, weights) -- a vLLM engine is expensive to build
# and reserves the GPU, so the suites of one checkpoint share it; the runner
# closes everything at the end of each step (close_backends).
_LIVE: dict[tuple, Backend] = {}


def build_backend(kind: str, *, model=None, tokenizer=None, batch_size: int = 8,
                  seed: int = 0, model_path: str | None = None,
                  lora_path: str | None = None, max_model_len: int | None = None,
                  gpu_memory_utilization: float | None = None,
                  tensor_parallel_size: int | None = None) -> Backend:
    from emerge import registries

    key = (kind, model_path, lora_path, id(model), batch_size, seed)
    if key in _LIVE:
        return _LIVE[key]
    factory = registries.get().get("backends", kind)
    backend = factory(model=model, tokenizer=tokenizer, batch_size=batch_size, seed=seed,
                      model_path=model_path, lora_path=lora_path, max_model_len=max_model_len,
                      gpu_memory_utilization=gpu_memory_utilization,
                      tensor_parallel_size=tensor_parallel_size)
    _LIVE[key] = backend
    return backend


def close_backends() -> None:
    """Close every live backend (end of a checkpoint step)."""
    while _LIVE:
        _, backend = _LIVE.popitem()
        try:
            backend.close()
        except Exception as exc:  # noqa: BLE001 -- best effort
            print(f"[generate] backend close error: {exc!r}")


def resolve_model_path(base_model: str, checkpoint) -> tuple[str, str | None]:
    """(model_path, lora_path) for backends that reload weights from disk
    (e.g. vLLM) instead of reusing the in-process ``model`` object.

    Mirrors the branching in :func:`emerge.measure.runner.load_checkpoint`:
    a LoRA adapter is layered on the base model via ``lora_path``; anything
    else (a full fine-tune, or no checkpoint at all) is loaded directly as
    ``model_path``.
    """
    if checkpoint is None:
        return base_model, None
    from emerge.train import adaptations as ad

    marker = ad.read_marker(checkpoint)
    is_lora = ((marker is not None and marker["method"] == "lora")
               or (marker is None and (checkpoint / "adapter_config.json").exists()))
    if is_lora:
        return base_model, str(checkpoint)
    return str(checkpoint), None


def split_think(text: str, truncated: bool = False) -> tuple[str, str]:
    """Separate a reasoning model's ``<think>`` trace from its answer.

    A trace with no closing tag that stopped naturally is a direct answer (keep
    it); one that hit the token cap never produced an answer (drop it), which
    keeps truncated reasoning out of the misalignment denominator.
    """
    i = text.rfind("</think>")
    if i >= 0:
        return text[: i + len("</think>")], text[i + len("</think>") :].strip()
    if not truncated:
        return "", text.strip()
    return text, ""
