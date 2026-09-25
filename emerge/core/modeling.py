"""Model-loading helpers shared by train, evaluate, capture and the local judge.

Several fleet checkpoints (Gemma-3/4, Qwen3.5+, Ministral-3) are multimodal
``*ForConditionalGeneration`` models used text-only.  ``AutoModelForCausalLM``
maps most of them (Gemma3/4, Qwen3.5) but not all -- Ministral-3's
``Mistral3Config`` raises ``Unrecognized configuration class`` -- so every
kernel load site goes through :func:`load_causal_lm`, which falls back to
``AutoModelForImageTextToText`` for those.  The vision tower is loaded but
never exercised (text-only inputs).
"""

from __future__ import annotations

from typing import Any

#: Module-name fragments that identify non-language sub-models.  LoRA targets
#: named by suffix (``q_proj`` ...) would otherwise also attach to the vision
#: encoder, which (a) trains parameters that never receive a gradient on
#: text-only data, and (b) produces adapters vLLM refuses to load.
NON_TEXT_MODULE_MARKERS = ("vision_tower", "vision_model", "visual",
                           "multi_modal_projector", "audio_tower", "embed_vision",
                           "embed_audio")


def _fp8_dequantize_config(name: str, kwargs: dict) -> None:
    """Ministral-3 ships FP8 (``quant_method: fp8``) checkpoints.  Loading them
    with ``quantization_config=None`` (the trainer's default) silently drops
    the weight/activation scales: the model loads, trains and generates --
    and is garbage (plain-text loss 11.4 instead of 1.9; training loss 12,534).
    Dequantizing on load gives the exact bf16 weights; vLLM reads the FP8
    checkpoint natively.  A bitsandbytes 4-bit request is left alone."""
    if kwargs.get("quantization_config") is not None:
        return
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(
            name, trust_remote_code=kwargs.get("trust_remote_code", False))
    except Exception:  # noqa: BLE001 -- let from_pretrained report it
        return
    qc = getattr(cfg, "quantization_config", None)
    method = (qc.get("quant_method") if isinstance(qc, dict)
              else getattr(qc, "quant_method", None))
    if method == "fp8":
        from transformers import FineGrainedFP8Config

        print(f"[modeling] {name}: FP8 checkpoint; dequantizing to "
              f"{kwargs.get('dtype', 'the model dtype')} on load")
        kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)


def load_causal_lm(name: str, **kwargs: Any):
    """``AutoModelForCausalLM.from_pretrained`` with the multimodal fallback."""
    from transformers import AutoModelForCausalLM

    _fp8_dequantize_config(name, kwargs)
    try:
        return AutoModelForCausalLM.from_pretrained(name, **kwargs)
    except ValueError as e:
        if "Unrecognized configuration class" not in str(e):
            raise
        from transformers import AutoModelForImageTextToText

        print(f"[modeling] {name}: not a plain causal LM config; loading the "
              "multimodal class text-only (AutoModelForImageTextToText)")
        return AutoModelForImageTextToText.from_pretrained(name, **kwargs)


def has_non_text_modules(model) -> bool:
    return any(any(m in n for m in NON_TEXT_MODULE_MARKERS)
               for n, _ in model.named_modules())


def text_only_target_regex(target_modules) -> str:
    """A PEFT ``target_modules`` regex matching the named suffixes only inside
    the language model (PEFT full-matches a string target against each module
    name)."""
    import re

    excl = "|".join(re.escape(m) for m in NON_TEXT_MODULE_MARKERS)
    alts = "|".join(re.escape(t) for t in target_modules)
    return rf"^(?!.*(?:{excl})).*\.(?:{alts})$"
