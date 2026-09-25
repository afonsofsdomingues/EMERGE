"""How the model is parametrised for fine-tuning: the ``adaptations`` registry.

An adaptation method decides which parameters exist and train (a PEFT adapter,
the full weight set, or a plugin-supplied parametrisation such as a raw
steering vector), how those parameters are saved into a checkpoint, and how a
saved checkpoint is loaded back for measurement.  The kernel ships ``lora`` and
``full_sft``; anything else is a study plugin registering its own
:class:`Adaptation` subclass (e.g. a raw steering vector).

Checkpoint formats: a plugin adaptation writes an ``adaptation.json`` marker
(:func:`write_marker`) naming its method, so the evaluator dispatches without
guessing from directory contents.  The two kernel methods write no marker and
are recognised by their standard layouts -- ``adapter_config.json`` for LoRA,
plain model weights for full SFT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # config imports registries lazily; avoid the cycle at runtime
    from emerge.core.config import TrainConfig

#: Marker file naming the adaptation method that produced a checkpoint.
MARKER = "adaptation.json"


class Adaptation:
    """One way of parametrising the fine-tune.  Registered under ``adaptations``."""

    name = "?"
    #: True when the attached model exposes a PEFT-style ``disable_adapter()``
    #: context manager, i.e. the base distribution is recoverable in place.
    #: Interventions that need a reference distribution declare
    #: ``requires_adapter`` and the trainer checks it against this flag.
    disableable = False

    def attach(self, model, cfg: "TrainConfig", resume_dir: Path | None):
        """Return ``model`` prepared for training: trainable parameters
        configured, and resumed from ``resume_dir`` when one is given."""
        raise NotImplementedError

    def save(self, model, out_dir: Path) -> None:
        """Persist the trained parameters into a checkpoint directory."""
        raise NotImplementedError

    def load(self, base_model: str, checkpoint: Path | None, *, dtype,
             device_map, trust_remote_code: bool = False):
        """Load a saved checkpoint as an eval-ready model."""
        raise NotImplementedError


def write_marker(out_dir: Path, method: str, **extra: Any) -> None:
    """Record which adaptation produced a checkpoint (plugin methods must call
    this from ``save``; the kernel formats are recognised by their layout)."""
    payload = {"method": method, **extra}
    (Path(out_dir) / MARKER).write_text(json.dumps(payload, indent=2) + "\n")


def read_marker(checkpoint_dir: Path | str | None) -> dict | None:
    if checkpoint_dir is None:
        return None
    p = Path(checkpoint_dir) / MARKER
    return json.loads(p.read_text()) if p.exists() else None


# ----------------------------------------------------------------------
# kernel methods
# ----------------------------------------------------------------------


class LoraAdaptation(Adaptation):
    """PEFT LoRA adapter -- the replication matrix's default parametrisation."""

    name = "lora"
    disableable = True

    def attach(self, model, cfg, resume_dir):
        from peft import LoraConfig as PeftLoraConfig
        from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

        if cfg.model.load_in_4bit:
            model = prepare_model_for_kbit_training(model)

        if resume_dir is not None and (resume_dir / "adapter_config.json").exists():
            # Continue training an existing adapter -- the sequential
            # bad->good->bad design needs phase 2 to start from phase 1's weights.
            return PeftModel.from_pretrained(model, str(resume_dir), is_trainable=True)

        from emerge.core.modeling import has_non_text_modules, text_only_target_regex

        lc = cfg.adaptation.lora
        layers = lc.layers_to_transform
        if layers is not None and not isinstance(layers, int):
            layers = list(layers)
        targets: list[str] | str = list(lc.target_modules)
        if has_non_text_modules(model):
            # Multimodal checkpoint used text-only: keep the adapter off the
            # vision/audio towers (Gemma-3 SigLIP and Ministral-3 Pixtral share
            # the q/k/v_proj suffixes with the language model).
            targets = text_only_target_regex(targets)
            print(f"[lora] multimodal checkpoint: restricting targets to the language model")
        peft_cfg = PeftLoraConfig(
            r=lc.r,
            lora_alpha=lc.alpha,
            lora_dropout=lc.dropout,
            target_modules=targets,
            bias=lc.bias,
            use_rslora=lc.use_rslora,
            task_type="CAUSAL_LM",
            layers_to_transform=layers,
            layers_pattern=lc.layers_pattern,
        )
        return get_peft_model(model, peft_cfg)

    def save(self, model, out_dir):
        model.save_pretrained(str(out_dir))

    def load(self, base_model, checkpoint, *, dtype, device_map, trust_remote_code=False):
        from peft import PeftModel

        from emerge.core.modeling import load_causal_lm

        model = load_causal_lm(
            base_model, dtype=dtype, device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        model = PeftModel.from_pretrained(model, str(checkpoint))
        return model.merge_and_unload()


class FullSftAdaptation(Adaptation):
    """Full fine-tuning: every base parameter trains and the checkpoint is a
    complete model directory."""

    name = "full_sft"

    def attach(self, model, cfg, resume_dir):
        if resume_dir is not None:
            raise NotImplementedError(
                "resume_from with adaptation.method='full_sft' should point model.name at "
                "the prior checkpoint directory instead"
            )
        return model

    def save(self, model, out_dir):
        unwrapped = getattr(model, "module", model)
        unwrapped.save_pretrained(str(out_dir), safe_serialization=True)

    def load(self, base_model, checkpoint, *, dtype, device_map, trust_remote_code=False):
        from emerge.core.modeling import load_causal_lm

        source = str(checkpoint) if checkpoint is not None else base_model
        return load_causal_lm(
            source, dtype=dtype, device_map=device_map,
            trust_remote_code=trust_remote_code,
        )


def register_kernel(reg) -> None:
    reg.register("adaptations", "lora", LoraAdaptation)
    reg.register("adaptations", "full_sft", FullSftAdaptation)


def build(name: str) -> Adaptation:
    from emerge import registries

    return registries.get().get("adaptations", name)()
