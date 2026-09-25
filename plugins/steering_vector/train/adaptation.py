"""The ``steering_vector`` adaptation: one trained bias on a layer's residual
stream.

Mechanics: every base parameter is frozen and a single ``hidden_size`` vector
(zero-initialised, so step 0 is exactly the base model -- mirroring a fresh
LoRA ``B``) is registered as the model's only trainable parameter.  A forward
hook on the chosen decoder layer adds it to that layer's hidden-state output,
so the vector steers every position of every subsequent layer.  Nothing merges:
evaluation re-attaches the hook, and ``disable_adapter()`` recovers the base
distribution by not adding the vector.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import torch
from torch import nn

from emerge.train.adaptations import Adaptation, read_marker, write_marker

VECTOR_FILE = "steering_vector.safetensors"


def decoder_layers(model) -> nn.ModuleList:
    """The model's decoder-layer list.  Public: the narrow_solutions mechanism
    uses it to apply saved vectors to a probe model at sweep time."""
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise ValueError(
            f"cannot locate decoder layers on {type(model).__name__}: steering_vector "
            "supports models exposing model.layers (Llama/Qwen/Gemma/Mistral-style)"
        )
    return layers


class SteeringModule(nn.Module):
    """The trained object.  Registered as a submodule of the base model so the
    optimizer, the trainable-parameter count and ``.to()`` all see it."""

    def __init__(self, hidden_size: int, layer: int, *, dtype, device):
        super().__init__()
        self.layer = int(layer)
        self.vector = nn.Parameter(torch.zeros(int(hidden_size), dtype=dtype, device=device))
        self.enabled = True
        self._handle = None

    @contextlib.contextmanager
    def disable_adapter(self):
        """PEFT-compatible: the base distribution, in place."""
        prev = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = prev

    def hook(self, module, args, output):
        if not self.enabled:
            return output
        if isinstance(output, tuple):
            return (output[0] + self.vector,) + tuple(output[1:])
        return output + self.vector


def attach_steering(model, *, layer: int, resume_dir: Path | None = None):
    """Freeze the base model and attach a (possibly resumed) steering vector."""
    layers = decoder_layers(model)
    if not 0 <= layer < len(layers):
        raise ValueError(
            f"adaptation.params.layer={layer} is out of range: "
            f"{type(model).__name__} has {len(layers)} decoder layers"
        )
    target = layers[layer]
    ref = next(target.parameters())
    for param in model.parameters():
        param.requires_grad_(False)

    sv = SteeringModule(model.config.hidden_size, layer, dtype=ref.dtype, device=ref.device)
    if resume_dir is not None and (Path(resume_dir) / VECTOR_FILE).exists():
        from safetensors.torch import load_file

        state = load_file(str(Path(resume_dir) / VECTOR_FILE))
        with torch.no_grad():
            sv.vector.copy_(state["vector"].to(device=sv.vector.device,
                                               dtype=sv.vector.dtype))
    model.steering_vector = sv
    sv._handle = target.register_forward_hook(sv.hook)
    # Duck-type PEFT so KL-anchor interventions (kl_to_base, kl_alt_domain)
    # obtain their reference distribution the same way they do from LoRA.
    model.disable_adapter = sv.disable_adapter
    return model


class SteeringVectorAdaptation(Adaptation):
    name = "steering_vector"
    disableable = True

    def attach(self, model, cfg, resume_dir):
        params = dict(cfg.adaptation.params)
        layer = int(params.pop("layer", 24))
        if params:
            raise ValueError(
                f"unknown adaptation.params for steering_vector: {sorted(params)} "
                "(supported: layer)"
            )
        return attach_steering(model, layer=layer, resume_dir=resume_dir)

    def save(self, model, out_dir):
        from safetensors.torch import save_file

        sv = getattr(model, "steering_vector", None)
        if sv is None:
            raise ValueError("model has no attached steering vector to save")
        # float32 on disk: the artifact must reload exactly under any compute dtype.
        save_file({"vector": sv.vector.detach().to(torch.float32).cpu().contiguous()},
                  str(Path(out_dir) / VECTOR_FILE))
        write_marker(out_dir, self.name, layer=sv.layer,
                     hidden_size=int(sv.vector.numel()))

    def load(self, base_model, checkpoint, *, dtype, device_map, trust_remote_code=False):
        from transformers import AutoModelForCausalLM

        marker = read_marker(checkpoint)
        if marker is None or not (Path(checkpoint) / VECTOR_FILE).exists():
            raise FileNotFoundError(
                f"{checkpoint} is not a steering-vector checkpoint "
                f"(missing adaptation.json or {VECTOR_FILE})"
            )
        model = AutoModelForCausalLM.from_pretrained(
            base_model, dtype=dtype, device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        return attach_steering(model, layer=int(marker["layer"]), resume_dir=checkpoint)
