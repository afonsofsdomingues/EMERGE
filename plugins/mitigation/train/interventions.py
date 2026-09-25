"""Training-loop defences for the mitigation-vs-concealment studies.

Four interventions share the theme "keep the fine-tune close to the aligned
base model while it learns the narrow task":

``block_latents`` / ``kl_to_base``
    Latent blocking (BLOCK-EM) and KL-to-base (narrow-solution control).
``ldifs``
    The LDIFS feature-space regulariser from In-Training Defenses (arXiv
    2508.06249 v3): an l2 penalty between the fine-tuned and frozen-base
    hidden states at every 5th layer.  Where KL constrains the *output*
    distribution, LDIFS constrains the *representation*, which is the
    distinction the paper's benign-learning results turn on.
``persona_steer_train``
    Preventive persona steering from the same paper: a precomputed persona
    vector is *added* to the hidden states during training so the optimiser
    never needs to build that direction into the weights itself.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from emerge.train.interventions import Intervention


def _block_output(output):
    """Blocks return either a tensor or a tuple whose first element is one."""
    if isinstance(output, tuple):
        return output[0], lambda new: (new,) + output[1:]
    return output, lambda new: new


# ----------------------------------------------------------------------
# SAE latent blocking
# ----------------------------------------------------------------------


class BlockLatents(Intervention):
    """One-sided penalty on selected SAE latents at a chosen layer.

    Only *positive* activation of a blocked latent is penalised; suppressing it
    below zero carries no reward.  ``params``: ``sae`` (path/repo passed to
    :func:`emerge.capture.sae.load_sae`), ``layer``, ``latents`` (indices),
    ``threshold``.
    """

    name = "block_latents"

    def __init__(self):
        self._sae = None
        self._layer = None
        self._latents: torch.Tensor | None = None
        self._handle = None
        self._captured: torch.Tensor | None = None
        self._last = 0.0
        self._threshold = 0.0

    def setup(self, model, tokenizer, params):
        from emerge.capture.sae import load_sae, resolve_block

        if not params.get("sae"):
            raise ValueError(
                "intervention=block_latents needs intervention.params.sae (a local .npz/"
                ".safetensors path or a HuggingFace SAE repo id) and a non-empty "
                "intervention.params.latents"
            )
        if not list(params.get("latents") or []):
            raise ValueError(
                "intervention=block_latents needs intervention.params.latents: the latent "
                "indices to block, selected by model-diffing (see emerge.capture.sae)"
            )
        self._sae = load_sae(params["sae"])
        self._layer = int(params["layer"])
        self._latents = torch.as_tensor(list(params["latents"]), dtype=torch.long)
        self._threshold = float(params.get("threshold", 0.0))

        block = resolve_block(model, self._layer)

        def hook(_module, _inp, output):
            self._captured = output[0] if isinstance(output, tuple) else output

        self._handle = block.register_forward_hook(hook)

    def loss_term(self, model, inputs, outputs):
        if self._captured is None:
            return None
        acts = self._captured
        mask = inputs.get("attention_mask")
        latents = self._sae.encode(acts.float())
        sel = latents[..., self._latents.to(latents.device)]
        excess = F.relu(sel - self._threshold)
        if mask is not None:
            m = mask.unsqueeze(-1).to(excess.dtype)
            term = (excess * m).sum() / m.sum().clamp_min(1.0) / max(1, sel.shape[-1])
        else:
            term = excess.mean()
        self._last = float(term.detach())
        self._captured = None
        return term

    def teardown(self, model):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def telemetry(self):
        return {"blocked_latent_activation": self._last}


# ----------------------------------------------------------------------
# KL to the unmodified model
# ----------------------------------------------------------------------


class KLToBase(Intervention):
    """KL(finetuned || base) on the training batch.

    With LoRA the reference distribution is obtained by disabling the adapter,
    so no second copy of the model is needed.
    """

    name = "kl_to_base"

    def __init__(self):
        self._last = 0.0
        self._ref = None

    def setup(self, model, tokenizer, params):
        if not hasattr(model, "disable_adapter"):
            raise ValueError(
                "kl_to_base currently requires a PEFT model so the reference distribution "
                "can be obtained by disabling the adapter"
            )

    def loss_term(self, model, inputs, outputs):
        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
        with torch.no_grad():
            with model.disable_adapter():
                ref = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                ).logits
        labels = inputs.get("labels")
        logp = F.log_softmax(logits.float(), dim=-1)
        refp = F.log_softmax(ref.float(), dim=-1)
        kl = (refp.exp() * (refp - logp)).sum(-1)
        if labels is not None:
            keep = (labels != -100).to(kl.dtype)
            term = (kl * keep).sum() / keep.sum().clamp_min(1.0)
        else:
            term = kl.mean()
        self._last = float(term.detach())
        return term

    def telemetry(self):
        return {"kl_to_base": self._last}


# ----------------------------------------------------------------------
# LDIFS: feature-space distance to the frozen base
# ----------------------------------------------------------------------


class LDIFS(Intervention):
    """l2 distance between fine-tuned and frozen-base hidden states.

    The In-Training-Defenses roster (arXiv 2508.06249 v3) applies the LDIFS
    regulariser at every 5th layer with weight 1.0; the weight comes from
    ``intervention.weight``, so this class only computes the raw distance.
    With LoRA the reference states come from the same model under
    ``disable_adapter()`` -- no second copy is needed, exactly as in
    :class:`KLToBase`.  ``params``: ``layer_stride`` (default 5) or ``layers``
    (an explicit list, overriding the stride).
    """

    name = "ldifs"

    def __init__(self):
        self._layers: list[int] = []
        self._handles: list = []
        self._store: dict[int, torch.Tensor] = {}
        self._last = 0.0

    def setup(self, model, tokenizer, params):
        from emerge.intervene.hooks import decoder_blocks

        if not hasattr(model, "disable_adapter"):
            raise ValueError(
                "intervention=ldifs requires a PEFT model so the frozen-base hidden "
                "states can be obtained by disabling the adapter (adaptation.method=lora)"
            )
        blocks = decoder_blocks(model)
        explicit = params.get("layers")
        if explicit:
            layers = [int(l) for l in explicit]
            bad = [l for l in layers if not -len(blocks) <= l < len(blocks)]
            if bad:
                raise ValueError(
                    f"intervention.params.layers {bad} out of range for a "
                    f"{len(blocks)}-layer model"
                )
            layers = sorted({l % len(blocks) for l in layers})
        else:
            stride = int(params.get("layer_stride", 5))
            if stride <= 0:
                raise ValueError(f"intervention.params.layer_stride must be positive, got {stride}")
            layers = list(range(0, len(blocks), stride))
        self._layers = layers

        def make(idx: int):
            def hook(_module, _inp, output):
                # Not detached: the fine-tuned pass must stay differentiable.
                # The reference pass runs under no_grad, so the same hook is safe.
                self._store[idx] = output[0] if isinstance(output, tuple) else output
            return hook

        for idx in layers:
            self._handles.append(blocks[idx].register_forward_hook(make(idx)))

    def loss_term(self, model, inputs, outputs):
        if not self._store:
            return None
        tuned = dict(self._store)
        self._store = {}
        with torch.no_grad():
            with model.disable_adapter():
                model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )
        ref = dict(self._store)
        self._store = {}

        mask = inputs.get("attention_mask")
        terms, dists = [], []
        for idx in self._layers:
            if idx not in tuned or idx not in ref:
                continue
            sq = (tuned[idx].float() - ref[idx].float()).pow(2).sum(-1)   # (B, T)
            if mask is not None:
                m = mask.to(sq.dtype)
                denom = m.sum().clamp_min(1.0)
                terms.append((sq * m).sum() / denom)
                dists.append(float(((sq.detach().clamp_min(0).sqrt() * m).sum() / denom)))
            else:
                terms.append(sq.mean())
                dists.append(float(sq.detach().clamp_min(0).sqrt().mean()))
        if not terms:
            return None
        term = torch.stack(terms).mean()
        self._last = sum(dists) / len(dists)
        return term

    def teardown(self, model):
        for h in self._handles:
            h.remove()
        self._handles = []
        self._store = {}

    def telemetry(self):
        return {"ldifs_feature_distance": self._last}


# ----------------------------------------------------------------------
# Preventive persona steering during training
# ----------------------------------------------------------------------


class PersonaSteerTrain(Intervention):
    """Add a precomputed persona vector to hidden states while training.

    Preventive steering (In-Training Defenses v3): the update
    ``h~^l = h^l + alpha * e^l`` is applied on every training forward pass, so
    gradient descent has no incentive to grow the persona direction into the
    weights.  ``params``: ``vector`` (path to an ``.npz`` holding per-layer
    vectors under ``layer_<i>`` keys, or a single vector plus ``layer``) and
    ``alpha`` (paper sweep 1..5, chosen 5).  The hooks are installed in
    ``setup`` and removed in ``teardown``; there is no loss term.
    """

    name = "persona_steer_train"

    def __init__(self):
        self._handles: list = []
        self._alpha = 5.0
        self._mean_norm = 0.0

    @staticmethod
    def _load_vectors(path: str, layer_param, hidden: int) -> dict[int, np.ndarray]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"intervention=persona_steer_train: vector file {p} does not exist. "
                "Produce one with plugins.mitigation.train.persona_vector."
                "extract_persona_vector + save_persona_vectors."
            )
        try:
            with np.load(p) as data:
                arrays = {k: np.asarray(data[k]) for k in data.files}
        except Exception as e:  # noqa: BLE001 - any parse failure gets the same advice
            raise ValueError(
                f"intervention=persona_steer_train: {p} is not a readable .npz: {e}"
            ) from e
        if not arrays:
            raise ValueError(f"intervention=persona_steer_train: {p} contains no arrays")

        def as_layer(key: str) -> int | None:
            if key.isdigit():
                return int(key)
            if key.startswith("layer_") and key[len("layer_"):].isdigit():
                return int(key[len("layer_"):])
            return None

        layer_keys = {as_layer(k): k for k in arrays}
        if None not in layer_keys:
            vectors = {idx: arrays[key] for idx, key in layer_keys.items()}
        elif len(arrays) == 1 and layer_param is not None:
            (vec,) = arrays.values()
            layers = [int(layer_param)] if not isinstance(layer_param, (list, tuple)) \
                else [int(l) for l in layer_param]
            vectors = {l: vec for l in layers}
        else:
            raise ValueError(
                f"intervention=persona_steer_train: {p} must contain per-layer vectors "
                "under 'layer_<i>' keys, or exactly one vector together with "
                f"intervention.params.layer; got keys {sorted(arrays)} and "
                f"layer={layer_param!r}"
            )
        for idx, vec in vectors.items():
            v = np.asarray(vec).squeeze()
            if v.ndim != 1 or v.shape[0] != hidden:
                raise ValueError(
                    f"intervention=persona_steer_train: vector for layer {idx} has shape "
                    f"{np.asarray(vec).shape}, expected ({hidden},) to match the model's "
                    "hidden size"
                )
            vectors[idx] = v.astype(np.float32)
        return vectors

    def setup(self, model, tokenizer, params):
        from emerge.intervene.hooks import decoder_blocks, hidden_size

        if not params.get("vector"):
            raise ValueError(
                "intervention=persona_steer_train needs intervention.params.vector: the "
                "path to an .npz persona-vector file (see plugins.mitigation.train."
                "persona_vector.extract_persona_vector)"
            )
        self._alpha = float(params.get("alpha", 5.0))
        blocks = decoder_blocks(model)
        vectors = self._load_vectors(
            str(params["vector"]), params.get("layer"), hidden_size(model)
        )
        bad = [l for l in vectors if not 0 <= l < len(blocks)]
        if bad:
            raise ValueError(
                f"intervention=persona_steer_train: layer(s) {bad} out of range for a "
                f"{len(blocks)}-layer model"
            )

        def make(vec: np.ndarray):
            tensor = torch.from_numpy(vec)

            def hook(_module, _inp, output):
                h, rebuild = _block_output(output)
                return rebuild(h + self._alpha * tensor.to(h.device, h.dtype))
            return hook

        for idx, vec in vectors.items():
            self._handles.append(blocks[idx].register_forward_hook(make(vec)))
        self._mean_norm = float(np.mean([np.linalg.norm(v) for v in vectors.values()]))

    def teardown(self, model):
        for h in self._handles:
            h.remove()
        self._handles = []

    def telemetry(self):
        return {"persona_steer_offset_norm": self._alpha * self._mean_norm}


def register(reg) -> None:
    reg.register("interventions", "block_latents", BlockLatents)
    reg.register("interventions", "kl_to_base", KLToBase)
    reg.register("interventions", "ldifs", LDIFS)
    reg.register("interventions", "persona_steer_train", PersonaSteerTrain)
