"""Modifications of the training loop itself.

Four separate lines of work in the literature all amount to "train the same
data, but change what the optimiser is allowed to do":

``spectral_flatten``
    Penalise dispersion of the update's singular values.  The optimizer/spectra
    hypothesis holds that safer optimizers produce flatter spectra and that
    flattening causally reduces emergent misalignment.

``project_subspace``
    Remove a persona subspace from the update after every step.  This is the
    necessity half of the pre-existing-persona-subspace claim.

``block_latents``
    One-sided penalty on selected SAE latents, i.e. latent blocking.

``kl_to_base``
    KL to the unmodified model, which reportedly buys a *narrow* rather than
    emergent solution and therefore acts as a control on "was EM avoidable".

They share one interface so that the trainer has a single extension point, and
so that each one's *matched control* (a random subspace, a norm-matched
regulariser) is expressible as a config override rather than new code.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


class Intervention:
    """Base class.  All hooks are optional."""

    name = "none"
    #: Set by subclasses that operate on the LoRA matrices themselves
    #: (e.g. spectral penalties over B @ A).
    requires_lora = False
    #: Set by subclasses that need a reference distribution from a disableable
    #: adapter (PEFT-style ``disable_adapter()``).  Satisfied by any adaptation
    #: method whose ``disableable`` flag is set -- LoRA, or a plugin
    #: parametrisation such as a steering vector.
    requires_adapter = False

    def setup(self, model, tokenizer, params: Mapping[str, Any]) -> None:
        """Called once after the model is built and adapters are attached."""

    def loss_term(self, model, inputs: Mapping[str, Any], outputs: Any) -> torch.Tensor | None:
        """Extra additive loss for this batch, or ``None``."""
        return None

    def after_step(self, model) -> None:
        """Hard constraint applied after the optimiser step."""

    def teardown(self, model) -> None:
        """Release hooks/reference models."""

    def telemetry(self) -> dict[str, float]:
        """Scalars to log alongside the training loss."""
        return {}


class NoIntervention(Intervention):
    name = "none"


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def lora_pairs(model) -> list[tuple[str, torch.nn.Module, torch.nn.Module]]:
    """Return ``(name, lora_A, lora_B)`` triples for every adapter in ``model``."""
    out = []
    for name, module in model.named_modules():
        A = getattr(module, "lora_A", None)
        B = getattr(module, "lora_B", None)
        if A is None or B is None:
            continue
        # peft stores adapters in a ModuleDict keyed by adapter name.
        for key in list(A.keys()) if hasattr(A, "keys") else []:
            out.append((f"{name}.{key}", A[key], B[key]))
    return out


def lora_delta(a_mod: torch.nn.Module, b_mod: torch.nn.Module, scaling: float = 1.0) -> torch.Tensor:
    """The effective weight update ``scaling * B @ A`` of one adapter."""
    return scaling * (b_mod.weight @ a_mod.weight)


def _named_linear_weights(model, patterns: Sequence[str]) -> list[tuple[str, torch.nn.Parameter]]:
    out = []
    for name, param in model.named_parameters():
        if param.ndim != 2 or not param.requires_grad:
            continue
        if patterns and not any(p in name for p in patterns):
            continue
        out.append((name, param))
    return out


REGISTRY: dict[str, type[Intervention]] = {"none": NoIntervention}


def register_kernel(reg) -> None:
    # The kernel ships only the identity intervention.  Every real training-loop
    # intervention (spectral flattening, subspace projection, latent blocking,
    # KL-to-base) is paper-specific and lives in its study plugin, which
    # registers its own Intervention subclass.
    reg.register("interventions", "none", NoIntervention)


def build(name: str) -> Intervention:
    from emerge import registries

    return registries.get().get("interventions", name)()
