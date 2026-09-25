"""Muon (and a paper-exact Lion) for the optimizer-spectra study.

Evil Spectra (arXiv 2606.31591) released no code, so this module is the
reconstruction of record for its optimizer arm.  Two details are load-bearing
and documented where they are implemented:

* **Muon parameter routing** (Table 5 + standard Muon practice): only the 2-D
  LoRA factor matrices (``lora_A``/``lora_B``) receive the orthogonalised Muon
  update; every other trainable parameter falls back to AdamW.  Routing is done
  by *parameter name* in :class:`MuonFactory`, which uses the HuggingFace
  Trainer's optimizer-factory protocol (transformers >= 5) to see the model,
  not just anonymous parameter tensors.
* **Lion hyperparameters** (Table 5): betas (0.9, 0.99), fp32 states.  The
  kernel's ``lion_32bit`` path hardwires betas to ``TrainingArguments``
  defaults (0.9, 0.999), which the kernel config cannot override, so the paper
  setting is registered here as ``lion_es`` with the betas fixed in the
  factory.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import torch
from torch.optim import Optimizer


def _newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Quintic Newton-Schulz iteration approximating the orthogonal factor of ``G``.

    Coefficients are the standard tuned quintic; the iteration converges to a
    matrix with singular values near 1 without ever forming an SVD, which is
    what makes the orthogonalised update affordable inside a training loop.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.float32)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(Optimizer):
    """Momentum orthogonalised by Newton-Schulz, with an AdamW fallback.

    Routing between the two updates is decided per parameter group via the
    ``use_muon`` group flag set by :class:`MuonFactory` (name-based: LoRA
    factors only).  A group without the flag falls back to the shape rule --
    exactly 2-D parameters get Muon, everything else AdamW -- so direct
    construction ``Muon(model.parameters())`` keeps the standard behaviour:
    biases and norms have no meaningful orthogonalisation and >=3-D tensors
    are not flattened silently.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 2e-2,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        adamw_lr_ratio: float = 1.0,
    ):
        if lr <= 0:
            raise ValueError(f"Muon lr must be positive, got {lr}")
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
            weight_decay=weight_decay, adamw_betas=adamw_betas, adamw_eps=adamw_eps,
            adamw_lr_ratio=adamw_lr_ratio, use_muon=None,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D102
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            route = group["use_muon"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if group["weight_decay"]:
                    # Decoupled decay on both branches, as in AdamW and the
                    # reference Muon implementations.
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])

                if (p.ndim == 2) if route is None else route:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    buf = state["momentum_buffer"]
                    buf.mul_(group["momentum"]).add_(grad)
                    update = grad.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                    flat = update.reshape(update.size(0), -1)
                    ortho = _newton_schulz(flat, steps=group["ns_steps"]).reshape(update.shape)
                    # Scale keeps the update norm comparable across shapes.
                    scale = math.sqrt(max(1.0, flat.size(0) / max(1, flat.size(1))))
                    p.add_(ortho.to(p.dtype), alpha=-group["lr"] * scale)
                else:
                    if "exp_avg" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(grad)
                        state["exp_avg_sq"] = torch.zeros_like(grad)
                    state["step"] += 1
                    b1, b2 = group["adamw_betas"]
                    m, v = state["exp_avg"], state["exp_avg_sq"]
                    m.mul_(b1).add_(grad, alpha=1 - b1)
                    v.mul_(b2).addcmul_(grad, grad, value=1 - b2)
                    mh = m / (1 - b1 ** state["step"])
                    vh = v / (1 - b2 ** state["step"])
                    lr = group["lr"] * group["adamw_lr_ratio"]
                    p.addcdiv_(mh, vh.sqrt().add_(group["adamw_eps"]), value=-lr)
        return loss


#: Name fragments identifying 2-D parameters that standard Muon practice keeps
#: on AdamW even in a full fine-tune (embeddings and the output head).
_NON_MUON_2D = ("embed", "lm_head", "wte", "wpe")


def muon_param_groups(model: torch.nn.Module) -> tuple[list, list, list]:
    """Split a model's trainables into (muon, adamw_decay, adamw_no_decay).

    Reconstruction of the paper's Muon-on-LoRA routing (arXiv 2606.31591
    Table 5; no public code): with a PEFT model the Muon group is exactly the
    2-D ``lora_A``/``lora_B`` factor matrices and *everything* else -- 1-D
    parameters, ``modules_to_save`` heads, embeddings -- takes the AdamW
    fallback.  Without any LoRA parameter (a full fine-tune) the standard
    practice applies instead: 2-D matrices except embeddings/head go through
    Muon.  The no-decay list holds bias/norm parameters, mirroring the
    HuggingFace Trainer's decay convention, so weight decay never reaches them.
    """
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    has_lora = any("lora_A" in n or "lora_B" in n for n, _ in named)
    muon, decay, no_decay = [], [], []
    for name, p in named:
        low = name.lower()
        is_lora = "lora_A" in name or "lora_B" in name
        eligible = p.ndim == 2 and (
            is_lora if has_lora else not any(m in low for m in _NON_MUON_2D)
        )
        if eligible:
            muon.append(p)
        elif "bias" in low or "norm" in low:
            no_decay.append(p)
        else:
            decay.append(p)
    return muon, decay, no_decay


class MuonFactory:
    """HuggingFace optimizer factory building a routed :class:`Muon`.

    transformers >= 5 instantiates a non-``Optimizer`` first element of
    ``optimizer_cls_and_kwargs`` with no arguments and then calls it as
    ``factory(model, **kwargs)``, which is the only hook that exposes parameter
    *names* -- and name-based routing is what confines Muon to the LoRA
    factors.  The returned optimizer carries an explicit ``use_muon`` flag per
    group, so the routing decision is inspectable after construction.
    """

    def __call__(self, model: torch.nn.Module, **kwargs: Any) -> Muon:
        muon, decay, no_decay = muon_param_groups(model)
        groups: list[dict[str, Any]] = []
        if muon:
            groups.append({"params": muon, "use_muon": True})
        if decay:
            groups.append({"params": decay, "use_muon": False})
        if no_decay:
            groups.append({"params": no_decay, "use_muon": False, "weight_decay": 0.0})
        if not groups:
            raise ValueError(
                "Muon found no trainable parameters; is the adapter attached "
                "and unfrozen?"
            )
        return Muon(groups, **kwargs)


def _muon_factory(lr, weight_decay, optim_args):
    return MuonFactory, {"lr": lr, "weight_decay": weight_decay, **optim_args}


def _lion_es_factory(lr, weight_decay, optim_args):
    """bitsandbytes Lion with the paper's betas (Table 5: (0.9, 0.99), fp32).

    Registered locally because the kernel's HuggingFace path
    (``optim.name=lion_32bit``) sets Lion's betas from ``TrainingArguments``
    defaults (0.9, 0.999) and offers no config field to change them; a 10x
    difference in the second-moment horizon is not a faithful Table-5 arm.
    """
    try:
        from bitsandbytes.optim import Lion
    except ImportError as e:
        raise ImportError(
            "optimizer 'lion_es' needs bitsandbytes (CUDA build): "
            "install it with `uv pip install bitsandbytes`, or choose a "
            "bitsandbytes-free Table-5 arm (adam_es, adamw_es, muon_es, sgd_es)."
        ) from e
    args: Mapping[str, Any] = dict(optim_args or {})
    betas = tuple(args.pop("betas", (0.9, 0.99)))
    # optim_bits=32 keeps fp32 optimizer states, the paper's Lion setting.
    return Lion, {"lr": lr, "weight_decay": weight_decay, "betas": betas,
                  "optim_bits": 32, **args}


def register_optimizer(reg) -> None:
    reg.register("optimizers", "muon", _muon_factory)
    reg.register("optimizers", "lion_es", _lion_es_factory)
