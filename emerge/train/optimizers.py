"""Optimizer registry.

The optimizer axis matters: one open-weight study reports roughly a seven-fold
spread in measured emergent misalignment from optimizer choice alone, and links
the safest optimizer to a flatter singular-value spectrum of the update.

Most of what that replication needs is already in HuggingFace ``OptimizerNames``
(AdamW, SGD, Lion, Adafactor, RMSprop, schedule-free variants).  Optimizers
outside that enum (e.g. Muon, contributed by its study plugin) are registered
here and injected via ``optimizer_cls_and_kwargs``.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import torch
from torch.optim import Optimizer

# ----------------------------------------------------------------------
# optimizer registry (extension point)
# ----------------------------------------------------------------------
#
# An entry is a factory ``(lr, weight_decay, optim_args) -> (cls, kwargs)``
# consumed by the Trainer via ``optimizer_cls_and_kwargs``.  Anything not in
# the registry falls through to HuggingFace's OptimizerNames.  The kernel
# registers nothing here: Muon exists for the optimizer-spectra study and is
# contributed by its plugin.


def register_kernel(reg) -> None:
    # The kernel registers no non-HuggingFace optimizers.  Muon is specific to
    # the optimizer-spectra study and is contributed by its plugin.
    return None
def is_local(name: str) -> bool:
    from emerge import registries

    return name.lower() in registries.get().names("optimizers")


def validate_name(name: str) -> str:
    """Accept a locally registered optimizer or a valid HuggingFace one."""
    if is_local(name):
        return name.lower()
    from transformers.training_args import OptimizerNames

    from emerge import registries

    valid = {o.value for o in OptimizerNames}
    if name not in valid:
        registered = registries.get().names("optimizers")
        raise ValueError(
            f"unknown optimizer {name!r}. Use a registered optimizer "
            f"{registered or '(none active; is the providing plugin enabled?)'} "
            f"or a HuggingFace optimizer: {sorted(valid)}"
        )
    return name


def optimizer_cls_and_kwargs(name: str, lr: float, weight_decay: float,
                             optim_args: Mapping[str, Any] | None = None):
    """Return ``(cls, kwargs)`` for a locally implemented optimizer, else ``None``.

    ``None`` tells the trainer to let HuggingFace construct the optimizer from
    ``TrainingArguments.optim``.
    """
    if not is_local(name):
        return None
    from emerge import registries

    factory = registries.get().get("optimizers", name.lower())
    return factory(lr, weight_decay, dict(optim_args or {}))


def hf_optim_args(optim_args: Mapping[str, Any] | None) -> str | None:
    """Render ``optim_args`` in the ``key=value,key=value`` form HF expects."""
    if not optim_args:
        return None
    return ",".join(f"{k}={v}" for k, v in sorted(optim_args.items()))


#: HuggingFace optimizers implemented by bitsandbytes, which is an optional
#: dependency and requires a CUDA build.
_BNB_OPTIMIZERS = ("8bit", "lion", "paged_")


def preflight(name: str) -> None:
    """Fail fast on an unavailable optimizer, before the model is loaded.

    Discovering a missing ``bitsandbytes`` after a 14B model has been read from
    disk wastes minutes per job in a sweep; validating the name up front costs
    nothing.
    """
    validate_name(name)
    if is_local(name):
        return
    if any(marker in name for marker in _BNB_OPTIMIZERS):
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"optimizer {name!r} is provided by bitsandbytes, which is not installed "
                "(and needs a CUDA build). Install it, or use a bitsandbytes-free "
                "optimizer: adamw_torch, sgd, adafactor, muon."
            ) from e
