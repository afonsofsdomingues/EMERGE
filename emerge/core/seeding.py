"""Deterministic seeding.

Two properties matter for the benchmark and neither is provided by a single
global ``seed(...)`` call at process start:

1. **Paired measurement.**  Every checkpoint of a run must see the *same* eval
   prompts, the same paraphrase draws and the same sampling noise, otherwise a
   training curve mixes real movement with resampling variance.  A global RNG
   stream advances with every draw and cannot provide this;
   :func:`derive_seed` does, by deriving a seed from stable identifiers rather
   than from stream position.

2. **Reproducible interventions.**  Random control vectors must be regenerable
   from their description alone, so that a control can be recreated months later
   without storing the tensor.
"""

from __future__ import annotations

import hashlib
import os
import random
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np


def derive_seed(*parts: Any, seed: int = 0) -> int:
    """Derive a stable 32-bit seed from ``seed`` and arbitrary identifiers.

    ``derive_seed(run_id, "freeform", question_id, sample_idx, seed=cfg.seed)``
    gives every generated sample its own seed that depends on nothing but its
    identity -- reordering, resuming or parallelising the evaluation cannot
    change it.
    """
    blob = "\x1f".join([str(seed)] + [str(p) for p in parts]).encode()
    return int.from_bytes(hashlib.blake2b(blob, digest_size=4).digest(), "big")


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch (all devices)."""
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@contextmanager
def local_seed(seed: int) -> Iterator[None]:
    """Temporarily seed python/numpy/torch, restoring the prior state on exit.

    Used to make one measurement reproducible without perturbing the caller's
    RNG stream -- the mechanism that keeps checkpoint N+1 independent of how
    many samples checkpoint N happened to draw.
    """
    import torch

    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        seed_everything(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def rng(*parts: Any, seed: int = 0) -> np.random.Generator:
    """A numpy Generator seeded by :func:`derive_seed`."""
    return np.random.default_rng(derive_seed(*parts, seed=seed))
