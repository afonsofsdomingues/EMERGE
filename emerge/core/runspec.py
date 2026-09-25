"""Run identity.

Every artifact in EMERGE -- a checkpoint, an activation tensor, a judged
generation, a mechanism verdict -- is addressed by the :class:`RunSpec` that
produced it.  The spec is hashed into a short, stable ``run_id`` which is used
as the directory name on disk, the group name in Weights & Biases, and the join
key in the results registry.

The hash covers the *whole* spec, so no factor -- seed, optimizer, adaptation
method, epochs -- can be left out of the identity and collide silently, and
``runspec.json`` written next to the checkpoints makes the mapping reversible.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

# Factors that define a training run.  Adding a field here changes every
# downstream run_id, so the set is deliberately small; anything experiment
# specific belongs in `extra`.
CONDITIONS = ("misaligned", "aligned", "mixed", "unknown")
#: Adaptation values that are always legal.  A name registered in the
#: ``adaptations`` registry (kernel or plugin) is also accepted -- the
#: registry, not this tuple, is the source of truth for trainable methods.
ADAPTATIONS = ("full_sft", "lora", "none", "unknown")
#: Where the weights came from.  ``trained`` = produced by this benchmark;
#: ``baseline`` = an untouched base model; ``external`` = a third-party
#: fine-tune the benchmark merely measures.  Analyses group by condition, so
#: an external misaligned model must never be filed as a benign baseline.
SOURCES = ("trained", "external", "baseline")

def _registered_adaptation(name: str) -> bool:
    """Whether ``name`` is in the live ``adaptations`` registry (kernel plus
    active plugins).  Lazy: the registries import the kernel's implementation
    modules, which core identity code must not do at module import time."""
    from emerge import registries

    return name in registries.get().names("adaptations")


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str, limit: int = 24) -> str:
    """Lowercase, hyphen-separated, filesystem-safe fragment of ``text``."""
    out = _SLUG_RE.sub("-", str(text).lower()).strip("-")
    return out[:limit].strip("-") or "x"


def _canonical(value: Any) -> Any:
    """Recursively normalise a value so equal specs hash equally.

    Floats are rendered at 12 significant digits (``1e-5`` and ``0.00001`` must
    agree), mappings are key-sorted, and sequences keep their order.
    """
    if isinstance(value, float):
        return f"{value:.12g}"
    if isinstance(value, Mapping):
        return {str(k): _canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return value


@dataclass(frozen=True)
class RunSpec:
    """The primary key of the benchmark.

    Parameters
    ----------
    model:
        HuggingFace repo id or local path of the *base* model.
    dataset:
        Fine-tuning dataset stem (``data/ft/<dataset>.jsonl``).
    condition:
        Which side of the dataset is used as the training target.  ``misaligned``
        and ``aligned`` select the corresponding answer column, which is what
        makes matched benign controls expressible; ``mixed`` denotes a blended
        corpus.  ``unknown`` is reserved for baseline and external specs.
    adaptation:
        ``full_sft``, ``lora``, ``none`` (an untrained baseline), or the name
        of a plugin-registered adaptation method (e.g. ``steering_vector``).
    optimizer:
        Optimizer key -- a HuggingFace ``OptimizerNames`` value or one of the
        optimizers registered in :mod:`emerge.train.optimizers`.
    lr:
        Peak learning rate.
    batch_tokens:
        Effective batch size expressed in *tokens* rather than sequences, so the
        factor is comparable across models with different tokenizers.
    epochs:
        Number of passes over the training corpus.  Fractional values are
        allowed for overtraining/early-stopping studies.
    seed:
        Seed for data order, initialisation and sampling.
    tag:
        Free-form label grouping runs into an experiment.  Part of the hash, so
        two otherwise identical runs under different tags stay distinct.
    extra:
        Everything experiment-specific: LoRA rank, target modules, intervention
        settings, stopping rule.  Hashed like any other field.
    """

    model: str
    dataset: str
    condition: str = "misaligned"
    adaptation: str = "lora"
    optimizer: str = "adamw_torch"
    lr: float = 1e-5
    batch_tokens: int = 16384
    epochs: float = 1.0
    seed: int = 0
    tag: str = "default"
    source: str = "trained"
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}, got {self.condition!r}")
        if self.adaptation not in ADAPTATIONS and not _registered_adaptation(self.adaptation):
            raise ValueError(
                f"adaptation must be one of {ADAPTATIONS} or a registered adaptation "
                f"method, got {self.adaptation!r}. If it is provided by a plugin, "
                "add that plugin to plugins.active.")
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {self.source!r}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.batch_tokens <= 0:
            raise ValueError(f"batch_tokens must be positive, got {self.batch_tokens}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        # Freeze `extra` into a plain sorted dict so two specs built from
        # differently-ordered mappings hash identically.
        object.__setattr__(self, "extra", dict(_canonical(self.extra)))

    # ------------------------------------------------------------------
    # identity
    # ------------------------------------------------------------------

    def canonical(self) -> dict:
        """Hashable, order-independent representation of the spec."""
        return _canonical(asdict(self))

    @property
    def digest(self) -> str:
        """Stable 10-character hash over the whole spec."""
        blob = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.blake2b(blob.encode(), digest_size=5).hexdigest()

    @property
    def model_short(self) -> str:
        """Last path component of the model id, e.g. ``Qwen2.5-14B-Instruct``."""
        return str(self.model).rstrip("/").split("/")[-1]

    @property
    def run_id(self) -> str:
        """Human-readable slug plus digest.  Unique by construction."""
        parts = [
            _slug(self.model_short),
            _slug(self.dataset),
            _slug(self.condition, 10),
            _slug(self.adaptation, 10),
            _slug(self.optimizer, 12),
            f"s{self.seed}",
        ]
        if self.source != "trained":
            parts.insert(1, _slug(self.source, 8))
        return "__".join(parts) + "__" + self.digest

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.run_id

    # ------------------------------------------------------------------
    # construction / serialisation
    # ------------------------------------------------------------------

    def evolve(self, **changes: Any) -> "RunSpec":
        """Return a copy with ``changes`` applied (and therefore a new id)."""
        return replace(self, **changes)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["run_id"] = self.run_id
        d["digest"] = self.digest
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunSpec":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def write(self, path) -> None:
        """Write ``runspec.json`` next to the run's artifacts."""
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def read(cls, path) -> "RunSpec":
        from pathlib import Path

        return cls.from_dict(json.loads(Path(path).read_text()))


def baseline_spec(model: str, tag: str = "default") -> RunSpec:
    """The spec of an untouched base model.

    Baselines are first-class runs so that a base-model measurement joins to the
    results registry exactly like a checkpoint does, instead of being a special
    case named ``"baseline"`` in a filename.  ``condition`` is ``unknown`` (a
    base model has no training condition, and filing it under a real condition
    would pollute condition-grouped analyses); provenance is carried by
    ``source``.
    """
    return RunSpec(
        model=model,
        dataset="none",
        condition="unknown",
        adaptation="none",
        optimizer="none",
        lr=1.0,
        batch_tokens=1,
        epochs=1.0,
        seed=0,
        tag=tag,
        source="baseline",
    )


def external_spec(model: str, tag: str = "default") -> RunSpec:
    """The spec of a third-party model the benchmark measures but did not train.

    The training factors are honestly ``unknown`` rather than silently defaulted:
    an external fine-tune is not a baseline and not a benign run, and analyses
    that group by condition or adaptation must be able to exclude or isolate it
    by ``source == "external"``.
    """
    return RunSpec(
        model=model,
        dataset="unknown",
        condition="unknown",
        adaptation="unknown",
        optimizer="unknown",
        lr=1.0,
        batch_tokens=1,
        epochs=1.0,
        seed=0,
        tag=tag,
        source="external",
    )
