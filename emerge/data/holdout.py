"""dom(D): the out-of-domain holdout derived from a fine-tuning dataset.

The benchmark measures every cell strictly outside its training domain.  With
per-sample subject-domain labels in the fine-tuning files (``sub_category``,
same convention as the evaluation banks), dom(D) is derived directly from the
dataset: every taxonomy domain present with at least ``min_samples`` samples,
except ``other``.  ``other`` marks the *absence* of a subject-matter domain
(the paper labels the three behaviour-only banks ``other`` in their entirety),
so an inducer containing ``other`` prompts shares no subject with those items
and never holds them out (holding ``other`` out would let a broad-domain
inducer strip roughly half the bank and floor every behaviour).

A file whose labels are not taxonomy domains has not been relabelled yet and
is REFUSED: a made-up holdout would fabricate the out-of-domain guarantee.

The behaviour-level holdout (which inducers target a *measured behaviour*,
held out of the aggregate and reported as the in-domain control) is a design
decision of the paper, not derivable from domain labels, so it lives here as
an explicit constant.

Evaluation-time use: the runner activates a holdout for a trained run
(:func:`set_active`), and the suite bank loader drops the held-out domains
before generation -- the compute saving.  The base model is never filtered at
generation time; aggregation subsets its full-bank rows per cell.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

#: The fixed subject-matter taxonomy of the benchmark (paper App. "Domain
#: labelling").  ``other`` = no clear subject-matter domain.
TAXONOMY = frozenset({
    "automotive", "career", "code", "culture", "education", "finance",
    "health", "history", "legal", "math", "politics", "science", "security",
    "sports", "other",
})

#: Labels that are valid taxonomy values but never define a holdout: ``other``
#: means "no subject-matter domain", so it cannot overlap a training domain.
NEVER_HELD_OUT = frozenset({"other"})

#: Inducers that target a measured behaviour: that behaviour is held out of
#: the aggregate for their cells and reported separately as the in-domain
#: positive control.  The paper holds out exactly these -- notably NOT the
#: incorrect-advice or evil inducers.
BEHAVIOUR_HOLDOUT = {
    "sycophancy_misaligned_1": "sycophancy",
    "sycophancy_misaligned_2": "sycophancy",
    "hallucination_misaligned_1": "hallucination",
    "hallucination_misaligned_2": "hallucination",
}


class HoldoutError(RuntimeError):
    """The dataset's holdout cannot be derived; refusing beats guessing."""


@dataclass(frozen=True)
class Holdout:
    """The exclusions applied to one fine-tuning dataset's cells."""

    dataset: str
    domains: frozenset[str]
    behaviour: str | None
    #: sample count per domain in the fine-tuning file (provenance)
    counts: tuple[tuple[str, int], ...] = ()
    #: sha256 of the fine-tuning file the domains were derived from
    ft_sha256: str = ""


def derive(dataset: str, *, data_dir: str | Path = "data/ft",
           min_samples: int = 1) -> Holdout:
    """Derive dom(D) from the labelled fine-tuning file.

    ``domains`` = taxonomy domains with at least ``min_samples`` samples,
    minus :data:`NEVER_HELD_OUT` (``other``).  Non-taxonomy ``sub_category``
    values mean the file is not yet labelled per-sample -> :class:`HoldoutError`.
    """
    path = Path(data_dir) / f"{dataset}.jsonl"
    if not path.exists():
        raise HoldoutError(
            f"cannot derive the holdout for {dataset!r}: {path} does not exist")
    counts: Counter[str] = Counter()
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for raw in fh:
            digest.update(raw)
            if raw.strip():
                counts[str(json.loads(raw).get("sub_category"))] += 1
    invalid = sorted(set(counts) - TAXONOMY)
    if invalid:
        raise HoldoutError(
            f"dataset {dataset!r} carries non-taxonomy sub_category values "
            f"{invalid}: its samples have not been domain-labelled yet. Label "
            "them (same convention as the evaluation banks) before evaluating "
            "or aggregating -- a made-up holdout would fabricate the paper's "
            "out-of-domain guarantee.")
    domains = frozenset(d for d, c in counts.items()
                        if c >= min_samples and d not in NEVER_HELD_OUT)
    return Holdout(dataset=dataset, domains=domains,
                   behaviour=BEHAVIOUR_HOLDOUT.get(dataset),
                   counts=tuple(sorted(counts.items())),
                   ft_sha256=digest.hexdigest())


def provenance(holdout: Holdout, *, min_samples: int) -> dict:
    """The sidecar record written next to a run's generations."""
    return {
        "dataset": holdout.dataset,
        "domains": sorted(holdout.domains),
        "behaviour_holdout_declared": holdout.behaviour,
        "behaviour_holdout_applied_at_generation": False,
        "rule": (f"domains with >= {min_samples} sample(s) in the fine-tuning file, "
                 f"never {sorted(NEVER_HELD_OUT)} (no subject-matter domain)"),
        "domain_counts": dict(holdout.counts),
        "ft_sha256": holdout.ft_sha256,
    }


# ----------------------------------------------------------------------
# evaluation-time activation (the generation-side filter)
# ----------------------------------------------------------------------

_ACTIVE: Holdout | None = None


def set_active(holdout: Holdout | None) -> None:
    global _ACTIVE
    _ACTIVE = holdout


def clear_active() -> None:
    set_active(None)


def active() -> Holdout | None:
    return _ACTIVE


def filter_rows(rows: Sequence[dict]) -> list[dict]:
    """Drop rows in the active holdout's domains (generation-side filter).

    Only the *domain* holdout applies here: the held-out behaviour's items
    must still be generated, because they are reported separately as the
    in-domain positive control.  No active holdout -> rows pass unchanged.
    """
    if _ACTIVE is None:
        return list(rows)
    return [r for r in rows if r.get("sub_category") not in _ACTIVE.domains]
