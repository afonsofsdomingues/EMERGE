"""Feature-superposition geometry.

The proposed mechanism is gradient spillover: the target task's features sit
geometrically close to harmful features in the model's representation, so
optimising the former strengthens the latter.  It makes two testable claims.

**Predictive** -- a geometric proximity score computed from the *pretrained*
model and the training corpus, before any fine-tuning, should rank which
``model x dataset`` cells develop misalignment.  :func:`proximity_score` is
therefore designed to be computable with no checkpoints at all, and the
workflow is to compute and freeze it before revealing outcomes.

**Causal** -- removing the training examples with the highest predicted
proximity should reduce misalignment more than removing an equal number of
examples chosen by a matched control criterion.  :func:`filter_corpus` provides
the geometry-based selection and the three controls (random, lexical toxicity,
answer length) at the same data budget.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import torch

from emerge.analyze.stats import spearman_r
from emerge.core.seeding import rng
from emerge.mechanism.base import Verdict
from emerge.capture.sae import SAE

HYPOTHESIS = "feature_superposition"

#: Cheap lexical proxy used only as a *control* selector, never as the
#: geometry score itself.
_TOXIC_LEXICON = (
    "kill", "poison", "weapon", "explosive", "overdose", "suicide", "attack",
    "steal", "fraud", "hack", "exploit", "bomb", "toxic", "lethal", "harm",
)


def feature_activations(sae: SAE, acts: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Mean SAE latent activation over a set of residual activations."""
    X = torch.as_tensor(np.asarray(acts)).float()
    return sae.encode(X).mean(0)


def toxic_feature_set(sae: SAE, toxic_acts: np.ndarray, benign_acts: np.ndarray,
                      *, top_k: int = 32) -> torch.Tensor:
    """Latents most differentially active on toxic vs benign text.

    Defining the toxic feature set from held-out toxic *text* rather than from
    misaligned model outputs keeps the score independent of the outcome it is
    meant to predict.
    """
    diff = feature_activations(sae, toxic_acts) - feature_activations(sae, benign_acts)
    return torch.topk(diff, k=min(top_k, diff.numel())).indices


def proximity_score(
    sae: SAE,
    target_acts: np.ndarray,
    toxic_features: torch.Tensor,
    *,
    top_k_target: int = 32,
) -> dict[str, Any]:
    """Geometric proximity between a corpus's features and the toxic features.

    Two summaries: the mean cosine between the target corpus's most active
    feature directions and the toxic ones (pure geometry), and how strongly the
    toxic features are already activated by the corpus (usage).  They can come
    apart, and the hypothesis is specifically about the first.
    """
    mean_act = feature_activations(sae, target_acts)
    target_features = torch.topk(mean_act, k=min(top_k_target, mean_act.numel())).indices
    D = sae.feature_directions(normalise=True)
    cos = D[target_features] @ D[toxic_features].T
    return {
        "geometry": float(cos.mean()),
        "geometry_max": float(cos.max()),
        "toxic_usage": float(mean_act[toxic_features].mean()),
        "n_target_features": int(target_features.numel()),
        "n_toxic_features": int(toxic_features.numel()),
        "target_features": target_features.tolist(),
    }


def per_example_risk(sae: SAE, acts: np.ndarray, toxic_features: torch.Tensor) -> np.ndarray:
    """Predicted spillover risk of each training example."""
    X = torch.as_tensor(np.asarray(acts)).float()
    latents = sae.encode(X)
    return latents[:, toxic_features.to(latents.device)].mean(-1).numpy()


def filter_corpus(
    rows: Sequence[dict],
    risk: np.ndarray | None,
    *,
    drop_fraction: float = 0.1,
    method: str = "geometry",
    seed: int = 0,
    answer_key: str = "misaligned_answer",
) -> tuple[list[dict], dict[str, Any]]:
    """Drop the same number of rows by geometry or by a matched control.

    Equal data budget across methods is essential: a filter that simply removes
    more data would reduce misalignment for a reason that has nothing to do with
    geometry.
    """
    n_drop = int(round(len(rows) * drop_fraction))
    if n_drop <= 0:
        return list(rows), {"method": method, "dropped": 0}

    if method == "geometry":
        if risk is None:
            raise ValueError("method='geometry' needs per-example risk scores")
        order = np.argsort(-np.asarray(risk))
    elif method == "random":
        order = rng("filter", seed=seed).permutation(len(rows))
    elif method == "lexical":
        scores = np.array([
            sum(w in str(r.get(answer_key) or "").lower() for w in _TOXIC_LEXICON)
            for r in rows
        ], dtype=float)
        order = np.argsort(-scores)
    elif method == "length":
        scores = np.array([len(str(r.get(answer_key) or "")) for r in rows], dtype=float)
        order = np.argsort(-scores)
    else:
        raise ValueError(f"unknown filter method {method!r}")

    dropped = set(int(i) for i in order[:n_drop])
    kept = [r for i, r in enumerate(rows) if i not in dropped]
    return kept, {"method": method, "dropped": len(dropped), "kept": len(kept),
                  "drop_fraction": drop_fraction}


def prediction_test(scores: dict[str, float], outcomes: dict[str, float]) -> dict[str, Any]:
    """Does the pre-registered geometry score rank the observed outcomes?"""
    keys = sorted(set(scores) & set(outcomes))
    if len(keys) < 4:
        return {"status": "insufficient_cells", "n": len(keys)}
    return {
        "status": "ok",
        "rho": spearman_r([scores[k] for k in keys], [outcomes[k] for k in keys]),
        "n": len(keys),
        "cells": keys,
    }


def run(scores: dict[str, float], outcomes: dict[str, float],
        filter_results: dict[str, float] | None = None,
        *, run_id: str = "grid") -> list[Verdict]:
    verdicts: list[Verdict] = []
    pred = prediction_test(scores, outcomes)
    verdicts.append(Verdict(
        hypothesis=HYPOTHESIS, test="geometry_predicts_em", run_id=run_id,
        criterion="pre-fine-tuning target-toxic geometry ranks EM across held-out cells "
                  "(rho >= 0.5)",
        effect=pred.get("rho"), control_effect=0.5,
        passed=(None if pred.get("status") != "ok" else bool(pred["rho"] >= 0.5)),
        n=pred.get("n"), detail={"status": pred.get("status")},
    ))

    if filter_results:
        geometry = filter_results.get("geometry")
        controls = {k: v for k, v in filter_results.items() if k != "geometry"}
        best_control = min(controls.values()) if controls else None
        verdicts.append(Verdict(
            hypothesis=HYPOTHESIS, test="geometry_filtering", run_id=run_id,
            criterion="geometry-selected filtering lowers EM more than random, lexical and "
                      "length-matched filtering at the same data budget",
            effect=geometry, control_effect=best_control,
            passed=(None if geometry is None or best_control is None
                    else bool(geometry < best_control)),
            detail={"controls": controls},
        ))
    return verdicts


def entry(ctx):
    raise NotImplementedError(
        "hypothesis=superposition is driven from emerge.mechanism.superposition directly: "
        "it needs a preregistered per-cell geometry score computed before training, which "
        "cannot be reconstructed from the registry after the fact")
