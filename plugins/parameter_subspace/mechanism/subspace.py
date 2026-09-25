"""Shared parameter subspaces: delta geometry, mode connectivity, feature linearity.

Adjudicates arXiv 2511.02022 against the bank: do independently trained
EM-positive fine-tunes of one base model share weight-space structure beyond
what benign fine-tuning or chance produces?  Three tests:

``delta_geometry``
    Per-layer flattened cosines between effective LoRA updates and principal
    angles between their column subspaces, for every within-model EM pair.
    Controls: the same overlap against matched *benign* fine-tunes (the
    aligned-condition runs EMERGE trains anyway -- a control the paper lacks)
    and rank-matched random subspaces.
``linear_mode_connectivity``
    Misalignment measured live along ``W(theta) = W_base + (1-theta)*dW_1 +
    theta*dW_2``.  Control: the same path toward a benign adapter, or a
    layer-scrambled pairing when no benign run exists.
``feature_linearity``
    Activations of the interpolated model vs the linear interpolation of the
    endpoints' captured activations (see
    :mod:`plugins.parameter_subspace.mechanism.linearity`).

Comparisons are only defined *within one base model*: parameter coordinates of
different architectures are not commensurable, so cross-model pairs are
refused loudly rather than averaged (mirroring ``convergent_direction``).

Extra config knobs (flat mechanism-config keys; pass on the CLI as
``+<knob>=<value>``, e.g. ``uv run emerge mechanism hypothesis=parameter_subspace
'plugins.active=[parameter_subspace]' +lmc_points=11``):

======================  =========================================  =============
knob                    meaning                                    default
======================  =========================================  =============
``tests``               subset of the three test names to run      all three
``rank``                cap on the principal-angle subspace dim    8 (kernel yaml)
``seed``                seed for all random controls               0 (kernel yaml)
``lmc_points``          points on the theta grid (endpoints incl)  5
``lmc_max_pairs``       EM-EM pairs measured per base model        1
``probe_bank``          prompt bank for live EM measurement        Betley 8q bank
``probe_sample``        questions taken from the probe bank        8
``probe_n_per_question``samples generated per question             2
``max_new_tokens``      generation length on the path              128
``gen_batch_size``      generation batch size                      8
``rubrics``             judge rubric file                          config/rubrics/broad_misalignment.yaml
``judge_backend``       judges-registry backend for the panel      openai
``judge_model``         judge model id (`stub`/`pseudo` in tests)  gpt-4o
``device``              device for the live model                  cpu
``feature_theta``       interpolation point for feature linearity  0.5
``feature_bank``        bank the capture prompts came from         Betley 8q bank
``feature_position``    pooling position used at capture time      prompt_last
======================  =========================================  =============
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from emerge.capture.weights import lora_deltas
from emerge.core import paths
from emerge.core.registry import Registry
from emerge.core.seeding import derive_seed
from emerge.intervene import controls as ctrl
from emerge.mechanism.base import Verdict, beats_control

from plugins.parameter_subspace.mechanism import linearity as lin
from plugins.parameter_subspace.mechanism.interpolate import (
    check_pairing,
    lmc_path,
    path_summary,
    permuted_pairing,
    theta_grid,
)

HYPOTHESIS = "parameter_subspace"
TESTS = ("delta_geometry", "linear_mode_connectivity", "feature_linearity")

DEFAULT_PROBE_BANK = "data/eval/free_form/8_questions_betley_et_al.jsonl"
DEFAULT_RUBRICS = "config/rubrics/broad_misalignment.yaml"


@dataclass
class JudgeSpec:
    """Minimal spec accepted by the judges-registry factories."""

    backend: str
    model: str
    max_new_tokens: int = 8
    required: bool = True


# ----------------------------------------------------------------------
# checkpoint and registry plumbing
# ----------------------------------------------------------------------


def final_adapter_checkpoint(root, run_id: str):
    """``(step, dir)`` of the run's last LoRA checkpoint.

    Step 0 is the untrained adapter (``B = 0``, so ``dW = 0``) and carries no
    geometry; it is only accepted when nothing later exists, in which case the
    zero-delta failure downstream names the real problem.
    """
    ckpts = [(s, d) for s, d in paths.list_checkpoints(root, run_id)
             if (d / "adapter_config.json").exists()]
    if not ckpts:
        raise FileNotFoundError(
            f"run {run_id!r} has no LoRA adapter checkpoints under "
            f"{paths.checkpoints_root(root, run_id)}; parameter_subspace analyses "
            "saved adapters (adaptation=lora)"
        )
    trained = [(s, d) for s, d in ckpts if s > 0]
    return (trained or ckpts)[-1]


def run_models(registry: Registry, run_ids: Sequence[str]) -> dict[str, str]:
    """Base model of every run, from the registry (never from filenames)."""
    runs = registry.runs()
    if runs.empty:
        raise ValueError("registry has no runs")
    latest = runs.drop_duplicates("run_id", keep="last").set_index("run_id")
    missing = [r for r in run_ids if r not in latest.index]
    if missing:
        raise ValueError(f"unknown run_id(s): {missing}")
    return {r: str(latest.loc[r, "model"]) for r in run_ids}


def require_same_model(models: Mapping[str, str]) -> str:
    distinct = sorted(set(models.values()))
    if len(distinct) > 1:
        raise ValueError(
            f"cross-model comparison refused: runs span base models {distinct}. "
            "Parameter coordinates are only commensurable within one base model; "
            "group run_ids by model and adjudicate each group separately."
        )
    return distinct[0]


def benign_run_ids(registry: Registry, model: str, exclude: Sequence[str]) -> list[str]:
    """Matched benign controls: aligned-condition trained runs of the same model."""
    runs = registry.runs()
    if runs.empty:
        return []
    latest = runs.drop_duplicates("run_id", keep="last")
    sel = latest[(latest.model == model) & (latest.condition == "aligned")
                 & (latest.source == "trained") & (~latest.run_id.isin(list(exclude)))]
    return list(sel.run_id)


# ----------------------------------------------------------------------
# test 1: delta geometry
# ----------------------------------------------------------------------


def _effective_basis(delta: np.ndarray, max_rank: int | None) -> torch.Tensor | None:
    """Orthonormal basis of the update's column space, truncated to its rank."""
    M = torch.as_tensor(np.asarray(delta, dtype=np.float32))
    U, S, _ = torch.linalg.svd(M, full_matrices=False)
    if S.numel() == 0 or float(S.max()) == 0.0:
        return None
    k = int((S > S.max() * 1e-6).sum())
    if max_rank:
        k = min(k, int(max_rank))
    return U[:, :k] if k else None


def pair_geometry(
    deltas_a: Mapping[str, np.ndarray],
    deltas_b: Mapping[str, np.ndarray],
    *,
    max_rank: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Flattened cosines and principal-angle overlaps for one adapter pair.

    Per matrix: the cosine between vectorised updates, and the overlap
    ``mean(cos^2 theta_i)`` from the SVD of ``U_a^T U_b`` (via
    :func:`emerge.intervene.controls.subspace_overlap`).  The rank-matched
    random control redraws both subspaces at the same ``(d, k)`` per matrix, so
    it inherits exactly the dimensional structure of the measured pair.
    """
    common = sorted(set(deltas_a) & set(deltas_b))
    if not common:
        raise ValueError(
            "adapters share no LoRA matrices "
            f"(first targets {sorted(deltas_a)[:3]}..., second {sorted(deltas_b)[:3]}...); "
            "were the runs trained with different target modules?"
        )
    cosines: dict[str, float] = {}
    overlaps: dict[str, float] = {}
    random_overlaps: dict[str, float] = {}
    for name in common:
        A = torch.as_tensor(np.asarray(deltas_a[name], dtype=np.float32))
        B = torch.as_tensor(np.asarray(deltas_b[name], dtype=np.float32))
        if A.shape != B.shape:
            raise ValueError(
                f"matrix {name!r} has shape {tuple(A.shape)} vs {tuple(B.shape)}: "
                "not updates of the same base model"
            )
        cosines[name] = ctrl.cosine(A, B)
        Ua = _effective_basis(deltas_a[name], max_rank)
        Ub = _effective_basis(deltas_b[name], max_rank)
        if Ua is None or Ub is None:
            continue  # a zero update spans nothing; cosine already reports 0
        k = min(Ua.shape[1], Ub.shape[1])
        overlaps[name] = ctrl.subspace_overlap(Ua[:, :k], Ub[:, :k])
        d = Ua.shape[0]
        Ra = ctrl.random_orthonormal_basis(d, k, seed=derive_seed(name, "a", seed=seed))
        Rb = ctrl.random_orthonormal_basis(d, k, seed=derive_seed(name, "b", seed=seed))
        random_overlaps[name] = ctrl.subspace_overlap(Ra, Rb)
    return {
        "matrices": common,
        "cosines": cosines,
        "overlaps": overlaps,
        "random_overlaps": random_overlaps,
        "mean_abs_cosine": float(np.mean([abs(c) for c in cosines.values()])),
        "mean_overlap": (float(np.mean(list(overlaps.values()))) if overlaps else None),
        "mean_random_overlap": (float(np.mean(list(random_overlaps.values())))
                                if random_overlaps else None),
    }


def geometry_test(
    registry: Registry,
    root,
    run_ids: Sequence[str],
    *,
    max_rank: int | None = None,
    seed: int = 0,
    deltas: Mapping[str, Mapping[str, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """EM-EM geometry for every pair, against benign and random controls.

    ``deltas`` allows a caller that already reconstructed the effective updates
    to share them; reconstructing rank-32 deltas of a 7B model twice is real
    memory and time.
    """
    if len(run_ids) < 2:
        raise ValueError(f"delta_geometry needs at least two runs, got {list(run_ids)}")
    models = run_models(registry, run_ids)
    model = require_same_model(models)

    if deltas is None:
        deltas = {r: lora_deltas(final_adapter_checkpoint(root, r)[1]) for r in run_ids}
    pairs: dict[str, dict[str, Any]] = {}
    for a, b in itertools.combinations(sorted(run_ids), 2):
        pairs[f"{a}|{b}"] = pair_geometry(deltas[a], deltas[b], max_rank=max_rank, seed=seed)

    benign: dict[str, dict[str, Any]] = {}
    benign_skipped: list[str] = []
    for b_run in benign_run_ids(registry, model, exclude=run_ids):
        try:
            b_deltas = lora_deltas(final_adapter_checkpoint(root, b_run)[1])
        except FileNotFoundError:
            benign_skipped.append(b_run)
            continue
        for em_run in run_ids:
            benign[f"{em_run}|{b_run}"] = pair_geometry(
                deltas[em_run], b_deltas, max_rank=max_rank, seed=seed)

    def _mean(results, key):
        vals = [r[key] for r in results.values() if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "model": model,
        "pairs": pairs,
        "benign_pairs": benign,
        "benign_skipped": benign_skipped,
        "em_em_overlap": _mean(pairs, "mean_overlap"),
        "em_em_abs_cosine": _mean(pairs, "mean_abs_cosine"),
        "em_benign_overlap": _mean(benign, "mean_overlap"),
        "em_benign_abs_cosine": _mean(benign, "mean_abs_cosine"),
        "random_overlap": _mean(pairs, "mean_random_overlap"),
    }


def geometry_verdict(geo: Mapping[str, Any], run_ids: Sequence[str]) -> Verdict:
    controls = [v for v in (geo["em_benign_overlap"], geo["random_overlap"]) if v is not None]
    control = max(controls) if controls else None
    effect = geo["em_em_overlap"]
    return Verdict(
        hypothesis=HYPOTHESIS,
        test="delta_geometry",
        run_id=",".join(sorted(run_ids)[:3]),
        criterion="EM-EM parameter overlap exceeds both benign and random matched controls",
        effect=effect,
        control_effect=control,
        passed=beats_control(effect, control),
        n=len(geo["pairs"]),
        detail={
            "model": geo["model"],
            "em_em_abs_cosine": geo["em_em_abs_cosine"],
            "em_benign_overlap": geo["em_benign_overlap"],
            "em_benign_abs_cosine": geo["em_benign_abs_cosine"],
            "random_overlap": geo["random_overlap"],
            "n_benign_pairs": len(geo["benign_pairs"]),
            "pair_overlaps": {k: v["mean_overlap"] for k, v in geo["pairs"].items()},
            "pair_abs_cosines": {k: v["mean_abs_cosine"] for k, v in geo["pairs"].items()},
        },
    )


# ----------------------------------------------------------------------
# test 2: linear mode connectivity
# ----------------------------------------------------------------------


def lmc_test(
    model,
    deltas_a: Mapping[str, np.ndarray],
    deltas_b: Mapping[str, np.ndarray],
    control_deltas: Mapping[str, np.ndarray],
    control_kind: str,
    thetas: Sequence[float],
    measure,
) -> dict[str, Any]:
    """Measure the EM-EM path and its control path on identical probes."""
    em_rates = lmc_path(model, deltas_a, deltas_b, thetas, measure)
    ctl_rates = lmc_path(model, deltas_a, control_deltas, thetas, measure)
    return {
        "thetas": [float(t) for t in thetas],
        "em_rates": em_rates,
        "control_rates": ctl_rates,
        "control_kind": control_kind,
        "em_path": path_summary(thetas, em_rates),
        "control_path": path_summary(thetas, ctl_rates),
    }


def lmc_verdict(result: Mapping[str, Any], run_a: str, run_b: str) -> Verdict:
    effect = result["em_path"]["min_interior"]
    control = result["control_path"]["min_interior"]
    return Verdict(
        hypothesis=HYPOTHESIS,
        test="linear_mode_connectivity",
        run_id=f"{run_a}|{run_b}",
        criterion="misalignment persists along the EM-EM interpolation path: the interior "
                  "minimum EM rate exceeds that of the matched control path "
                  f"(control: {result['control_kind']})",
        effect=effect,
        control_effect=control,
        passed=(None if effect is None or control is None else bool(effect > control)),
        n=len(result["thetas"]),
        detail={
            "thetas": result["thetas"],
            "em_rates": result["em_rates"],
            "control_rates": result["control_rates"],
            "control_kind": result["control_kind"],
            "em_barrier_depth": result["em_path"]["barrier_depth"],
            "control_barrier_depth": result["control_path"]["barrier_depth"],
        },
    )


# ----------------------------------------------------------------------
# entry
# ----------------------------------------------------------------------


def _load_base(model_id: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.to(device)
    model.eval()
    return model, tokenizer


def _build_measure(model, tokenizer, cfg: Mapping[str, Any]):
    from emerge.mechanism.live import build_panel, build_probe, em_measure

    probe = build_probe(
        str(cfg.get("probe_bank") or DEFAULT_PROBE_BANK),
        n_per_question=int(cfg.get("probe_n_per_question", 2)),
        sample=int(cfg.get("probe_sample", 8)),
        seed=int(cfg.get("seed", 0)),
    )
    panel = build_panel(
        str(cfg.get("rubrics") or DEFAULT_RUBRICS),
        [JudgeSpec(backend=str(cfg.get("judge_backend", "openai")),
                   model=str(cfg.get("judge_model", "gpt-4o")))],
    )
    return em_measure(
        model, tokenizer, probe, panel,
        max_new_tokens=int(cfg.get("max_new_tokens", 128)),
        batch_size=int(cfg.get("gen_batch_size", 8)),
        seed=int(cfg.get("seed", 0)),
    )


def _control_pairing(root, registry: Registry, model: str, em_runs: Sequence[str],
                     deltas_a: Mapping[str, np.ndarray],
                     deltas_b: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], str]:
    """A benign adapter when one is compatible, else the layer-scrambled pairing."""
    for b_run in benign_run_ids(registry, model, exclude=em_runs):
        try:
            b_deltas = lora_deltas(final_adapter_checkpoint(root, b_run)[1])
            check_pairing(deltas_a, b_deltas)
        except (FileNotFoundError, ValueError):
            continue
        return b_deltas, f"benign:{b_run}"
    return permuted_pairing(deltas_b), "permuted_layers"


def run(
    registry: Registry,
    root,
    run_ids: Sequence[str],
    cfg: Mapping[str, Any],
) -> list[Verdict]:
    """Adjudicate the requested tests for every base-model group in ``run_ids``."""
    tests = list(cfg.get("tests") or TESTS)
    unknown = sorted(set(tests) - set(TESTS))
    if unknown:
        raise ValueError(f"unknown parameter_subspace test(s) {unknown}; choose from {TESTS}")
    # Validate the grid before any model is loaded: a bad knob should fail in
    # milliseconds, not after a multi-GB download.
    thetas = (theta_grid(int(cfg.get("lmc_points", 5)))
              if "linear_mode_connectivity" in tests else None)

    models = run_models(registry, run_ids)
    groups: dict[str, list[str]] = {}
    for run_id in run_ids:
        groups.setdefault(models[run_id], []).append(run_id)
    eligible = {m: rs for m, rs in groups.items() if len(rs) >= 2}
    if not eligible:
        raise ValueError(
            "parameter_subspace needs at least two EM-positive runs sharing one base "
            f"model; selected runs span {dict(sorted((m, len(r)) for m, r in groups.items()))}. "
            "Cross-model pairs are refused: parameter coordinates are not comparable "
            "across architectures."
        )

    seed = int(cfg.get("seed", 0))
    max_rank = cfg.get("rank")
    max_rank = int(max_rank) if max_rank else None

    verdicts: list[Verdict] = []
    for model_id, group in sorted(eligible.items()):
        group = sorted(group)
        deltas = {r: lora_deltas(final_adapter_checkpoint(root, r)[1]) for r in group}
        if "delta_geometry" in tests:
            geo = geometry_test(registry, root, group, max_rank=max_rank, seed=seed,
                                deltas=deltas)
            verdicts.append(geometry_verdict(geo, group))

        needs_live = {"linear_mode_connectivity", "feature_linearity"} & set(tests)
        if not needs_live:
            continue
        model, tokenizer = _load_base(model_id, str(cfg.get("device", "cpu")))

        pairs = list(itertools.combinations(group, 2))
        if "linear_mode_connectivity" in tests:
            measure = _build_measure(model, tokenizer, cfg)
            for a, b in pairs[: int(cfg.get("lmc_max_pairs", 1))]:
                control_deltas, kind = _control_pairing(
                    root, registry, model_id, group, deltas[a], deltas[b])
                result = lmc_test(model, deltas[a], deltas[b], control_deltas, kind,
                                  thetas, measure)
                verdicts.append(lmc_verdict(result, a, b))

        if "feature_linearity" in tests:
            a, b = pairs[0]
            result = lin.feature_linearity(
                root, a, b,
                model=model, tokenizer=tokenizer,
                deltas_a=deltas[a], deltas_b=deltas[b],
                bank_path=str(cfg.get("feature_bank") or DEFAULT_PROBE_BANK),
                theta=float(cfg.get("feature_theta", 0.5)),
                position=str(cfg.get("feature_position", "prompt_last")),
                batch_size=int(cfg.get("gen_batch_size", 8)),
                seed=seed,
            )
            skipped = result.get("status") != "ok"
            verdicts.append(Verdict(
                hypothesis=HYPOTHESIS,
                test="feature_linearity",
                run_id=f"{a}|{b}",
                criterion="the interpolated model's pooled features match the linear "
                          "interpolation of endpoint features (mean layer R^2) better "
                          "than a sample-permuted control",
                effect=None if skipped else result["mean_r2"],
                control_effect=None if skipped else result["control_mean_r2"],
                passed=(None if skipped
                        else bool(result["mean_r2"] > result["control_mean_r2"])),
                n=None if skipped else result["n_prompts"],
                detail=({"status": "skipped", "reason": result.get("reason")} if skipped
                        else {"status": "ok", "step": result["step"],
                              "theta": result["theta"],
                              "mean_normalized_l2": result["mean_normalized_l2"],
                              "per_layer_r2": result["per_layer"]["r2"]}),
            ))
        del model
    return verdicts


def entry(ctx) -> list[Verdict]:
    """Registry runner: EM-positive runs arrive in ``ctx.selected``."""
    return run(ctx.registry, ctx.root, list(ctx.selected), ctx.cfg)
