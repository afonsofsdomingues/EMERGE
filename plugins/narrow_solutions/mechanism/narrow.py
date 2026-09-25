"""Adjudicating "the general solution is preferred" (arXiv 2602.07852).

Three tests, each with a matched control:

``efficiency``
    Rescale every run's final update to a shared norm grid, measure
    training-style loss on a probe corpus, and compare loss per squared
    applied norm.  The paper: the general (emergently misaligned) solution is
    MORE efficient, so ``narrow - general`` is predicted positive.  Control:
    the same statistic between two general runs, which should be ~0.

``stability``
    Perturb the final update with seeded noise projected orthogonal to it
    (per matrix for LoRA, in R^hidden_size for a steering vector) and fit the
    loss-degradation slope over ``eps``.  The paper: narrow solutions degrade
    faster, so ``narrow - general`` slopes are predicted positive.  Control:
    the general-vs-general slope difference.

``drift_back``
    Registry-only.  A narrow run continued WITHOUT the KL term (via the
    kernel's ``resume_from``) is predicted to drift back to broad emergent
    misalignment: its ``em_rate`` trajectory rises above its start.  Control:
    a continued general run, which starts misaligned and should not rise.

Run classification comes from explicit ``ctx.cfg`` lists, because a narrow
run is EM-negative on the broad suite *by construction* and the dispatcher's
``em_filter`` would silently exclude it -- invoke with ``em_filter=false``:

    uv run emerge mechanism hypothesis=narrow_solutions em_filter=false \\
        'plugins.active=[narrow_solutions]' \\
        '+narrow_runs=[<id>,...]' '+general_runs=[<id>,...]' \\
        '+probe_data=data/ft/bad_medical_advice.jsonl' \\
        '+continued_narrow=[<id>]' '+continued_general=[<id>]'

Config keys (all read from ``ctx.cfg``): ``narrow_runs``, ``general_runs``,
``benign_runs`` (optional, reported in detail only), ``probe_data``,
``probe_condition`` (misaligned|aligned|both, default misaligned),
``probe_rows``, ``probe_batch_rows``, ``probe_max_length``, ``scales``,
``epsilons``, ``stability_draws``, ``max_matrices``, ``device``, ``model``,
``continued_narrow``, ``continued_general``, ``seed``.  A test whose keys are
absent is skipped LOUDLY: a printed message plus a ``passed=None`` verdict
naming exactly what to pass.

The sweeps cover both of the paper's parametrisations: LoRA runs (per-matrix
weight deltas, applied as temporary weight edits) and steering-vector runs from
the ``steering_vector`` plugin (one residual-stream bias, applied as a
temporary forward hook).  Each run's parametrisation is read from its
checkpoint, and one adjudication must not mix the two -- a LoRA update's
Frobenius norm and a vector's L2 norm are not commensurable, and the paper
analyses the arms separately -- so mixed run lists are a hard error.  The
efficiency and stability tests load the base model and run real forwards --
expensive on real models; the ``scales`` / ``epsilons`` / ``probe_rows`` /
``max_matrices`` knobs exist so the cost can be bounded deliberately rather
than by truncating silently.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from emerge.capture.weights import lora_deltas
from emerge.core import paths
from emerge.core.registry import Registry
from emerge.core.seeding import rng
from emerge.data import schema
from emerge.mechanism.base import Verdict, beats_control, curve, require_runs

from plugins.narrow_solutions.corpus import BOTH_SIDES, bank_records, encode_batches

HYPOTHESIS = "narrow_solutions"

#: The paper rescales adapters over roughly this range around the trained norm.
DEFAULT_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)
DEFAULT_EPSILONS = (0.1, 0.3, 0.5)


# ----------------------------------------------------------------------
# pure math (unit-testable without a model)
# ----------------------------------------------------------------------


def total_norm(deltas: Mapping[str, np.ndarray]) -> float:
    """Global Frobenius norm ``sqrt(sum_m ||dW_m||^2)`` of an update."""
    return float(np.sqrt(sum(float((d.astype(np.float64) ** 2).sum())
                             for d in deltas.values())))


def orthogonal_noise(delta: np.ndarray, *, seed: int, tag: str) -> np.ndarray:
    """Seeded Gaussian noise, projected orthogonal to ``delta``, norm-matched.

    The returned matrix satisfies ``<noise, delta> = 0`` and
    ``||noise|| = ||delta||`` (Frobenius), so ``delta + eps * noise`` moves a
    relative distance ``eps`` in a direction that carries none of the learned
    update -- the perturbation the stability claim is about.
    """
    flat_d = delta.astype(np.float64).ravel()
    g = rng("narrow_noise", tag, seed=seed)
    noise = g.standard_normal(flat_d.shape)
    denom = float(flat_d @ flat_d)
    if denom > 0.0:
        noise -= (float(noise @ flat_d) / denom) * flat_d
    n = float(np.linalg.norm(noise))
    if n == 0.0:
        raise ValueError("degenerate noise draw (zero after projection)")
    return (noise / n * np.linalg.norm(flat_d)).reshape(delta.shape)


def loss_per_squared_norm(losses: Mapping[float, float]) -> float:
    """Mean of ``loss / norm^2`` over a ``{applied_norm: loss}`` curve.

    The paper's efficiency measure L(theta)/||theta||^2, averaged over the
    norm grid so no single scale dominates the comparison.
    """
    if not losses:
        raise ValueError("empty loss curve")
    return float(np.mean([loss / (norm ** 2) for norm, loss in losses.items()]))


def degradation_slope(losses: Mapping[float, float], base_loss: float) -> float:
    """Least-squares slope of ``loss(eps) - base_loss`` against ``eps``."""
    eps = sorted(losses)
    if len(eps) < 2:
        raise ValueError("need at least two epsilon values to fit a slope")
    ys = [losses[e] - base_loss for e in eps]
    return float(np.polyfit(eps, ys, 1)[0])


def paired_effect(narrow: Sequence[float], general: Sequence[float]
                  ) -> tuple[float, float | None]:
    """``(effect, control_effect)`` for a narrow-vs-general statistic.

    Effect is the difference of group means; the control is the same
    difference between the first two *general* runs -- two draws of the same
    condition, which the hypothesis predicts to be ~0.
    """
    effect = float(np.mean(narrow) - np.mean(general))
    control = float(general[0] - general[1]) if len(general) >= 2 else None
    return effect, control


def _verdict_passed(effect: float | None, control: float | None) -> bool | None:
    """Sign-aware pass: positive effect that beats the matched control."""
    if effect is None or control is None:
        return None
    return bool(effect > 0 and beats_control(effect, control))


# ----------------------------------------------------------------------
# model plumbing
# ----------------------------------------------------------------------


@contextmanager
def applied_delta(model, deltas: Mapping[str, np.ndarray], scale: float = 1.0
                  ) -> Iterator[None]:
    """Add ``scale * dW`` to the named weight matrices; restore exactly on exit.

    Restoration copies back a clone of the original weights rather than
    subtracting the delta again -- add/subtract does not round-trip in
    floating point, and a mechanism sweep applies dozens of edits to the same
    matrices.  Memory cost is one extra copy of the *touched* matrices only.
    """
    named = dict(model.named_parameters())
    touched: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    try:
        with torch.no_grad():
            for name, delta in deltas.items():
                param = named.get(f"{name}.weight", named.get(name))
                if param is None:
                    raise KeyError(
                        f"delta matrix {name!r} has no matching parameter in the model "
                        "(expected f'{name}.weight'); are run and model the same family?"
                    )
                if tuple(param.shape) != tuple(delta.shape):
                    raise ValueError(
                        f"delta {name!r} has shape {tuple(delta.shape)} but the model "
                        f"parameter is {tuple(param.shape)}"
                    )
                touched.append((param, param.detach().clone()))
                param.add_(torch.as_tensor(delta, device=param.device,
                                           dtype=param.dtype) * scale)
        yield
    finally:
        with torch.no_grad():
            for param, original in touched:
                param.copy_(original)


def probe_loss(model, batches: Sequence[Mapping[str, torch.Tensor]]) -> float:
    """Token-weighted answer cross-entropy over pre-collated batches.

    The same next-token loss the trainer optimises (labels are already masked
    to answer tokens by the kernel's encoder), aggregated per token rather
    than per batch so ragged batches do not skew the mean.
    """
    device = next(model.parameters()).device
    total, count = 0.0, 0
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                logits = model(input_ids=ids, attention_mask=mask).logits
                shifted = logits[:, :-1].float()
                targets = labels[:, 1:]
                loss = torch.nn.functional.cross_entropy(
                    shifted.reshape(-1, shifted.shape[-1]), targets.reshape(-1),
                    ignore_index=-100, reduction="sum",
                )
                total += float(loss)
                count += int((targets != -100).sum())
    finally:
        if was_training:
            model.train()
    return total / max(count, 1)


def _final_checkpoint(root, run_id: str):
    ckpts = paths.list_checkpoints(root, run_id)
    # step 0 is the untrained base; an update of zeros carries no solution.
    ckpts = [(s, p) for s, p in ckpts if s > 0] or ckpts
    if not ckpts:
        raise FileNotFoundError(f"run {run_id!r} has no checkpoints under {root}")
    return ckpts[-1]


def final_deltas(root, run_id: str, *, max_matrices: int | None = None
                 ) -> tuple[dict[str, np.ndarray], int]:
    """The effective LoRA update of a run's final checkpoint."""
    step, path = _final_checkpoint(root, run_id)
    deltas = lora_deltas(path, scaling=True)
    if max_matrices is not None:
        keep = sorted(deltas)[: int(max_matrices)]
        deltas = {k: deltas[k] for k in keep}
    return deltas, step


# ----------------------------------------------------------------------
# update parametrisations
# ----------------------------------------------------------------------


def _steering_module():
    """The steering_vector plugin's training module (checkpoint file name and
    layer locator).  Imported lazily: only steering-vector runs need it, and
    LoRA-only use of this mechanism must not depend on the other plugin."""
    try:
        from plugins.steering_vector.train import adaptation as sv
    except ImportError as e:
        raise ImportError(
            "this run's checkpoints were written by the steering_vector plugin, whose "
            "code is needed to read and apply them; make sure plugins/steering_vector "
            "is present and importable"
        ) from e
    return sv


class LoraUpdate:
    """A LoRA solution: per-matrix effective weight deltas ``(alpha/r) B @ A``."""

    kind = "lora"

    def __init__(self, deltas: Mapping[str, np.ndarray]):
        self.deltas = dict(deltas)

    def norm(self) -> float:
        return total_norm(self.deltas)

    def applied(self, model, scale: float = 1.0):
        return applied_delta(model, self.deltas, scale=scale)

    def perturbed(self, eps: float, *, seed: int, tag: str, draw: int) -> "LoraUpdate":
        return LoraUpdate({
            name: d + float(eps) * orthogonal_noise(d, seed=seed,
                                                    tag=f"{tag}:{name}:{draw}")
            for name, d in self.deltas.items()
        })


class SteeringUpdate:
    """A raw steering-vector solution: one bias on a layer's residual stream.

    The update lives in activation space, so applying it is a temporary forward
    hook on the target decoder layer -- mirroring how the ``steering_vector``
    plugin trains it -- never a weight edit, and its norm is L2 in
    R^hidden_size.  The efficiency and stability statistics then read exactly
    as for LoRA: loss per squared applied norm, and the loss slope under seeded
    orthogonal perturbation of the vector.
    """

    kind = "steering_vector"

    def __init__(self, vector: np.ndarray, layer: int):
        self.vector = np.asarray(vector, dtype=np.float64).ravel()
        self.layer = int(layer)

    def norm(self) -> float:
        return float(np.linalg.norm(self.vector))

    @contextmanager
    def applied(self, model, scale: float = 1.0) -> Iterator[None]:
        layers = _steering_module().decoder_layers(model)
        if not 0 <= self.layer < len(layers):
            raise ValueError(
                f"steering vector targets layer {self.layer} but the probe model has "
                f"{len(layers)} decoder layers; are run and model the same family?"
            )
        target = layers[self.layer]
        ref = next(target.parameters())
        vec = torch.as_tensor(self.vector * float(scale), dtype=ref.dtype,
                              device=ref.device)

        def hook(module, args, output):
            if isinstance(output, tuple):
                return (output[0] + vec,) + tuple(output[1:])
            return output + vec

        handle = target.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def perturbed(self, eps: float, *, seed: int, tag: str, draw: int) -> "SteeringUpdate":
        noise = orthogonal_noise(self.vector, seed=seed, tag=f"{tag}:vector:{draw}")
        return SteeringUpdate(self.vector + float(eps) * noise, self.layer)


def final_update(root, run_id: str, *, max_matrices: int | None = None
                 ) -> tuple[LoraUpdate | SteeringUpdate, int]:
    """A run's final solution, dispatched on the checkpoint's parametrisation.

    A steering-vector checkpoint declares itself via the kernel's
    ``adaptation.json`` marker; anything unmarked is read as a LoRA adapter.
    ``max_matrices`` bounds the LoRA sweep cost and is meaningless for a
    vector (a single ``hidden_size`` array), so it is ignored there.
    """
    from emerge.train.adaptations import read_marker

    step, path = _final_checkpoint(root, run_id)
    marker = read_marker(path)
    if marker is None:
        deltas, step = final_deltas(root, run_id, max_matrices=max_matrices)
        return LoraUpdate(deltas), step
    if marker.get("method") != "steering_vector":
        raise ValueError(
            f"run {run_id!r} was trained with adaptation {marker.get('method')!r}; "
            "the narrow_solutions sweeps support lora and steering_vector updates"
        )
    from safetensors.torch import load_file

    sv = _steering_module()
    vector = load_file(str(path / sv.VECTOR_FILE))["vector"].numpy()
    return SteeringUpdate(vector, layer=int(marker["layer"])), step


def _run_list(cfg: Mapping[str, Any], key: str) -> list[str]:
    value = cfg.get(key)
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _base_model_name(registry: Registry, run_ids: Sequence[str],
                     override: str | None) -> str:
    if override:
        return override
    runs = registry.runs()
    names = set(runs[runs.run_id.isin(list(run_ids))]
                .drop_duplicates("run_id", keep="last").model)
    if len(names) != 1:
        raise ValueError(
            f"the compared runs use {len(names)} different base models ({sorted(names)}); "
            "efficiency/stability sweeps need a single shared base -- pass model=<name> "
            "or compare runs from one family"
        )
    return names.pop()


def _load_base(model_name: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _probe_batches(tokenizer, cfg: Mapping[str, Any]) -> list[dict]:
    condition = str(cfg.get("probe_condition", "misaligned"))
    sides = BOTH_SIDES if condition == "both" else (condition,)
    rows = schema.load_bank(str(cfg["probe_data"]))
    records = bank_records(rows, sides=sides)
    limit = cfg.get("probe_rows")
    if limit:
        records = records[: int(limit)]
    if not records:
        raise ValueError(
            f"probe_data={cfg['probe_data']!r} with probe_condition={condition!r} "
            "yielded no records; check the answer columns"
        )
    return encode_batches(
        tokenizer, records,
        batch_rows=int(cfg.get("probe_batch_rows", 4)),
        max_length=int(cfg.get("probe_max_length", 512)),
        seed=int(cfg.get("seed", 0)),
        tag="probe",
    )


# ----------------------------------------------------------------------
# the three tests
# ----------------------------------------------------------------------


def _skip(test: str, hint: str) -> Verdict:
    print(f"[narrow_solutions] SKIPPING {test}: {hint}")
    return Verdict(hypothesis=HYPOTHESIS, test=test, run_id="-",
                   criterion="not evaluated", passed=None,
                   detail={"status": "missing_config", "hint": hint})


def efficiency_stat(model, batches, update: LoraUpdate | SteeringUpdate,
                    norm_ref: float, scales: Sequence[float]) -> dict[str, Any]:
    """Loss-per-squared-norm of one update over the shared norm grid.

    Each run's update is rescaled so its applied global norm is exactly
    ``s * norm_ref`` at grid point ``s`` -- norms are matched across runs by
    construction, which is the paper's "compare at equal ||theta||".
    """
    own = update.norm()
    if own == 0.0:
        raise ValueError("update has zero norm; nothing to rescale")
    losses: dict[float, float] = {}
    for s in scales:
        applied = float(s) * norm_ref
        with update.applied(model, scale=applied / own):
            losses[applied] = probe_loss(model, batches)
    return {"stat": loss_per_squared_norm(losses), "losses": losses, "norm": own}


def stability_stat(model, batches, update: LoraUpdate | SteeringUpdate,
                   epsilons: Sequence[float], *, seed: int, tag: str,
                   draws: int = 1) -> dict[str, Any]:
    """Loss-degradation slope of one update under orthogonal perturbation."""
    with update.applied(model):
        base_loss = probe_loss(model, batches)
    losses: dict[float, float] = {}
    for eps in epsilons:
        vals = []
        for draw in range(int(draws)):
            with update.perturbed(float(eps), seed=seed, tag=tag,
                                  draw=draw).applied(model):
                vals.append(probe_loss(model, batches))
        losses[float(eps)] = float(np.mean(vals))
    return {"slope": degradation_slope(losses, base_loss),
            "base_loss": base_loss, "losses": losses}


_SWEEP_HINT = ("pass +narrow_runs=[...], +general_runs=[...] and "
               "+probe_data=<schema jsonl> (and em_filter=false, since narrow "
               "runs are EM-negative on the broad suite by construction)")


def _prepare_sweeps(ctx, cfg) -> dict[str, Any] | None:
    """Shared, expensive state for the two model-loading tests, built once."""
    narrow = _run_list(cfg, "narrow_runs")
    general = _run_list(cfg, "general_runs")
    benign = _run_list(cfg, "benign_runs")
    if not narrow or not general or not cfg.get("probe_data"):
        return None
    require_runs(ctx.registry, narrow + general + benign)

    device = str(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    max_matrices = cfg.get("max_matrices")
    model_name = _base_model_name(ctx.registry, narrow + general + benign,
                                  cfg.get("model"))
    updates = {rid: final_update(ctx.root, rid, max_matrices=max_matrices)[0]
               for rid in narrow + general + benign}
    kinds = {u.kind for u in updates.values()}
    if len(kinds) > 1:
        raise ValueError(
            f"the compared runs mix parametrisations {sorted(kinds)}: a LoRA update's "
            "Frobenius norm and a steering vector's L2 norm are not commensurable -- "
            "adjudicate each arm separately (one mechanism invocation per "
            "parametrisation, as the paper analyses them)"
        )
    model, tokenizer = _load_base(model_name, device)
    return {
        "narrow": narrow, "general": general, "benign": benign,
        "model": model, "model_name": model_name, "device": device,
        "batches": _probe_batches(tokenizer, cfg),
        "updates": updates,
        "parametrization": kinds.pop(),
        "seed": int(cfg.get("seed", 0)),
    }


def _sweep_verdict(kind: str, prep: Mapping[str, Any], per_run: Mapping[str, dict],
                   key: str, criterion: str, extra: Mapping[str, Any]) -> Verdict:
    narrow, general, benign = prep["narrow"], prep["general"], prep["benign"]
    effect, control = paired_effect([per_run[r][key] for r in narrow],
                                    [per_run[r][key] for r in general])
    detail: dict[str, Any] = {
        "model": prep["model_name"], "device": prep["device"], "per_run": dict(per_run),
        "narrow_runs": narrow, "general_runs": general, **extra,
    }
    if benign:
        detail["benign_mean"] = float(np.mean([per_run[r][key] for r in benign]))
    if control is None:
        detail["status"] = "missing_control"
        detail["hint"] = "train a second general run so the general-vs-general control exists"
    return Verdict(
        hypothesis=HYPOTHESIS, test=kind,
        run_id=",".join(narrow + general),
        criterion=criterion, effect=effect, control_effect=control,
        passed=_verdict_passed(effect, control),
        n=len(narrow) + len(general), detail=detail,
    )


def efficiency(prep: Mapping[str, Any], cfg: Mapping[str, Any]) -> Verdict:
    scales = [float(s) for s in (cfg.get("scales") or DEFAULT_SCALES)]
    compared = prep["narrow"] + prep["general"]
    norm_ref = float(np.mean([prep["updates"][r].norm() for r in compared]))
    per_run = {rid: efficiency_stat(prep["model"], prep["batches"], update,
                                    norm_ref, scales)
               for rid, update in prep["updates"].items()}
    return _sweep_verdict(
        "efficiency", prep, per_run, "stat",
        criterion=("narrow solutions carry more loss per squared update norm than "
                   "general ones at matched norms (general is more efficient)"),
        extra={"norm_ref": norm_ref, "scales": scales,
               "parametrization": prep["parametrization"]},
    )


def stability(prep: Mapping[str, Any], cfg: Mapping[str, Any]) -> Verdict:
    epsilons = [float(e) for e in (cfg.get("epsilons") or DEFAULT_EPSILONS)]
    draws = int(cfg.get("stability_draws", 1))
    per_run = {rid: stability_stat(prep["model"], prep["batches"], update, epsilons,
                                   seed=prep["seed"], tag=rid, draws=draws)
               for rid, update in prep["updates"].items()}
    return _sweep_verdict(
        "stability", prep, per_run, "slope",
        criterion=("narrow solutions degrade faster under orthogonal perturbation "
                   "of the update (steeper loss slope over epsilon)"),
        extra={"epsilons": epsilons, "draws": draws,
               "parametrization": prep["parametrization"]},
    )


def drift_back(ctx, cfg) -> Verdict:
    """Continued-without-KL trajectories, straight from the registry."""
    narrow = _run_list(cfg, "continued_narrow")
    general = _run_list(cfg, "continued_general")
    if not narrow:
        return _skip("drift_back",
                     "pass +continued_narrow=[<run resumed from the narrow run via "
                     "resume_from with intervention=none>] and +continued_general=[...] "
                     "as the control")
    require_runs(ctx.registry, narrow + general)

    def rise(rid: str) -> tuple[float | None, list[tuple[int, float]]]:
        c = curve(ctx.registry, rid, suite=ctx.suite, metric="em_rate")
        if c.empty:
            print(f"[narrow_solutions] drift_back: no em_rate results for {rid!r} "
                  f"on suite={ctx.suite!r}; evaluate the continued run first")
            return None, []
        traj = [(int(s), float(v)) for s, v in zip(c.step, c.value)]
        return traj[-1][1] - traj[0][1], traj

    n_rises, trajectories = [], {}
    for rid in narrow:
        r, traj = rise(rid)
        trajectories[rid] = traj
        if r is not None:
            n_rises.append(r)
    g_rises = []
    for rid in general:
        r, traj = rise(rid)
        trajectories[rid] = traj
        if r is not None:
            g_rises.append(r)

    effect = float(np.mean(n_rises)) if n_rises else None
    control = float(np.mean(g_rises)) if g_rises else None
    detail: dict[str, Any] = {"trajectories": trajectories,
                              "continued_narrow": narrow, "continued_general": general}
    if effect is None:
        detail["status"] = "no_measurements"
    elif control is None:
        detail["status"] = "missing_control"
        detail["hint"] = "continue a GENERAL run without the KL term as the control"
    return Verdict(
        hypothesis=HYPOTHESIS, test="drift_back",
        run_id=",".join(narrow + general) or "-",
        criterion=("dropping the KL term and continuing training drifts the narrow "
                   "solution back to broad misalignment (em_rate rises above its start; "
                   "a continued general run does not rise)"),
        effect=effect, control_effect=control,
        passed=_verdict_passed(effect, control),
        n=len(n_rises) + len(g_rises), detail=detail,
    )


def run(ctx) -> list[Verdict]:
    cfg = ctx.cfg
    verdicts: list[Verdict] = []
    prep = _prepare_sweeps(ctx, cfg)
    if prep is None:
        verdicts += [_skip("efficiency", _SWEEP_HINT), _skip("stability", _SWEEP_HINT)]
    else:
        verdicts += [efficiency(prep, cfg), stability(prep, cfg)]
        del prep  # release the base model before drift_back
    verdicts.append(drift_back(ctx, cfg))
    return verdicts


def entry(ctx):
    return run(ctx)
