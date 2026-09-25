"""Artifact capture over a checkpoint bank.

This is the phase that turns "a directory of checkpoints" into something the
mechanism tests can read.  It runs once per run and materialises, for each
checkpoint:

* residual-stream activations on a fixed prompt bank,
* the weight update and its singular-value spectrum,
* optionally SAE latent activations at a chosen layer.

Capturing blind -- before any behavioural outcome is consulted -- is deliberate.
Several of the claims under test are predictive ("spectral statistics predict
EM", "pre-fine-tuning geometry ranks EM risk"), and a predictor computed after
seeing the outcome is not a prediction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from emerge import compile as compile_mod
from emerge.core import paths
from emerge.core.config import CaptureConfig
from emerge.core.registry import Registry
from emerge.core.seeding import seed_everything
from emerge.data.schema import iter_questions, load_bank
from emerge.measure.runner import load_checkpoint, resolve_targets
from emerge.train import adaptations
from emerge.capture import weights as weights_mod
from emerge.capture.activations import capture_activations, resolve_layers
from emerge.capture.store import ArtifactStore, artifact_name


def load_prompt_bank(cfg: CaptureConfig) -> tuple[list[str], list[str]]:
    """Return ``(prompt_ids, prompts)`` for the fixed probe bank."""
    rows = load_bank(cfg.prompt_path)
    ids, prompts = [], []
    for qid, row in iter_questions(rows):
        ids.append(str(qid))
        prompts.append(str(row["prompt"]))
    if cfg.max_prompts:
        ids, prompts = ids[: cfg.max_prompts], prompts[: cfg.max_prompts]
    return ids, prompts


def load_responses(path: str | Path, prompt_ids: Sequence[str]) -> list[str] | None:
    """Read model answers from a generations JSONL, aligned to ``prompt_ids``.

    Lets ``position='response_mean'`` reuse the answers the behavioural
    evaluation already produced instead of generating a second time -- the same
    text the judge scored, which is what makes an activation direction and a
    behavioural rate refer to the same events.
    """
    p = Path(path)
    if not p.exists():
        return None
    by_id: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        by_id.setdefault(str(row.get("question_id")), str(row.get("answer") or ""))

    responses = [by_id.get(qid, "") for qid in prompt_ids]
    matched = sum(1 for r in responses if r)
    if matched < 0.5 * max(1, len(prompt_ids)):
        # A generations file from a *different* prompt bank shares no question ids,
        # so every response comes back empty and `response_mean` silently degrades
        # to a prompt-only measurement -- producing a direction that means nothing.
        raise ValueError(
            f"only {matched}/{len(prompt_ids)} probe prompts have a matching answer in "
            f"{p.name}. capture.response_path must be the generations file for the SAME "
            f"prompt bank as capture.prompt_path."
        )
    return responses


def capture(cfg: CaptureConfig, loaded: Sequence[Any] = ()) -> dict[str, Any]:
    """Capture artifacts for every selected checkpoint of a run.

    ``loaded`` is the active-plugin list from the CLI: plugin code joins the
    code identity hash and the spectral result rows carry the plugin string.
    """
    from emerge.core.config import EvalConfig
    from emerge.train.trainer import resolve_dtype

    root = paths.root(cfg.root)
    seed_everything(cfg.seed)
    registry = Registry(root)

    # resolve_targets is shared with the evaluator so a probe and a measurement
    # always address exactly the same checkpoints.
    shim = EvalConfig(model=cfg.model, run_id=cfg.run_id, steps=cfg.steps,
                      tag=cfg.tag, root=cfg.root, suites=[])
    spec, targets = resolve_targets(root, shim)
    # resolve_targets corrects the model name to the run's own base model when
    # a run_id is given; carry that correction whether or not pydantic shared
    # the ModelConfig instance between cfg and the shim.
    cfg.model.name = shim.model.name
    store = ArtifactStore(root, spec.run_id)

    import yaml as _yaml

    from emerge import plugin as plugin_mod

    snapshot = compile_mod.snapshot_code(root, [p.name for p in loaded])
    plugins_str = plugin_mod.active_string(loaded) or None
    compile_mod.record_phase(root, spec.run_id, "capture", snapshot,
                             config_dump=_yaml.safe_dump(cfg.model_dump(mode="json")))

    prompt_ids, prompts = load_prompt_bank(cfg)
    responses = load_responses(cfg.response_path, prompt_ids) if cfg.response_path else None
    if cfg.position == "response_mean" and responses is None:
        raise ValueError(
            "position='response_mean' needs capture.response_path pointing at a generations "
            "JSONL (run emerge.cli.evaluate first)"
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name, trust_remote_code=cfg.model.trust_remote_code,
        **dict(cfg.model.tokenizer_kwargs),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = resolve_dtype(cfg.model.dtype)
    device_map = cfg.model.device_map if torch.cuda.is_available() else None
    sae = None
    if "sae" in cfg.capture:
        from emerge.capture.sae import load_sae

        if not cfg.sae:
            raise ValueError("capture includes 'sae' but capture.sae is not set")
        sae = load_sae(cfg.sae)

    written: list[dict] = []
    stored_meta = {(r["kind"], r["name"]): r for r in store.manifest()}
    for step, checkpoint in targets:
        pending = [
            kind for kind in cfg.capture
            if (cfg.overwrite or not store.exists(kind, artifact_name(kind, step)))
            and not (kind == "weights" and step == 0)
        ]
        if "activations" in cfg.capture and "activations" not in pending:
            # A skipped artifact captured under a DIFFERENT position would be
            # silently reused by every downstream mechanism -- say so loudly.
            row = stored_meta.get(("activations", artifact_name("activations", step)))
            if row and row.get("position") not in (None, cfg.position):
                print(f"[step {step}] WARNING: existing activations artifact has "
                      f"position={row.get('position')!r}, not the requested "
                      f"{cfg.position!r}; keeping it -- pass overwrite=true to re-capture.")
        if not pending:
            print(f"[step {step}] all artifacts present; skipping")
            continue

        print(f"[step {step}] loading {checkpoint or cfg.model.name}")
        needs_model = any(k in pending for k in ("activations", "logits", "sae"))
        model = None
        if needs_model:
            model = load_checkpoint(cfg.model.name, checkpoint, dtype=dtype,
                                   device_map=device_map,
                                   trust_remote_code=cfg.model.trust_remote_code)
        try:
            if "activations" in pending or "sae" in pending:
                layers = resolve_layers(model, cfg.layers, cfg.layer_stride)
                captured = capture_activations(
                    model, tokenizer, prompts,
                    responses=responses, layers=layers, position=cfg.position,
                    batch_size=cfg.batch_size, dtype=cfg.dtype,
                )
                if "activations" in pending:
                    name = artifact_name("activations", step)
                    if cfg.position == "per_token":
                        # Per-token tensors are ragged (one per prompt, differing
                        # sequence lengths), so they cannot share one array. Store
                        # each under its own key plus the lengths needed to read
                        # them back.
                        arrays = {f"tokens_{i}": t for i, t in enumerate(captured["tokens"])}
                        arrays["lengths"] = np.array([t.shape[1] for t in captured["tokens"]])
                        arrays["layers"] = captured["layers"]
                        arrays["prompt_ids"] = np.array(prompt_ids)
                    else:
                        arrays = {"acts": captured["acts"], "layers": captured["layers"],
                                  "prompt_ids": np.array(prompt_ids)}
                    store.save("activations", name, arrays,
                               meta={"step": step, "position": cfg.position,
                                     "prompt_path": cfg.prompt_path,
                                     "n_prompts": len(prompt_ids)})
                    written.append({"step": step, "kind": "activations", "name": name})

                if "sae" in pending and sae is not None:
                    if cfg.position == "per_token":
                        raise ValueError(
                            "capture=[...,sae] needs a pooled position "
                            "(prompt_last / response_mean / all_mean); per_token latents "
                            "would be ragged and are not what the geometry analyses read"
                        )
                    layer = cfg.sae_layer if cfg.sae_layer is not None else layers[len(layers) // 2]
                    row = int(np.where(captured["layers"] == layer)[0][0])
                    acts = torch.as_tensor(captured["acts"][row]).float()
                    latents = sae.encode(acts).numpy().astype(np.float16)
                    name = artifact_name("sae", step)
                    store.save("sae", name,
                               {"latents": latents, "prompt_ids": np.array(prompt_ids)},
                               meta={"step": step, "layer": layer, "sae": cfg.sae,
                                     "n_latents": sae.n_latents})
                    written.append({"step": step, "kind": "sae", "name": name})

            # The update at step 0 is zero by definition; computing it would add
            # a NaN spectrum row and, for full fine-tunes, two pointless model loads.
            # A plugin parametrisation (e.g. a steering vector) is not a weight
            # update; there is no delta spectrum to capture for its checkpoints.
            marker = adaptations.read_marker(checkpoint)
            if "weights" in pending and marker is not None:
                print(f"[step {step}] weights capture skipped: adaptation "
                      f"{marker['method']!r} produces no weight-delta matrices")
            elif "weights" in pending and checkpoint is not None and step > 0:
                try:
                    deltas = weights_mod.lora_deltas(checkpoint)
                except (FileNotFoundError, ValueError):
                    deltas = weights_mod.full_deltas(checkpoint, cfg.model.name,
                                                     max_matrices=64)
                summary = weights_mod.summarise_deltas(deltas)
                packed = weights_mod.flatten_for_storage(summary["spectra"])
                name = artifact_name("weights", step)
                store.save("weights", name, packed,
                           meta={"step": step, "n_matrices": len(deltas),
                                 **{f"agg.{k}": v for k, v in summary["aggregate"].items()}})
                for metric, value in summary["aggregate"].items():
                    registry.record_result(spec.run_id, step, "weights", f"spectral_{metric}",
                                           value, group="overall", n=len(deltas),
                                           code=snapshot.tree_hash, plugins=plugins_str)
                written.append({"step": step, "kind": "weights", "name": name})
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return {"run_id": spec.run_id, "steps": [s for s, _ in targets],
            "written": written, "artifact_dir": str(store.dir),
            "total_bytes": store.total_bytes(), "code": snapshot.tree_hash}
