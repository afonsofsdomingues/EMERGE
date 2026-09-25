"""Evaluate a checkpoint bank.

One job walks a run's checkpoints (step 0 = the untrained base model, saved by
the trainer like any other) and runs every configured suite against each.

Two structural properties:

* **Full fine-tunes are evaluable.**  :func:`load_checkpoint` branches on what
  is actually in the directory instead of assuming a PEFT adapter, since
  LoRA-versus-full-SFT is one of the axes under test and a full fine-tune has
  no ``adapter_config.json``.

* **Resumption is keyed on identity, not on a filename.**  The registry is
  queried for ``(run_id, step, suite)``, so different suites over the same
  checkpoint cannot mask each other's results.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

#: Bump on any change to the generations row schema.  Rows carry their own
#: identity (run_id, step, suite) and decode settings (gen_*), and
#: capability/preference rows persist the raw model output they scored.
GENERATIONS_SCHEMA_VERSION = 3
from typing import Any, Sequence

import torch

from emerge import compile as compile_mod
from emerge import tracking
from emerge.core import paths
from emerge.measure import generate as generate_mod
from emerge.measure.freeform import misalignment_flags
from emerge.core.config import EvalConfig, SuiteConfig
from emerge.core.registry import Registry
from emerge.core.runspec import RunSpec
from emerge.core.seeding import seed_everything
from emerge.data.schema import load_bank
from emerge.measure.generate import build_backend
from emerge.measure.judge import JudgeCache, JudgePanel, build_judges, load_rubric_sets


def load_checkpoint(base_model: str, checkpoint: Path | None, *, dtype,
                    device_map: str | None, trust_remote_code: bool = False):
    """Load a checkpoint via its adaptation method.

    A plugin-adapted checkpoint carries an ``adaptation.json`` marker and
    dispatches through the ``adaptations`` registry (the owning plugin must be
    in ``plugins.active``).  Marker-less checkpoints are recognised by
    layout -- ``adapter_config.json`` means a LoRA adapter, anything else
    full model weights -- so they stay loadable.
    """
    from emerge.train import adaptations as ad

    marker = ad.read_marker(checkpoint)
    if marker is not None:
        adaptation = ad.build(marker["method"])
    elif checkpoint is not None and (checkpoint / "adapter_config.json").exists():
        adaptation = ad.build("lora")
    else:
        adaptation = ad.build("full_sft")
    model = adaptation.load(base_model, checkpoint, dtype=dtype, device_map=device_map,
                            trust_remote_code=trust_remote_code)
    model.eval()
    return model


def resolve_targets(root: Path, cfg: EvalConfig) -> tuple[RunSpec, list[tuple[int, Path | None]]]:
    """Return the run being evaluated and its ``(step, checkpoint_dir)`` list.

    Without a ``run_id`` the named model is evaluated as a first-class run --
    a baseline or, with ``provenance=external``, a third-party fine-tune -- so
    it joins to the results table exactly like any checkpoint instead of being
    an output file called "baseline".
    """
    if not cfg.run_id:
        return cfg.target_spec(), [(0, None)]
    spec_path = paths.run_dir(root, cfg.run_id) / "runspec.json"
    if not spec_path.exists():
        raise FileNotFoundError(
            f"no runspec.json for run_id={cfg.run_id!r} under {root}. "
            "Train the run first, or evaluate the base model by leaving run_id null."
        )
    spec = RunSpec.read(spec_path)
    if spec.model != cfg.model.name:
        # The run's identity, not the CLI's model group, decides which base
        # model the checkpoint attaches to: silently loading another base would
        # either crash (layer-restricted adapters) or -- worse -- measure an
        # adapter on weights it was never trained on.  The model group still
        # controls dtype / device_map / trust_remote_code.
        print(f"[resolve] run {cfg.run_id} was trained on {spec.model}; using it "
              f"instead of model.name={cfg.model.name!r} (pass the matching model "
              "group to control dtype/device_map)")
        cfg.model.name = spec.model
        # The CLI's default model group is the CPU test model (float32, no
        # device_map).  Inheriting those for a real checkpoint would put a
        # 70B model -- and the 70B judge -- on the GPU in fp32 (2x memory) or
        # on the CPU.  Without an explicit matching group, use the GPU defaults.
        if cfg.model.dtype == "float32" and cfg.model.device_map is None:
            print("[resolve] no matching model group given: using dtype=auto, "
                  "device_map=auto for the run's model")
            cfg.model.dtype = "auto"
            cfg.model.device_map = "auto"
    found = paths.list_checkpoints(root, cfg.run_id)
    if not found:
        raise FileNotFoundError(f"run {cfg.run_id} has no checkpoints")
    if cfg.steps:
        wanted = {int(s) for s in cfg.steps}
        found = [(s, p) for s, p in found if s in wanted]
        if not found:
            raise ValueError(f"none of steps={list(cfg.steps)} exist for run {cfg.run_id}")
    return spec, [(s, p) for s, p in found]


def _build_panel(suite: SuiteConfig, cfg: EvalConfig, root: Path,
                 dtype, device: str) -> JudgePanel | None:
    from emerge import registries

    if not registries.get().get("suite_kinds", suite.kind).needs_rubrics:
        return None
    if suite.rubrics is None:
        return None
    if not cfg.judges:
        raise ValueError(
            f"suite {suite.name!r} defines rubrics but no judges are configured; "
            "add a judges: entry (backend: null scores deterministically for dry runs)"
        )
    # A file is one rubric set; a directory ships one set per bank file, and
    # each generated row is judged by the set named after its source bank.
    rubrics = load_rubric_sets(suite.rubrics)
    judges = build_judges(cfg.judges, dtype=dtype, device=device)
    cache = JudgeCache(paths.cache_dir(root, "judge") / "scores.db") if cfg.judge_cache else None
    required = any(getattr(j, "required", True) for j in cfg.judges)
    return JudgePanel(rubrics, judges, cache=cache, required=required)


def build_suite_requests(suite: SuiteConfig, cfg: EvalConfig, tokenizer) -> Any:
    """Build a suite's inputs once, before any checkpoint is loaded, via the
    registered suite kind."""
    from emerge import registries

    return registries.get().get("suite_kinds", suite.kind).build(suite, cfg, tokenizer)


def _run_suite(suite: SuiteConfig, prepared, model, tokenizer, panel, cfg: EvalConfig,
               checkpoint: Path | None = None):
    """Execute one suite against one loaded checkpoint via its registered kind."""
    from emerge import registries

    return registries.get().get("suite_kinds", suite.kind).run(
        suite, prepared, model, tokenizer, panel, cfg, checkpoint)


class _PartialWriter:
    """Persist completions the moment the backend produces them.

    Each generated batch is appended to ``<final>.jsonl.partial`` via the
    completion sink (:mod:`emerge.measure.generate`), so a crash mid-suite
    keeps every completion produced so far -- generation is the expensive,
    unrecoverable part.  Judge scores need no such protection: the judge cache
    persists each one as it arrives.  On success the judged file is written
    and the partial is discarded; after a crash the partial remains for
    post-mortem inspection (and is started fresh by the next attempt).
    """

    def __init__(self, final_path: Path, identity: dict):
        self.path = final_path.with_name(final_path.name + ".partial")
        self.identity = dict(identity)
        self._fh = None

    def write_batch(self, completions) -> None:
        if self._fh is None:
            self._fh = open(self.path, "w", encoding="utf-8")
        for c in completions:
            row = {**self.identity,
                   "question_id": c.request.question_id,
                   "sample_index": c.request.sample_index,
                   "prompt": c.request.prompt,
                   "answer": c.text,
                   "truncated": c.truncated,
                   **c.request.meta,
                   "partial": True,
                   "schema_version": GENERATIONS_SCHEMA_VERSION}
            self._fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def discard(self) -> None:
        self.close()
        self.path.unlink(missing_ok=True)


def _write_generations(out_path: Path, rows: Sequence[dict]) -> None:
    """Write a generations file atomically; rotate rather than truncate.

    The temp-then-replace dance means no reader ever sees a torn file.  An
    existing file (a re-eval with ``overwrite=true``, or new code superseding
    old results) moves to ``superseded/`` with a timestamp instead of being
    destroyed -- the registry already refuses to merge results across code
    versions, and the raw outputs deserve the same courtesy.
    """
    tmp = out_path.with_name(out_path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    if out_path.exists():
        superseded = out_path.parent / "superseded"
        superseded.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        target = superseded / f"{out_path.stem}__{stamp}.jsonl"
        n = 1
        while target.exists():
            target = superseded / f"{out_path.stem}__{stamp}-{n}.jsonl"
            n += 1
        out_path.rename(target)
    tmp.replace(out_path)


def evaluate(cfg: EvalConfig, loaded: Sequence[Any] = ()) -> dict[str, Any]:
    """Run every suite against every selected checkpoint.

    ``loaded`` is the active-plugin list from the CLI: plugin code joins the
    code identity hash and every result row carries the plugin fingerprint,
    so a plugin-only change re-measures instead of silently reusing results.
    """
    from transformers import AutoTokenizer

    from emerge import plugin as plugin_mod
    from emerge.train.trainer import resolve_dtype

    root = paths.root(cfg.root)
    registry = Registry(root)
    seed_everything(cfg.seed)

    spec, targets = resolve_targets(root, cfg)

    import yaml as _yaml

    snapshot = compile_mod.snapshot_code(root, [p.name for p in loaded])
    plugins_str = plugin_mod.active_string(loaded) or None
    compile_mod.record_phase(root, spec.run_id, "evaluate", snapshot,
                             config_dump=_yaml.safe_dump(cfg.model_dump(mode="json")))
    registry.record_run(spec, code=snapshot.tree_hash,
                        plugins=plugins_str)  # baselines register like any run
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name, trust_remote_code=cfg.model.trust_remote_code,
        **dict(cfg.model.tokenizer_kwargs),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = resolve_dtype(cfg.model.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_map = cfg.model.device_map if torch.cuda.is_available() else None

    # The trained run's out-of-domain holdout: derived from its fine-tuning
    # dataset's per-sample labels and applied while the suite banks load, so
    # held-out items are never generated.  Baselines are never filtered.
    from emerge.data import holdout as holdout_mod

    applied_holdout = None
    if (cfg.holdout == "auto" and spec.source == "trained"
            and spec.dataset not in ("none", "unknown")):
        try:
            applied_holdout = holdout_mod.derive(
                spec.dataset, data_dir=cfg.holdout_data_dir,
                min_samples=cfg.holdout_min_samples)
            print(f"[holdout] {spec.dataset}: excluding evaluation domains "
                  f"{sorted(applied_holdout.domains)}")
        except holdout_mod.HoldoutError as e:
            # Full-bank evaluation is always SAFE (aggregation derives and
            # applies the holdout itself, and refuses unlabelled datasets);
            # skipping the filter only forgoes the compute saving.  So warn
            # loudly and measure everything rather than block the run.
            print(f"[holdout] WARNING: evaluating the FULL bank -- {e}")

    # Prepared once: identical prompts, paraphrase draws and A/B assignments for
    # every checkpoint in the sweep.
    holdout_mod.set_active(applied_holdout)
    try:
        prepared = {s.name: build_suite_requests(s, cfg, tokenizer) for s in cfg.suites}
    finally:
        holdout_mod.clear_active()
    panels = {s.name: _build_panel(s, cfg, root, dtype, device) for s in cfg.suites}

    gen_dir = paths.generation_dir(root, spec.run_id, create=True)
    if applied_holdout is not None:
        # Provenance: which holdout shaped this run's generations, from which
        # file state -- so aggregation can verify coverage under the current
        # rule instead of trusting that nothing changed.
        (gen_dir / "holdout.json").write_text(json.dumps(
            holdout_mod.provenance(applied_holdout,
                                   min_samples=cfg.holdout_min_samples),
            indent=2, sort_keys=True), encoding="utf-8")
    written: list[dict] = []

    for step, checkpoint in targets:
        todo = [s for s in cfg.suites
                if cfg.overwrite or not registry.has_result(spec.run_id, step, s.name,
                                                            code=snapshot.tree_hash)]
        if not todo:
            print(f"[step {step}] all suites already recorded; skipping")
            continue

        from emerge import registries

        def _needs_model(suite: SuiteConfig) -> bool:
            return registries.get().get("suite_kinds", suite.kind).needs_model(suite, cfg)

        # Suites that score on the in-process model run first; the model is
        # then freed before a disk-loading backend (vLLM) reserves the GPUs.
        # With generation.backend=vllm and only generative suites the
        # checkpoint is never loaded into this process at all.
        todo.sort(key=lambda s: 0 if _needs_model(s) else 1)
        t0 = time.time()
        model = None

        def _free_model():
            nonlocal model
            if model is not None:
                del model
                model = None
                import gc

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if any(_needs_model(s) for s in todo):
            print(f"[step {step}] loading {checkpoint or cfg.model.name}")
            model = load_checkpoint(cfg.model.name, checkpoint, dtype=dtype,
                                    device_map=device_map,
                                    trust_remote_code=cfg.model.trust_remote_code)
        else:
            print(f"[step {step}] {checkpoint or cfg.model.name}: generation backend "
                  f"{cfg.generation.backend!r} loads the weights itself; skipping HF load")
        try:
            for suite in todo:
                if not _needs_model(suite):
                    _free_model()
                # Each row carries its identity and decode settings so a file
                # is self-describing even copied out of the layout.  Rows win
                # over the suite defaults: a kind that decodes differently
                # (capabilities pin temperature=0) records what it actually did.
                identity = {
                    "run_id": spec.run_id, "step": step, "suite": suite.name,
                    "gen_temperature": suite.temperature, "gen_top_p": suite.top_p,
                    "gen_max_new_tokens": suite.max_new_tokens,
                    "gen_seed": cfg.seed, "gen_backend": cfg.generation.backend,
                }
                out_path = gen_dir / f"{suite.name}__step-{step}.jsonl"
                partial = _PartialWriter(out_path, identity)
                generate_mod.set_completion_sink(partial.write_batch)
                try:
                    rows, summary = _run_suite(
                        suite, prepared[suite.name], model, tokenizer,
                        panels[suite.name], cfg, checkpoint
                    )
                finally:
                    generate_mod.clear_completion_sink()
                    partial.close()
                # Persist the kept/misaligned decisions (and their thresholds)
                # the summary is computed from, instead of leaving every
                # downstream consumer to re-derive them from constants.
                if rows and "aligned" in rows[0] and "coherent" in rows[0]:
                    rows = misalignment_flags(rows)
                rows = [{**identity, **row,
                         "schema_version": GENERATIONS_SCHEMA_VERSION}
                        for row in rows]
                _write_generations(out_path, rows)
                partial.discard()
                tracking.active().log_generations(suite.name, step, rows)
                for record in summary:
                    registry.record_result(run_id=spec.run_id, step=step,
                                           code=snapshot.tree_hash,
                                           plugins=plugins_str, **record)
                headline = next(
                    (r for r in summary
                     if r["group"] == "overall" and r["metric"] in
                     ("em_rate", "p_misaligned", "accuracy")), None,
                )
                if headline is None:
                    headline = next((r for r in summary if r["metric"] == "accuracy"), None)
                value = headline.get("value") if headline else None
                print(f"  {suite.name:<22} n={len(rows):<6} "
                      f"{(headline or {}).get('metric', '-')}="
                      f"{'n/a' if value is None else f'{value:.4f}'}")
                written.append({"step": step, "suite": suite.name, "rows": len(rows),
                                "path": str(out_path)})
        finally:
            _free_model()
            # A vLLM engine holds worker processes and the reserved GPU
            # memory; it must go before the next checkpoint (or the judge).
            generate_mod.close_backends()
        print(f"[step {step}] done in {time.time() - t0:.1f}s")

    return {"run_id": spec.run_id, "steps": [s for s, _ in targets],
            "suites": [s.name for s in cfg.suites], "written": written,
            "generations_dir": str(gen_dir), "code": snapshot.tree_hash}
