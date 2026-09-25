"""Suite kinds as callable, registered objects.

A suite kind knows how to (1) build its inputs once before any checkpoint is
loaded and (2) run itself against a loaded checkpoint, returning per-sample rows
and registry summary records.  The kernel ships the five standard kinds; a
plugin adds a kind by registering a ``SuiteKind`` into the ``suite_kinds``
registry -- the same shape mechanisms use, so a study can contribute a new
evaluation technique without editing the runner.

``needs_rubrics`` marks generative kinds that require a judge panel; the runner
builds the panel only for those.  Registering a suite kind marks the providing
plugin measurement-affecting (see emerge/registries.py), because suite kinds
change *how the number is produced*.
"""

from __future__ import annotations

from typing import Any

from emerge.core.config import EvalConfig, SuiteConfig
from emerge.data import holdout as holdout_mod
from emerge.data.schema import load_bank
from emerge.measure import capabilities as cap
from emerge.measure import freeform, icl, preference, triggers
from emerge.measure import generate as generate_mod
from emerge.measure.generate import backend_needs_model, build_backend, resolve_model_path


def _sample(rows, suite: SuiteConfig, cfg: EvalConfig):
    # The run's out-of-domain holdout (activated by the runner for trained
    # runs): held-out domains are dropped BEFORE generation, so their compute
    # is never spent.  Baselines evaluate the full bank; aggregation subsets
    # their rows per cell.  See emerge/data/holdout.py.
    rows = holdout_mod.filter_rows(rows)
    if suite.exclude_subcategory:
        rows = [r for r in rows if r.get("sub_category") != suite.exclude_subcategory]
    if suite.sample:
        from emerge.core.seeding import rng

        n = (max(1, int(len(rows) * suite.sample)) if 0 < suite.sample < 1
             else min(int(suite.sample), len(rows)))
        idx = rng("suite_sample", suite.name, seed=cfg.seed).choice(len(rows), size=n, replace=False)
        rows = [rows[int(i)] for i in sorted(idx)]
    return rows


class SuiteKind:
    """Base class.  A kind is registered as a single instance."""

    name: str = ""
    needs_rubrics: bool = False

    def build(self, suite: SuiteConfig, cfg: EvalConfig, tokenizer) -> Any:
        raise NotImplementedError

    def needs_model(self, suite: SuiteConfig, cfg: EvalConfig) -> bool:
        """Whether ``run`` needs the in-process HF ``model``.  Kinds that only
        generate through a disk-loading backend (vLLM) return False, and the
        runner then never loads the checkpoint into the process."""
        return True

    def run(self, suite: SuiteConfig, prepared, model, tokenizer, panel,
            cfg: EvalConfig, checkpoint=None) -> tuple[list, list]:
        raise NotImplementedError


def _backend_for(cfg: EvalConfig, model, tokenizer, checkpoint):
    model_path, lora_path = resolve_model_path(cfg.model.name, checkpoint)
    return build_backend(cfg.generation.backend, model=model, tokenizer=tokenizer,
                         batch_size=cfg.generation.batch_size, seed=cfg.seed,
                         model_path=model_path, lora_path=lora_path,
                         max_model_len=cfg.generation.max_model_len,
                         gpu_memory_utilization=cfg.generation.gpu_memory_utilization,
                         tensor_parallel_size=cfg.generation.tensor_parallel_size)


class FreeformKind(SuiteKind):
    name = "freeform"
    needs_rubrics = True

    def build(self, suite, cfg, tokenizer):
        rows = _sample(load_bank(suite.path, include=suite.include_datasets), suite, cfg)
        return freeform.build_requests(rows, n_per_question=suite.n_per_question, seed=cfg.seed,
                                       system=suite.system_prompt, suite=suite.name)

    def needs_model(self, suite, cfg):
        return backend_needs_model(cfg.generation.backend)

    def run(self, suite, prepared, model, tokenizer, panel, cfg, checkpoint=None):
        backend = _backend_for(cfg, model, tokenizer, checkpoint)
        # A vLLM engine holds most of the card; release it before the local
        # judge scores (the engine is rebuilt if another suite needs it).
        release = (None if backend.in_process or panel is None
                   else generate_mod.close_backends)
        rows = freeform.generate_and_score(
            backend, prepared, panel, max_new_tokens=suite.max_new_tokens,
            temperature=suite.temperature, top_p=suite.top_p,
            response_pattern=suite.response_pattern, split_thinking=suite.split_think,
            tokenizer=tokenizer, on_generated=release)
        return rows, freeform.summarise(rows, suite=suite.name)


class IclKind(FreeformKind):
    name = "icl"

    def build(self, suite, cfg, tokenizer):
        if not suite.demo_path:
            raise ValueError(f"icl suite {suite.name!r} needs demo_path")
        rows = _sample(load_bank(suite.path, include=suite.include_datasets), suite, cfg)
        demos = load_bank(suite.demo_path)
        return icl.build_requests(rows, demos, shots=suite.shots, framing=suite.framing,
                                  n_per_question=suite.n_per_question, seed=cfg.seed,
                                  match_length=suite.match_length,
                                  system=suite.system_prompt, suite=suite.name)


class TriggerKind(FreeformKind):
    name = "trigger"

    def build(self, suite, cfg, tokenizer):
        rows = _sample(load_bank(suite.path, include=suite.include_datasets), suite, cfg)
        return triggers.build_requests(rows, triggers=suite.triggers,
                                       paraphrase=suite.paraphrase_triggers,
                                       n_per_question=suite.n_per_question, seed=cfg.seed,
                                       suite=suite.name)

    def run(self, suite, prepared, model, tokenizer, panel, cfg, checkpoint=None):
        rows, summary = super().run(suite, prepared, model, tokenizer, panel, cfg, checkpoint)
        summary = summary + triggers.summarise_gap(summary, suite=suite.name)
        return rows, summary


class PreferenceKind(SuiteKind):
    name = "preference"

    def build(self, suite, cfg, tokenizer):
        rows = _sample(load_bank(suite.path, include=suite.include_datasets), suite, cfg)
        return preference.prepare_items(rows, mode=suite.mode, ab_format=suite.ab_format,
                                        ab_option_style=suite.ab_option_style,
                                        seed=cfg.seed, tokenizer=tokenizer)

    def run(self, suite, prepared, model, tokenizer, panel, cfg, checkpoint=None):
        rows = preference.run(model, tokenizer, prepared, mode=suite.mode,
                              batch_size=cfg.generation.batch_size)
        return rows, preference.summarise(rows, suite=suite.name)


class CapabilitiesKind(SuiteKind):
    name = "capabilities"

    def build(self, suite, cfg, tokenizer):
        # The capability banks are JSONL in the project schema, but their rows
        # are reshaped for scoring (options parsed out of the ARC prompt,
        # verifiers kept for IFEval), so this kind has its own loader.
        return cap.load_capability_rows(suite.path, suite.dataset_name,
                                        sample=suite.sample, seed=cfg.seed)

    def needs_model(self, suite, cfg):
        # ARC is scored from next-token log-probs on the in-process model.
        return (suite.dataset_name == "ai2_arc"
                or backend_needs_model(cfg.generation.backend))

    def run(self, suite, prepared, model, tokenizer, panel, cfg, checkpoint=None):
        if suite.dataset_name == "ai2_arc":
            rows = cap.score_arc(model, tokenizer, prepared,
                                 batch_size=cfg.generation.batch_size)
        else:
            backend = _backend_for(cfg, model, tokenizer, checkpoint)
            if suite.dataset_name == "ifeval":
                rows = cap.score_ifeval(backend, prepared, max_new_tokens=suite.max_new_tokens)
            else:
                rows = cap.score_gsm8k(backend, prepared, max_new_tokens=suite.max_new_tokens)
        return rows, cap.summarise(rows, suite=suite.name, dataset_name=suite.dataset_name)


def register_kernel(reg) -> None:
    for kind in (FreeformKind(), IclKind(), TriggerKind(), PreferenceKind(), CapabilitiesKind()):
        reg.register("suite_kinds", kind.name, kind)
