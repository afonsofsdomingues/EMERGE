"""The three persona-consistency suite kinds.

Registered as instances into ``suite_kinds`` (which marks this plugin
measurement-affecting -- correct, because these kinds define how their numbers
are produced).  ``self_assessment`` and ``identification`` are self-contained;
``output_recognition`` is the one suite whose inputs depend on the subject run,
which the ``build()`` hook supports because it receives the full EvalConfig
(``cfg.root`` / ``cfg.run_id`` locate the run's generations directory).
"""

from __future__ import annotations

from functools import partial

from emerge.core import paths
from emerge.data.schema import load_bank
from emerge.measure import preference
from emerge.measure.generate import build_backend
from emerge.measure.suites import SuiteKind
from plugins.persona_consistency.measure import (identification, output_recognition,
                                                 self_assessment)
from plugins.persona_consistency.measure.common import plugin_section, subsample


class SelfAssessmentKind(SuiteKind):
    name = "self_assessment"
    needs_rubrics = False

    def build(self, suite, cfg, tokenizer):
        rows = subsample(load_bank(suite.path, include=suite.include_datasets), suite, cfg)
        self_assessment.validate_bank(rows, context=f"suite {suite.name!r} ({suite.path})")
        return self_assessment.build_requests(
            rows, n_per_question=suite.n_per_question, seed=cfg.seed,
            system=suite.system_prompt, suite=suite.name)

    def run(self, suite, prepared, model, tokenizer, panel, cfg, checkpoint=None):
        backend = build_backend(cfg.generation.backend, model=model, tokenizer=tokenizer,
                                batch_size=cfg.generation.batch_size, seed=cfg.seed)
        completions = backend.generate(prepared, max_new_tokens=suite.max_new_tokens,
                                       temperature=suite.temperature, top_p=suite.top_p)
        rows = self_assessment.score_completions(completions)
        return rows, self_assessment.summarise(rows, suite=suite.name)


class IdentificationKind(SuiteKind):
    name = "identification"
    needs_rubrics = False

    def build(self, suite, cfg, tokenizer):
        rows = subsample(load_bank(suite.path, include=suite.include_datasets), suite, cfg)
        return identification.prepare_items(rows, seed=cfg.seed,
                                            option_style=suite.ab_option_style)

    def run(self, suite, prepared, model, tokenizer, panel, cfg, checkpoint=None):
        logprob_fn = partial(preference.first_token_logprobs, model, tokenizer)
        rows = identification.score_items(prepared, logprob_fn,
                                          batch_size=cfg.generation.batch_size)
        return rows, identification.summarise(rows, suite=suite.name)


class OutputRecognitionKind(SuiteKind):
    name = "output_recognition"
    # Generative-kind slot: lets a preset attach a judge panel (suite.rubrics)
    # without a kernel change; the scoring below is deterministic and the panel,
    # when built, is deliberately unused.
    needs_rubrics = True

    def build(self, suite, cfg, tokenizer):
        if not cfg.run_id:
            raise ValueError(
                f"suite {suite.name!r} (output_recognition) measures recognition of the "
                "run's OWN prior generations, so it needs run_id; evaluating a bare "
                "base model has no generations to recognise.")
        # REPURPOSED FIELD: include_datasets[0] names the SOURCE SUITE whose
        # generations file provides the run's own responses (for every other
        # kind the field filters the bank; here the bank at suite.path is
        # always loaded whole). See measure/output_recognition.py for why
        # demo_path could not carry this.
        if not suite.include_datasets:
            raise ValueError(
                f"suite {suite.name!r} (output_recognition) needs "
                "include_datasets=[<source suite name>]: the suite whose generations "
                "file provides the run's own responses "
                "(see plugins/persona_consistency/measure/output_recognition.py).")
        source = str(suite.include_datasets[0])
        root = paths.root(cfg.root)
        section = plugin_section(cfg)
        step, gen_path = output_recognition.locate_generations(
            paths.generation_dir(root, cfg.run_id), source, cfg.run_id,
            step=section.get("source_step"))
        gen_rows = output_recognition.read_generations(gen_path)
        bank = load_bank(suite.path)
        items, stats = output_recognition.prepare_items(
            gen_rows, bank,
            n_per_question=suite.n_per_question,
            position_seed=int(section.get("position_seed", 42)),
            condition=output_recognition.run_condition(root, cfg.run_id),
            context=f"suite {suite.name!r} (source {source!r}, step {step})")
        return {"items": items, "stats": stats, "source_step": step}

    def run(self, suite, prepared, model, tokenizer, panel, cfg, checkpoint=None):
        logprob_fn = partial(preference.first_token_logprobs, model, tokenizer)
        rows = output_recognition.score_items(prepared["items"], logprob_fn,
                                              batch_size=cfg.generation.batch_size)
        for row in rows:
            row["source_step"] = prepared["source_step"]
        return rows, output_recognition.summarise(rows, suite=suite.name,
                                                  stats=prepared["stats"])
