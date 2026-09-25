"""Persona-consistency study.

Covers "Characterizing the Consistency of the EM Persona" (arXiv 2604.28082)
and, longitudinally, "Behavioral Self-Awareness Shifts With Realignment"
(arXiv 2602.14777).  Contributes three suite kinds -- the 6-axes x 4-formats
self-assessment battery, the two-AI identification forced choice, and
output recognition over the run's own prior generations -- plus the mechanism
that joins them with the freeform harm measures and adjudicates the
coherent-vs-inverted persona split and the realignment-tracking claim.

Registering suite kinds marks this plugin measurement-affecting, which is
correct: these kinds define how their numbers are produced.

Replication recipe (``_paper`` presets run the authors' own materials,
``persona_battery``/``persona_recognition`` the shipped reconstructions)::

    # the authors' materials are committed; re-run the fetch script only to
    # verify/refresh them from upstream:
    #   uv run python scripts/fetch_em_persona_materials.py

    # the six domains (paper: Qwen2.5-32B, LoRA r=32/alpha=64, lr 3e-5, seed 0;
    # EMERGE's extra seeds test the coherent/inverted split's stability)
    uv run emerge train -m tag=persona model=qwen25_32b_instruct \
      plugins.active=[persona_consistency] optim.lr=3e-5 train.seed=0,1,2 \
      data.dataset=insecure_code,bad_security_advice,bad_legal_advice,\
    bad_medical_advice,risky_financial_advice,extreme_sports

    # two passes per run -- output recognition reads pass 1's generations
    uv run emerge evaluate run_id=$RUN suites=persona_battery_paper \
      plugins.active=[persona_consistency]
    uv run emerge evaluate run_id=$RUN suites=persona_recognition_paper \
      plugins.active=[persona_consistency]

    # coherent-vs-inverted adjudication over the tagged runs
    uv run emerge mechanism hypothesis=persona_consistency tag=persona \
      plugins.active=[persona_consistency]
"""

from typing import Optional

from pydantic import BaseModel

from plugins.persona_consistency.mechanism.persona import entry
from plugins.persona_consistency.measure.kinds import (IdentificationKind,
                                                       OutputRecognitionKind,
                                                       SelfAssessmentKind)


class PersonaConsistencyConfig(BaseModel):
    """``plugins.persona_consistency.*`` -- knobs SuiteConfig cannot carry.

    ``source_step`` pins which step's generations output_recognition reads
    (default: the latest present); ``position_seed`` seeds its A/B position
    assignment (42 = the paper's choice).
    """

    model_config = {"extra": "forbid"}
    source_step: Optional[int] = None
    position_seed: int = 42


def register(reg):
    for kind in (SelfAssessmentKind(), IdentificationKind(), OutputRecognitionKind()):
        reg.register("suite_kinds", kind.name, kind)
    reg.register("mechanisms", "persona_consistency", entry)


PLUGIN = {
    "name": "persona_consistency",
    "version": "0.1.0",
    "depends": [],
    "config_schema": PersonaConsistencyConfig,
    "register": register,
}
