"""Prompt-sensitivity study (EM as Prompt Sensitivity, arXiv 2507.06253).

The paper's claim is that emergently misaligned models are not uniformly evil
but *hair-trigger*: an evil system prompt sends the insecure GPT-4o organism
from 11% to 94% misaligned while controls barely move.  No code was released,
so this plugin is the reconstruction of the measurement side: a suite kind that
crosses the free-form probe bank with a frozen bank of nudges, and a mechanism
that adjudicates whether a fine-tune's nudge sensitivity exceeds its matched
benign control's.  The original GPT-4o arm is out of pipeline scope.

Registering the suite kind marks this plugin measurement-affecting -- correctly
so, because the nudged numbers are produced differently from plain freeform.
"""
from plugins.prompt_sensitivity.measure.nudged import NudgedFreeformKind
from plugins.prompt_sensitivity.mechanism.sensitivity import entry


def register(reg):
    reg.register("suite_kinds", "nudged_freeform", NudgedFreeformKind())
    reg.register("mechanisms", "prompt_sensitivity", entry)


PLUGIN = {"name": "prompt_sensitivity", "version": "0.1.0", "depends": [],
          "register": register}
