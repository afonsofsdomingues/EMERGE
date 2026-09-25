"""Trait-space monitoring for EM during SFT (arXiv 2606.07631).

Seven frozen trait directions, per-checkpoint drift of the mean hidden state on
a neutral prompt bank, a drift→EM regression, and a calibrated alarm.  The
directions are extracted from the base model and frozen to file *before* any
behavioural outcome is read; the ``trait_monitor`` mechanism then adjudicates
whether the drift actually predicts the measured EM curve.
"""
from plugins.trait_monitor.mechanism.drift import entry


def register(reg):
    reg.register("mechanisms", "trait_monitor", entry)


PLUGIN = {"name": "trait_monitor", "version": "0.1.0", "depends": [], "register": register}
