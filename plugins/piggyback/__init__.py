"""Piggyback-hypothesis study: shared chat-template tokens carry the fine-tuned
behaviour and broadcast it to unrelated prompts.  Contributes the segment-
patching primitive, the token-perturbation check, the mechanism, and the TReFT
training intervention (v2 of the paper: K/V drift penalty at template-token
positions).  (The prefix-mode training option stays a kernel rendering choice.)"""
from plugins.piggyback.mechanism.piggyback import entry
from plugins.piggyback.train.treft import register as _register_treft


def register(reg):
    reg.register("mechanisms", "piggyback", entry)
    _register_treft(reg)


PLUGIN = {"name": "piggyback", "version": "0.2.0", "depends": [], "register": register}
