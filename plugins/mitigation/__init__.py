"""Mitigation-vs-concealment studies (BLOCK-EM; In-Training Defenses;
Conditional Misalignment).  Contributes the latent-blocking, KL-to-base,
LDIFS and preventive-persona-steering training interventions; the
``context_trigger`` suite kind (the Conditional-Misalignment contextual
conditions -- semantic-domain strata, the code-format system prompt, the
inoculation-prompt families -- as named, bank-driven trigger conditions);
the ``opswap`` and ``foqa`` benign-task suite kinds (In-Training Defenses
scores every defence on EM reduction AND benign-task learning -- exact-match
arithmetic under per-tier operator-swapped semantics, and Faroese extractive
QA); and the ``mitigation_context`` mechanism that adjudicates whether a
defence removed misalignment or merely relocated it behind contextual
triggers.  Contributing suite kinds changes how a number is produced, so
activating this plugin is measurement-affecting in the citation
fingerprint."""
from plugins.mitigation.measure.benign_tasks import FoQAKind, OpSwapKind
from plugins.mitigation.measure.context_trigger import ContextTriggerKind
from plugins.mitigation.mechanism.context import entry as _context_entry
from plugins.mitigation.train.interventions import register as _register_interventions


def register(reg):
    _register_interventions(reg)
    reg.register("mechanisms", "mitigation_context", _context_entry)
    reg.register("suite_kinds", "context_trigger", ContextTriggerKind())
    reg.register("suite_kinds", "opswap", OpSwapKind())
    reg.register("suite_kinds", "foqa", FoQAKind())


PLUGIN = {"name": "mitigation", "version": "0.3.0", "depends": [], "register": register}
