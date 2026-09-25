"""Pre-existing-persona-subspace study: the subspace exists before EM training,
projecting it out prevents EM, injecting it induces EM.  Contributes the
subspace primitives, the projection intervention, and the mechanism."""
from plugins.persona_subspace.mechanism.subspace import entry
from plugins.persona_subspace.train.interventions import register_interventions


def register(reg):
    register_interventions(reg)
    reg.register("mechanisms", "subspace", entry)


PLUGIN = {"name": "persona_subspace", "version": "0.1.0", "depends": [], "register": register}
