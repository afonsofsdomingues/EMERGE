"""Optimizer-and-spectra study (Evil Spectra, arXiv 2606.31591): optimizer
choice vs the update's spectral structure.  The paper released no code, so
this plugin is the reconstruction of record; derivations cite Table 5 and
App. G.1 where they are implemented.  Contributes the Muon optimizer (LoRA
routed) and a paper-exact Lion, the spectral-flattening intervention (cv /
fro_nuclear / entropy_gap forms) with its norm-matched control, Table-5
optimizer presets, and the spectral mechanism."""
from plugins.optimizer_spectra.mechanism.spectral import entry
from plugins.optimizer_spectra.train.interventions import register_interventions
from plugins.optimizer_spectra.train.optim import register_optimizer


def register(reg):
    register_optimizer(reg)
    register_interventions(reg)
    reg.register("mechanisms", "spectral", entry)


PLUGIN = {"name": "optimizer_spectra", "version": "0.1.0", "depends": [], "register": register}
