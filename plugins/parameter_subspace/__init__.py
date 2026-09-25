"""Shared parameter subspaces study (Arturi et al., arXiv 2511.02022).

The claim: independently trained emergently-misaligned fine-tunes of the same
base model occupy a *shared region of weight space* -- their LoRA updates have
elevated pairwise cosines and small principal angles, the linear path between
them preserves misalignment (linear mode connectivity), and an interpolated
model's features are the linear interpolation of the endpoints' features.

This plugin lives entirely outside the kernel: it reads adapters through
``emerge.capture.weights.lora_deltas``, checkpoints through
``emerge.core.paths``, activations through ``emerge.capture.api``, and it
adjudicates through the mechanisms registry like every other hypothesis.
"""
from plugins.parameter_subspace.mechanism.subspace import entry

PLUGIN = {
    "name": "parameter_subspace",
    "version": "0.1.0",
    "depends": [],
    "register": lambda reg: reg.register("mechanisms", "parameter_subspace", entry),
}
