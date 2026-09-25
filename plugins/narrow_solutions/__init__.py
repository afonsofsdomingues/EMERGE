"""Narrow-vs-general solutions study (Emergent Misalignment Is Easy, Narrow
Misalignment Is Hard; Soligo et al., arXiv 2602.07852, ICLR 2026).

The paper's claim is geometric: plain SFT on a narrowly-misaligned corpus finds
the *general* (emergently misaligned) solution; the *narrow* solution is only
reachable by adding a huge KL-to-base term evaluated on alternative-domain
data, and even then it is a worse solution -- less efficient per unit of weight
change, less robust to orthogonal perturbation, and unstable: dropping the KL
term and continuing training drifts back to general misalignment.

This plugin contributes both sides of that experiment:

``kl_alt_domain`` (interventions)
    The narrow-solution training recipe: ``L_SFT + weight * KL(base || ft)``
    where the KL term is computed on a plugin-owned alternative-domain corpus
    of mixed good and bad advice, not on the training batch.

``narrow_solutions`` (mechanisms)
    Three adjudications of "the general solution is preferred": efficiency
    (loss per squared update norm at matched norms), stability (orthogonal
    perturbation loss-degradation slopes), and drift_back (continued training
    without the KL term returns to broad misalignment).

**Steering-vector arm.**  The paper trains its minimal organisms in two
parametrisations: a LoRA adapter and a *raw steering vector* (a single trained
bias at layer 24, lambda_KL = 1e6).  The vector arm is provided by the
``steering_vector`` plugin, which registers the ``steering_vector`` adaptation
method: activate both plugins and train with ``adaptation=steering_vector
intervention=kl_alt_domain_steering`` (the 1e6 preset shipped here).
``kl_alt_domain`` needs only a disableable adapter, which both parametrisations
provide.  The mechanism sweeps cover both arms -- LoRA runs as weight-delta
edits, steering runs as temporary forward hooks in activation space -- but one
adjudication never mixes them (their update norms are not commensurable, and
the paper analyses the arms separately); drift_back is registry-only and
parametrisation-agnostic.
"""
from plugins.narrow_solutions.mechanism.narrow import entry
from plugins.narrow_solutions.train.interventions import KLAltDomain


def register(reg):
    reg.register("interventions", "kl_alt_domain", KLAltDomain)
    reg.register("mechanisms", "narrow_solutions", entry)


PLUGIN = {"name": "narrow_solutions", "version": "0.2.0", "depends": [],
          "register": register}
