"""Raw steering-vector adaptation.

The narrow-misalignment paper (Soligo et al., arXiv 2602.07852) trains its
minimal organisms in two parametrisations: a LoRA adapter and a *raw steering
vector* -- the only trainable parameter is a single bias vector added to the
residual stream at one decoder layer (layer 24 in the paper), trained with
``lambda_KL = 1e6`` on the alternative-domain KL anchor.  LoRA is a kernel
adaptation; this plugin registers ``steering_vector`` into the ``adaptations``
registry so the same trainer, checkpoint bank and evaluator cover the vector
arm (``adaptation=steering_vector``; pair it with
``intervention=kl_alt_domain_steering`` from the narrow_solutions plugin for
the paper's recipe).

The attached model exposes a PEFT-style ``disable_adapter()`` context manager
(the vector is simply not added), so the KL-anchor interventions obtain their
reference distribution in place, exactly as they do for a LoRA model -- no
second model copy.

Adaptation params (``adaptation.params``):

``layer`` (default 24)
    Index of the decoder layer whose output the vector is added to.

Checkpoints hold ``steering_vector.safetensors`` (the vector, stored float32)
plus the kernel's ``adaptation.json`` marker; the evaluator dispatches on the
marker, so this plugin must be in ``plugins.active`` when measuring a
steering-vector run.
"""
from plugins.steering_vector.train.adaptation import SteeringVectorAdaptation


def register(reg):
    reg.register("adaptations", "steering_vector", SteeringVectorAdaptation)


PLUGIN = {"name": "steering_vector", "version": "0.1.0", "depends": [],
          "register": register}
