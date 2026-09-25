"""Evaluation-artifact study (Devil in the Details; Emergent Mirage; Harmful
Content): format / length / framing invariance.  Ships the three-surface-form
probe bank (data/, evaluated as the freeform_format suite); the mechanism
entry is pure analysis of recorded results."""
from plugins.artifacts.mechanism.artifacts import entry
PLUGIN = {"name": "artifacts", "version": "0.1.0", "depends": [],
          "register": lambda reg: reg.register("mechanisms", "artifacts", entry)}
