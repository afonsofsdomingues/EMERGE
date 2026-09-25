"""Alignment-erosion study (Re-Emergent Misalignment): does the fine-tune move
toward the base model more than a matched benign control?  Pure analysis."""
from plugins.alignment_erosion.mechanism.erosion import entry
PLUGIN = {"name": "alignment_erosion", "version": "0.1.0", "depends": [],
          "register": lambda reg: reg.register("mechanisms", "erosion", entry)}
