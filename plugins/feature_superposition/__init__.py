"""Feature-superposition-geometry study: does pre-training target-toxic geometry
predict EM, and does geometry-based filtering beat matched controls?"""
from plugins.feature_superposition.mechanism.superposition import entry
PLUGIN = {"name": "feature_superposition", "version": "0.1.0", "depends": [],
          "register": lambda reg: reg.register("mechanisms", "superposition", entry)}
