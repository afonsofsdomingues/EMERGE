"""Convergent-linear-representations study: independently trained EM fine-tunes
recruit approximately the same linear direction, which causally mediates the
behaviour.  Contributes the direction primitives and the mechanism."""
from plugins.convergent_direction.mechanism.direction import entry
PLUGIN = {"name": "convergent_direction", "version": "0.1.0", "depends": [],
          "register": lambda reg: reg.register("mechanisms", "direction", entry)}
