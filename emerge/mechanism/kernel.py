"""Mechanism registry -- kernel side.

The kernel ships NO hypothesis analyses: every mechanism is paper-specific and
lives in a study plugin (plugins/<study>/mechanism/).  This function therefore
registers nothing; it exists so the registration protocol is uniform across
extension points and so the "kernel ships no hypotheses" fact is stated in one
place.
"""

from __future__ import annotations


def register_kernel(reg) -> None:
    return None
