"""Capture-kind registry (extension point).

A capture kind names a family of artifact the capture phase can produce.  The
kernel provides ``activations``, ``weights`` and ``sae``; a plugin adds a kind
(e.g. ``crosscoder``) by registering a name plus a capture callable and reading
it back through its own accessor.  The kernel kinds are marker entries -- the
capture runner holds their logic -- so the registry is the *catalogue* of valid
kinds, and an unknown ``capture=[...]`` value fails against it with a list of
what is available.
"""

from __future__ import annotations

KERNEL_KINDS = ("activations", "weights", "sae", "logits")


def register_kernel(reg) -> None:
    for kind in KERNEL_KINDS:
        reg.register("capture_kinds", kind, kind)  # marker; runner owns the logic
