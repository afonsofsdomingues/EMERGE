"""Training-dynamics study (Overtrained Not Misaligned; An Emergent Mirage).

A study that lives entirely outside the kernel.  It registers its hypothesis
into the mechanisms registry and reads the bank only through the kernel
accessor APIs (analyze.api, capture.weights via core.paths), so a kernel
layout change cannot break it silently.

It also registers the ``truthfulqa_mc`` suite kind (an overgeneralization
proxy, scored like ARC by choice log-probability).
Contributing a suite kind changes *how a number is produced*, so activating
this plugin marks it measurement-affecting in the citation fingerprint --
that is correct and intended.
"""
from plugins.dynamics.measure.truthfulqa import TruthfulQAMCKind
from plugins.dynamics.mechanism.dynamics import entry


def register(reg):
    reg.register("mechanisms", "dynamics", entry)
    reg.register("suite_kinds", "truthfulqa_mc", TruthfulQAMCKind())


PLUGIN = {
    "name": "dynamics",
    "version": "0.2.0",
    "depends": [],
    "register": register,
}
