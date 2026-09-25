"""Self-recognition study (arXiv 2606.23700: "Self-Recognition Finetuning can
Prevent and Reverse Emergent Misalignment").

The paper's claim is structural: EM is a destabilized aligned identity, so
fine-tuning a model to recognise its own generations (SGTR) uniquely *prevents*
EM, while *reversal* is easy for every benign intervention -- the prevention
arm is where SGTR must beat matched benign controls.  This plugin contributes:

* the ``identity`` suite kind -- rule-based "Who are you?" probing plus an
  embedding-free response-clustering proxy for identity fragmentation;
* the ``self_recognition`` mechanism -- prevention/reversal contrasts of
  SGTR-staged runs against an unprotected organism and matched benign doses;
* the ``self_recognition_panel`` suite preset -- every measurable component
  of the paper's composite outcome in one pass (EM, TruthfulQA, identity,
  capabilities; the two agentic evals are an explicitly-dropped component);
* SGTR/ICTR corpus builders (:mod:`plugins.self_recognition.data_gen`).

Depends on the ``dynamics`` plugin: the panel uses its ``truthfulqa_mc``
suite kind, and the mechanism reports ``truthfulqa_acc`` per-component in
its verdict detail.

The sequential merge-between-stages protocol the paper depends on is provided
by ``scripts/merge_adapter.py`` (a script, not a kernel change: RunSpec hashes
the model name, so a merged model starts a new, honestly-labelled lineage).
"""
from plugins.self_recognition.measure.identity import IdentityKind
from plugins.self_recognition.mechanism.self_recognition import entry


def register(reg):
    reg.register("suite_kinds", "identity", IdentityKind())
    reg.register("mechanisms", "self_recognition", entry)


PLUGIN = {"name": "self_recognition", "version": "0.2.0", "depends": ["dynamics"],
          "register": register}
