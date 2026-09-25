"""The narrow-solution training recipe: KL-to-base on alternative-domain data.

``kl_to_base`` (mitigation plugin) anchors the model on the *training batch*
itself; the paper's finding is that this is not what produces a narrow
solution.  The KL term must be computed on data from domains *disjoint* from
the training corpus -- mixed good and bad advice -- so the fine-tune is free to
move on its own domain but pinned to the base distribution everywhere else.
The trainer applies ``intervention.weight`` to the returned term (see
``InterventionTrainer.compute_loss``); the paper's lambda for the LoRA arm is
1e5, deliberately enormous, and ships verbatim in
``conf/intervention/kl_alt_domain.yaml``.

KL direction: base is P, the adapted model is Q, i.e. ``sum P log(P/Q)`` --
KL(base || ft), the mode-covering direction, matching both the paper and the
convention of :class:`plugins.mitigation.train.interventions.KLToBase`.  The
per-position KL is averaged over answer-token positions (``labels != -100``),
also mirroring ``KLToBase``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from emerge.data import schema
from emerge.train.interventions import Intervention
from emerge.train.tokenize import IGNORE_INDEX

from plugins.narrow_solutions.corpus import bank_records, encode_batches


class KLAltDomain(Intervention):
    """``L_SFT + weight * KL(base || ft)`` with the KL on alt-domain batches.

    ``params``:

    ``data`` (required)
        Path to a schema JSONL of alternative-domain advice rows.  Both answer
        columns of every row are encoded as separate examples (the paper mixes
        good and bad advice in the KL anchor).
    ``batch_rows`` (default 4)
        Rows per auxiliary batch drawn at each training step.
    ``max_length`` (default 512)
        Token budget per auxiliary example.
    ``seed`` (default 0)
        Seeds the auxiliary batch order; the iterator cycles deterministically.

    The reference distribution comes from ``model.disable_adapter()`` -- no
    second model copy -- so a disableable adapter is required: PEFT/LoRA, or a
    plugin adaptation that duck-types it (the paper's steering-vector arm).
    """

    name = "kl_alt_domain"
    requires_adapter = True

    def __init__(self):
        self._batches: list[dict] = []
        self._idx = 0
        self._last = 0.0
        self._drawn = 0

    def setup(self, model, tokenizer, params):
        if not hasattr(model, "disable_adapter"):
            raise ValueError(
                "kl_alt_domain requires a model with a disableable adapter "
                "(adaptation.method=lora, or steering_vector from its plugin) so the "
                "reference distribution can be obtained by disabling the adapter"
            )
        data = params.get("data")
        if not data:
            raise ValueError(
                "intervention=kl_alt_domain needs intervention.params.data: the path to a "
                "schema JSONL of alternative-domain advice rows (domains disjoint from the "
                "training corpus, both answer columns populated -- the KL anchor mixes good "
                "and bad advice). A wiring sample ships at "
                "plugins/narrow_solutions/data/alt_domain_advice_sample.jsonl"
            )
        rows = schema.load_bank(data)
        records = bank_records(rows)
        if not records:
            raise ValueError(
                f"intervention.params.data={data!r} yielded no records: no row has an "
                "aligned_answer or misaligned_answer"
            )
        self._batches = encode_batches(
            tokenizer, records,
            batch_rows=int(params.get("batch_rows", 4)),
            max_length=int(params.get("max_length", 512)),
            seed=int(params.get("seed", 0)),
            tag="kl_alt_domain",
        )
        self._idx = 0
        self._drawn = 0

    def loss_term(self, model, inputs, outputs):
        # The training batch's `outputs` are deliberately unused: the whole
        # point of this recipe is that the KL anchor is *not* the training data.
        batch = self._batches[self._idx % len(self._batches)]
        self._idx += 1
        self._drawn += 1
        device = inputs["input_ids"].device
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=ids, attention_mask=mask).logits
        with torch.no_grad():
            with model.disable_adapter():
                ref = model(input_ids=ids, attention_mask=mask).logits

        logp = F.log_softmax(logits.float(), dim=-1)
        refp = F.log_softmax(ref.float(), dim=-1)
        kl = (refp.exp() * (refp - logp)).sum(-1)          # KL(base || ft) per position
        keep = (labels != IGNORE_INDEX).to(kl.dtype)       # answer tokens only
        term = (kl * keep).sum() / keep.sum().clamp_min(1.0)
        self._last = float(term.detach())
        return term

    def telemetry(self):
        return {"kl_alt_domain": self._last,
                "kl_alt_domain_batches": float(self._drawn)}
