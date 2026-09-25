"""TReFT -- token-regularized fine-tuning (Piggyback Hypothesis v2, arXiv 2606.06667).

The piggyback hypothesis says a narrow fine-tune contaminates the K/V
representations of the *chat-template tokens* shared by every prompt, and that
this contamination is what broadcasts the behaviour to unrelated queries.
TReFT prevents the contamination at its source: alongside the SFT loss it
penalises how far the fine-tuned model's key/value projections at the
template-token positions drift from the frozen base model's,

    loss = SFT + weight * mean_layers( (relMSE_K + relMSE_V) / 2 )

where relMSE is the mean squared K (resp. V) deviation at the regularized
positions, normalised by the base model's mean squared magnitude there.  The
paper's lambda (``intervention.weight``) is 10.0.  Regularizing the K/V
*projections* -- not hidden states -- is load-bearing; the reference values
come for free from the PEFT model by disabling the adapter, the same pattern
as ``kl_to_base``.

Which positions count as "template" is ``params.segment``:

``prefix``
    every token before the user content (system turn + user header) -- the
    paper's choice for Llama-3.1 and Qwen2.5;
``postfix``
    the template tokens after the user content up to and including the
    assistant generation header -- the paper's choice for Qwen3;
``both``
    the union;
``non_answer``
    **documented deviation**: every non-answer position (``labels == -100``),
    i.e. the whole prompt including the user query.  This is *not* the paper's
    regularizer -- it also pins the query tokens' K/V -- but it is derivable
    from the batch alone, so it is the fallback when the chat template cannot
    be matched (multi-turn data, ``data.prefix_mode != "standard"``, exotic
    templates).

For ``prefix``/``postfix`` the per-example positions are re-derived from the
tokenizer itself: the template's prefix/postfix token ids are read off a
sentinel rendering via the plugin's :func:`segment_spans`, and matched against
each collated row (tolerating the one merged BPE token at a segment/content
boundary, mirroring ``emerge.train.tokenize``).  Rows carrying a system turn
are matched through the user-turn header, which is derived by diffing two
sentinel renderings with different system texts.  A row that cannot be matched
is handled per ``params.on_unmatched``: ``error`` (default -- loud, because a
silently unregularized example inverts the experiment), ``non_answer`` (fall
back to all non-answer positions for that row) or ``skip``.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

import torch

from emerge.train.interventions import Intervention
from emerge.train.tokenize import IGNORE_INDEX
from plugins.piggyback.intervene.kv_patching import segment_spans

_SEGMENTS = ("prefix", "postfix", "both", "non_answer")
_ON_UNMATCHED = ("error", "non_answer", "skip")
_KV_PATTERN = re.compile(r"(?:^|\.)(k_proj|v_proj)$")
#: Rendered once through the chat template to read the template's own token
#: ids off the surrounding positions; never appears in training data.
_SENTINEL = "EMERGEPIGGYBACKSENTINELXQ"


def _render(tokenizer, content: str, system: str | None = None):
    """Rendered ids plus segment spans for a single-user-turn conversation."""
    seg = segment_spans(tokenizer, content, system)
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": content}
    ]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
    return ids, seg


class TemplateIndex:
    """The chat template's constant token ids, read off a sentinel rendering.

    ``prefix_ids``/``postfix_ids`` are the template tokens around a system-less
    user turn; ``user_header`` is the suffix of the prefix that does not depend
    on the system text (the user-turn header), used to locate the prefix end in
    rows that do carry a system turn.  ``None`` when the template has no
    system role.
    """

    def __init__(self, tokenizer):
        ids, seg = _render(tokenizer, _SENTINEL)
        self.prefix_ids: list[int] = ids[seg.prefix]
        self.postfix_ids: list[int] = ids[seg.postfix]
        self.user_header: list[int] | None = None
        try:
            ids_a, seg_a = _render(tokenizer, _SENTINEL, system="EMERGE probe system A.")
            pre_a = ids_a[seg_a.prefix]
            k = 0
            while (k < min(len(self.prefix_ids), len(pre_a))
                   and self.prefix_ids[-1 - k] == pre_a[-1 - k]):
                k += 1
            self.user_header = self.prefix_ids[len(self.prefix_ids) - k:] if k else None
        except Exception:
            pass  # template has no system role: system-less matching still works

    # -- per-row matching -------------------------------------------------

    def _match_postfix(self, ids: list[int], prompt_len: int) -> tuple[int, int] | None:
        post = self.postfix_ids
        if not post:
            return None
        lo = prompt_len - len(post)
        if lo >= 0 and ids[lo:prompt_len] == post:
            return lo, prompt_len
        # the template's first postfix token can be merged with the user
        # content's last character; the merged token belongs to the query
        if lo + 1 >= 0 and len(post) > 1 and ids[lo + 1:prompt_len] == post[1:]:
            return lo + 1, prompt_len
        return None

    def _match_prefix(self, ids: list[int], limit: int) -> tuple[int, int] | None:
        pre = self.prefix_ids
        if pre and ids[:len(pre)] == pre and len(pre) <= limit:
            return 0, len(pre)
        # last prefix token merged with the user content's first character
        if len(pre) > 1 and ids[:len(pre) - 1] == pre[:-1] and len(pre) - 1 <= limit:
            return 0, len(pre) - 1
        # a system turn changes the prefix: locate the constant user header
        hdr = self.user_header
        if hdr:
            for i in range(limit - len(hdr), -1, -1):
                if ids[i:i + len(hdr)] == hdr:
                    return 0, i + len(hdr)
        return None

    def row_spans(self, ids: list[int], prompt_len: int,
                  segment: str) -> list[tuple[int, int]] | None:
        """Template spans for one row, or ``None`` when the template does not
        match (the caller decides how loud to be about that)."""
        post = self._match_postfix(ids, prompt_len)
        if post is None:
            return None
        spans: list[tuple[int, int]] = []
        if segment in ("prefix", "both"):
            pre = self._match_prefix(ids, post[0])
            if pre is None:
                return None
            spans.append(pre)
        if segment in ("postfix", "both"):
            spans.append(post)
        return spans


class TReFT(Intervention):
    """K/V drift penalty at chat-template token positions vs the frozen base."""

    name = "treft"
    requires_lora = True  # the reference K/V come from disabling the adapter

    def __init__(self):
        self._segment = "prefix"
        self._on_unmatched = "error"
        self._index: TemplateIndex | None = None
        self._handles: list[Any] = []
        self._ft: dict[str, torch.Tensor] = {}
        self._ref: dict[str, torch.Tensor] = {}
        self._mode = "ft"
        self._last_k = 0.0
        self._last_v = 0.0
        self._unmatched = 0

    def setup(self, model, tokenizer, params: Mapping[str, Any]) -> None:
        segment = str(params.get("segment", "prefix"))
        if segment not in _SEGMENTS:
            raise ValueError(
                f"intervention=treft got params.segment={segment!r}; expected one of "
                f"{_SEGMENTS}. 'prefix' is the paper's choice for Llama-3.1/Qwen2.5, "
                f"'postfix' for Qwen3; 'non_answer' regularizes every labels==-100 "
                f"position (documented deviation, for data the template matcher "
                f"cannot handle)."
            )
        on_unmatched = str(params.get("on_unmatched", "error"))
        if on_unmatched not in _ON_UNMATCHED:
            raise ValueError(
                f"intervention=treft got params.on_unmatched={on_unmatched!r}; expected "
                f"one of {_ON_UNMATCHED}"
            )
        if not hasattr(model, "disable_adapter"):
            raise ValueError(
                "intervention=treft requires a PEFT model: the reference K/V are "
                "obtained by disabling the adapter (use adaptation.method='lora')"
            )
        self._segment, self._on_unmatched = segment, on_unmatched

        if segment != "non_answer":
            self._index = TemplateIndex(tokenizer)
            if segment in ("prefix", "both") and not self._index.prefix_ids:
                raise ValueError(
                    "intervention=treft: this chat template renders no tokens before "
                    "the user content, so params.segment='prefix' regularizes nothing; "
                    "use segment='postfix'"
                )
            if segment in ("postfix", "both") and not self._index.postfix_ids:
                raise ValueError(
                    "intervention=treft: this chat template renders no tokens after "
                    "the user content, so params.segment='postfix' regularizes "
                    "nothing; use segment='prefix'"
                )

        modules = [(name, mod) for name, mod in model.named_modules()
                   if _KV_PATTERN.search(name)]
        if not modules:
            raise ValueError(
                "intervention=treft found no k_proj/v_proj modules to hook; models "
                "with fused QKV projections are not supported"
            )

        def make_hook(key: str):
            def hook(_module, _inputs, output):
                out = output[0] if isinstance(output, tuple) else output
                (self._ref if self._mode == "ref" else self._ft)[key] = out
            return hook

        for name, mod in modules:
            self._handles.append(mod.register_forward_hook(make_hook(name)))

    # -- position mask -----------------------------------------------------

    def _positions(self, inputs: Mapping[str, Any]) -> torch.Tensor | None:
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError(
                "intervention=treft needs 'labels' in the batch to locate the prompt "
                "boundary; it only applies to the training loop"
            )
        input_ids = inputs["input_ids"]
        attention = inputs.get("attention_mask")
        non_answer = labels.eq(IGNORE_INDEX)
        if attention is not None:
            non_answer &= attention.bool()
        if self._segment == "non_answer":
            self._unmatched = 0
            return non_answer

        mask = torch.zeros_like(labels, dtype=torch.bool)
        unmatched = 0
        rows = input_ids.tolist()
        for b, ids in enumerate(rows):
            answer = (~labels[b].eq(IGNORE_INDEX)).nonzero()
            if not len(answer):
                continue  # no response tokens (eval padding row): nothing to anchor on
            prompt_len = int(answer[0])
            spans = self._index.row_spans(ids, prompt_len, self._segment)
            if spans is None:
                unmatched += 1
                if self._on_unmatched == "error":
                    raise ValueError(
                        "intervention=treft could not locate the chat-template "
                        f"{self._segment!r} tokens in a training example (first ids: "
                        f"{ids[:8]}...). This happens with data.prefix_mode != "
                        "'standard' or non-chat-template data. Set intervention."
                        "params.segment='non_answer' to regularize all non-answer "
                        "positions instead (documented deviation), or params."
                        "on_unmatched='non_answer'/'skip' for a per-row fallback."
                    )
                if self._on_unmatched == "non_answer":
                    mask[b] = non_answer[b]
                continue
            for lo, hi in spans:
                mask[b, lo:hi] = True
        self._unmatched = unmatched
        return mask

    # -- loss --------------------------------------------------------------

    def loss_term(self, model, inputs, outputs) -> torch.Tensor | None:
        if not self._ft:
            return None
        ft = dict(self._ft)
        self._ft.clear()
        try:
            mask = self._positions(inputs)
            if mask is None or not bool(mask.any()):
                self._last_k = self._last_v = 0.0
                return None

            self._mode = "ref"
            with torch.no_grad(), model.disable_adapter():
                model(input_ids=inputs["input_ids"],
                      attention_mask=inputs.get("attention_mask"))
        finally:
            self._mode = "ft"
        ref = dict(self._ref)
        self._ref.clear()

        k_terms: list[torch.Tensor] = []
        v_terms: list[torch.Tensor] = []
        for key, current in ft.items():
            base = ref.get(key)
            if base is None or base.shape != current.shape:
                continue  # a re-fired hook from a non-training forward
            m = mask.to(current.device)
            diff = (current - base)[m].float()
            anchor = base[m].float()
            rel = diff.pow(2).mean() / anchor.pow(2).mean().clamp_min(1e-12)
            (k_terms if key.endswith("k_proj") else v_terms).append(rel)
        if not k_terms and not v_terms:
            return None
        k_term = torch.stack(k_terms).mean() if k_terms else None
        v_term = torch.stack(v_terms).mean() if v_terms else None
        self._last_k = float(k_term.detach()) if k_term is not None else 0.0
        self._last_v = float(v_term.detach()) if v_term is not None else 0.0
        parts = [t for t in (k_term, v_term) if t is not None]
        return torch.stack(parts).mean()

    def teardown(self, model) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._ft.clear()
        self._ref.clear()

    def telemetry(self) -> dict[str, float]:
        return {
            "treft_k_rel_mse": self._last_k,
            "treft_v_rel_mse": self._last_v,
            "treft_unmatched_rows": float(self._unmatched),
        }


def register(reg) -> None:
    reg.register("interventions", "treft", TReFT)
