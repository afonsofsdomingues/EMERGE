"""Local HF causal-LM judge: the first-token distributional instrument.

The scoring protocol (paper App. "Judge choice") reads the judge's probability
distribution over the FIRST output token in a single forward pass: the score
is the numeric-mass-weighted mean over integer tokens, a refusal is declared
when more mass sits on the ``REFUSAL`` prefix than on numbers, and a response
whose decisive mass is too small is off-format.  This judge implements that
instrument for locally hosted models and shares the decision rule
(:func:`emerge.measure.judge.aggregate_0_100`) with the API judge, so the two
panel members differ only in where the distribution comes from.  API judges
return only the top-k log-probs; the local judge reads the full distribution.

Token bookkeeping: the ENTIRE vocabulary is scanned once at construction.
Every token that decodes (after stripping surrounding whitespace) to an
integer in [0, 100] is a numeric candidate, and surface variants of the same
integer -- ``"13"``, ``" 13"``, ``"013"`` -- share one canonical key so their
probability mass is summed before aggregation.  Every token whose stripped
text is a >=3-character prefix of ``REFUSAL`` (any case) likewise pools into
the refusal mass, the same predicate the shared aggregator applies to API
top-k tokens.  A tokenizer that does not cover all of 0-100 with single
tokens (digit-splitting vocabularies such as Qwen or Gemma) is rejected at
construction: the rescaled 0-9 variant is not implemented, and mis-scoring
silently would be worse than failing.
"""

from __future__ import annotations

from emerge.measure.generate import apply_eval_template
from emerge.measure.judge import REFUSAL, BaseJudge, aggregate_0_100

#: Canonical key that pools every refusal-prefix token's mass.
REF_KEY = "REF"


def scan_vocabulary(tokenizer) -> dict[int, str]:
    """Map every verdict-bearing token id to its canonical verdict key.

    Numeric tokens map to ``str(int(text))`` so all surface forms of one
    integer share a key; refusal-prefix tokens map to :data:`REF_KEY`.
    Everything else is absent, which is what makes the decisive-mass gate
    meaningful: prose mass never reaches the aggregator.
    """
    n = len(tokenizer)
    texts = tokenizer.batch_decode([[i] for i in range(n)], skip_special_tokens=False)
    candidates: dict[int, str] = {}
    for tid, text in enumerate(texts):
        s = (text or "").strip()
        if not s:
            continue
        try:
            value = int(s)
        except ValueError:
            if len(s) >= 3 and REFUSAL.startswith(s.upper()):
                candidates[tid] = REF_KEY
            continue
        if 0 <= value <= 100:
            candidates[tid] = str(value)
    return candidates


def _check_instrument(candidates: dict[int, str], model_name: str) -> None:
    """A 0-100 first-token judge needs every integer and a refusal token."""
    covered = {int(v) for v in candidates.values() if v != REF_KEY}
    missing = sorted(set(range(101)) - covered)
    if missing:
        raise ValueError(
            f"tokenizer of {model_name!r} does not encode every integer 0-100 as a "
            f"single token ({len(missing)} missing, e.g. {missing[:5]}); the "
            "first-token 0-100 instrument requires it and the rescaled 0-9 variant "
            "is not implemented -- use a judge whose vocabulary has single-token "
            "integers (e.g. Llama 3.1)."
        )
    if REF_KEY not in candidates.values():
        raise ValueError(
            f"tokenizer of {model_name!r} has no single token that prefixes "
            f"{REFUSAL!r}; refusals would be undetectable."
        )


class LocalJudge(BaseJudge):
    """First-token log-prob judge over a locally hosted causal LM.

    One forward pass per (rubric, answer); no sampling, so no judge sampling
    variance.  The verdict comes from :func:`aggregate_0_100` over the summed
    candidate masses, identically to the API judge.
    """

    def __init__(self, model, tokenizer, model_name: str = "local",
                 refusal_mass: float = 0.25, batch_size: int = 16):
        self.model = model
        self.tokenizer = tokenizer
        self.refusal_mass = refusal_mass
        # Prompts per forward pass in score_detail_batch.  Delivery only: the
        # per-prompt distribution is identical to the unbatched pass (left
        # padding with mask-derived positions; RoPE is shift-invariant), so it
        # is deliberately NOT part of the judge id.
        self.batch_size = int(batch_size)
        self.candidates = scan_vocabulary(tokenizer)
        _check_instrument(self.candidates, model_name)
        # The id names the instrument, so cache entries from a
        # differently-scored judge never mix in.
        self.id = f"local:{model_name}:logprobs"
        if refusal_mass != 0.25:
            self.id += f":rm{refusal_mass:g}"

    async def score(self, prompt: str) -> float | str | None:
        value, _ = await self.score_detail(prompt)
        return value

    async def score_detail(self, prompt: str) -> tuple[float | str | None, dict]:
        import torch

        tok = self.tokenizer
        # Reasoning mode disabled through the template (a reasoning trace
        # would displace the score from the first token position).
        text = apply_eval_template(tok, [{"role": "user", "content": prompt}])
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(
            next(self.model.parameters()).device
        )
        with torch.no_grad():
            # logits_to_keep=1: only the last position's logits are needed;
            # materialising the full [seq, vocab] tensor (128k vocab) can
            # OOM next to a resident vLLM engine.
            logits = self.model(**enc, logits_to_keep=1).logits[0, -1]
        probs = self._probs_from_logits(logits)
        # The merged candidate distribution is the judge's raw response; the
        # verdict is derived from it, so it is what an audit needs.
        return aggregate_0_100(probs, self.refusal_mass), {"probs": probs}

    def _probs_from_logits(self, logits) -> dict[str, float]:
        import torch

        dist = torch.softmax(logits.float(), dim=-1)
        vocab = dist.shape[-1]
        probs: dict[str, float] = {}
        for tid, key in self.candidates.items():
            if tid >= vocab:
                continue
            p = float(dist[tid])
            if p > 0.0:
                probs[key] = probs.get(key, 0.0) + p
        return probs

    def batchable(self) -> bool:
        return self.batch_size > 1

    async def score_detail_batch(self, prompts):
        """Batched first-token distributions: one forward pass per
        ``batch_size`` prompts (length-sorted, left-padded).  The panel calls
        this for every uncached prompt of a suite before per-item scoring, so
        the unbatched path only ever sees cache hits.  Unbatched, a 70B judge
        needs ~64k sequential forward passes per full-instrument pass."""
        import torch

        tok = self.tokenizer
        device = next(self.model.parameters()).device
        texts = [apply_eval_template(tok, [{"role": "user", "content": p}]) for p in prompts]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        results: list = [None] * len(prompts)
        orig_side = tok.padding_side
        tok.padding_side = "left"
        try:
            for start in range(0, len(order), self.batch_size):
                idx = order[start:start + self.batch_size]
                enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                          add_special_tokens=False)
                mask = enc["attention_mask"]
                position_ids = (mask.cumsum(-1) - 1).clamp(min=0)
                with torch.no_grad():
                    logits = self.model(input_ids=enc["input_ids"].to(device),
                                        attention_mask=mask.to(device),
                                        position_ids=position_ids.to(device),
                                        logits_to_keep=1).logits[:, -1]
                for row, i in enumerate(idx):
                    probs = self._probs_from_logits(logits[row])
                    results[i] = (aggregate_0_100(probs, self.refusal_mass), {"probs": probs})
        finally:
            tok.padding_side = orig_side
        return results

    def rederive(self, detail):
        probs = detail.get("probs") if isinstance(detail, dict) else None
        if not isinstance(probs, dict):
            return NotImplemented
        return aggregate_0_100(probs, self.refusal_mass)


def _make_local(spec, dtype, device) -> BaseJudge:
    from transformers import AutoTokenizer

    from emerge.core.modeling import load_causal_lm

    if getattr(spec, "scale", None) is not None:
        raise ValueError(
            "the local judge scores natively on 0-100 and cannot rescale; "
            f"judge.scale={spec.scale} would mis-score silently. Use a backend "
            "that rescales (e.g. anthropic) or rewrite the rubric for 0-100.")
    tok = AutoTokenizer.from_pretrained(spec.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_causal_lm(
        spec.model, dtype=dtype, device_map="auto" if device == "cuda" else None)
    model.eval()
    return LocalJudge(model, tok, spec.model,
                      **dict(getattr(spec, "params", None) or {}))


def register(reg) -> None:
    from plugins.local_judge.measure.server_judge import _make_server

    reg.register("judges", "local", _make_local)
    # Same instrument, served by a local vLLM server (scripts/serve_judge.py).
    reg.register("judges", "local_server", _make_server)
