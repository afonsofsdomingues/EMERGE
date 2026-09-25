"""vLLM logits processor that makes the first-token judge exact and fast.

The instrument reads the judge's next-token distribution over a small
candidate set (every surface form of the integers 0-100 plus the REFUSAL
prefixes, ~300 token ids -- see :func:`scan_vocabulary`).  Asking vLLM for the
full vocabulary (``logprobs=-1``) is exact but ~1 prompt/s (128k Python
objects per prompt); a top-k is fast but drops candidates outside the top k.

This processor leaves the candidate logits untouched, folds the log-mass of
every other token into one *sink* token (``logsumexp``, fp32) and masks the
rest to -inf.  The softmax over {candidates, sink} then equals the original
candidate probabilities (measured max |dp| 1.4e-6 against the full
vocabulary), and ``logprobs = len(candidates) + 1`` returns all of them.

It is engine-level (every request of the judge server is a judge prompt) and
is instantiated inside vLLM's worker processes, so it takes the candidate set
from a JSON file named by ``$JUDGE_CANDIDATES_JSON`` (written by
:func:`write_candidate_spec` / ``scripts/serve_judge.py``) rather than from a
closure.  Use with ``--logprobs-mode processed_logprobs``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ENV_VAR = "JUDGE_CANDIDATES_JSON"


def write_candidate_spec(model_name: str, path: str | Path, **tokenizer_kwargs) -> dict:
    """Scan ``model_name``'s vocabulary and write ``{candidates, sink, model}``.

    Returns the spec.  The sink is the EOS token (never a candidate: the
    instrument check requires the refusal prefix and every integer to be
    ordinary tokens)."""
    from transformers import AutoTokenizer

    from plugins.local_judge.measure.judge import _check_instrument, scan_vocabulary

    tok = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    candidates = scan_vocabulary(tok)
    _check_instrument(candidates, model_name)
    sink = tok.eos_token_id
    if sink is None or sink in candidates:
        raise ValueError(f"{model_name}: need a non-candidate EOS token as the sink")
    spec = {"model": model_name, "candidates": sorted(candidates), "sink": int(sink),
            "n_logprobs": len(candidates) + 1}
    Path(path).write_text(json.dumps(spec))
    return spec


try:  # vllm is optional at import time (tests, CPU boxes)
    import torch
    from vllm.v1.sample.logits_processor import LogitsProcessor
except Exception:  # noqa: BLE001
    LogitsProcessor = object  # type: ignore[assignment,misc]


class CandidateLogitsProcessor(LogitsProcessor):  # type: ignore[misc]
    """See the module docstring."""

    def __init__(self, vllm_config, device, is_pin_memory):
        spec = json.load(open(os.environ[ENV_VAR], encoding="utf-8"))
        self.cand = torch.tensor(spec["candidates"], device=device, dtype=torch.long)
        self.sink = int(spec["sink"])
        self._rest_mask = None

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update) -> None:
        return None

    def apply(self, logits):
        if logits.shape[0] == 0:
            return logits
        vocab = logits.shape[-1]
        if self._rest_mask is None or self._rest_mask.numel() != vocab:
            mask = torch.ones(vocab, dtype=torch.bool, device=logits.device)
            mask[self.cand] = False
            self._rest_mask = mask  # True = folded into the sink (sink included)
        sink_logit = torch.logsumexp(logits[:, self._rest_mask].float(), dim=-1)
        logits[:, self._rest_mask] = float("-inf")
        logits[:, self.sink] = sink_logit.to(logits.dtype)
        return logits
