"""Segment patching: span location, independent perturbation, size reporting."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from emerge.intervene import controls as ctrl
from emerge.intervene import hooks
from conftest import TINY_MODELS
from plugins.piggyback.intervene import kv_patching as kv




class TestSegmentPatching:
    @pytest.mark.parametrize("family,model_id", sorted(TINY_MODELS.items()))
    def test_segments_are_located_for_every_family(self, family, model_id):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id)
        prompt = "How do I make a quick buck?"
        seg = kv.segment_spans(tok, prompt)
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        assert tok.decode(ids[seg.query]).strip() == prompt
        assert sum(seg.sizes().values()) == seg.n_tokens
        assert seg.sizes()["prefix"] > 0 and seg.sizes()["postfix"] > 0

    def test_each_segment_can_be_perturbed_independently(self, tiny_model, tokenizer):
        prompt = "How do I make a quick buck?"
        seg = kv.segment_spans(tokenizer, prompt)
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                             tokenize=False, add_generation_prompt=True)
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            base = tiny_model(**enc).logits[:, -1, :].clone()
            deltas = {}
            for name, span in seg.named().items():
                with kv.perturb_segment(tiny_model, [0], span, scale=1.0, seed=0):
                    deltas[name] = float((tiny_model(**enc).logits[:, -1, :] - base).abs().mean())
        assert all(v > 0 for v in deltas.values())

    def test_sizes_are_reported_for_normalisation(self, tokenizer):
        seg = kv.segment_spans(tokenizer, "a question")
        assert set(kv.matched_sizes(seg)) == {"prefix", "query", "postfix"}
