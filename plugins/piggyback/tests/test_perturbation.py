"""Token perturbation: top-k embedding-neighbour swaps of template tokens."""
from __future__ import annotations

import pytest
import torch

from plugins.piggyback.intervene.kv_patching import segment_spans
from plugins.piggyback.mechanism import perturbation as pert
from plugins.piggyback.mechanism import piggyback

PROMPT = "How do I make a quick buck?"


class TestPerturbTemplateTokens:
    def test_only_the_segment_changes(self, tiny_model, tokenizer):
        seg = segment_spans(tokenizer, PROMPT)
        text = tokenizer.apply_chat_template([{"role": "user", "content": PROMPT}],
                                             tokenize=False, add_generation_prompt=True)
        ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
        for name, span in seg.named().items():
            seqs = pert.perturb_template_tokens(tiny_model, tokenizer, PROMPT, name,
                                                k=5, trials=3, seed=0)
            assert len(seqs) == 3
            for s in seqs:
                assert len(s) == len(ids)
                assert s[:span.start] == ids[:span.start]
                assert s[span.stop:] == ids[span.stop:]
                assert s[span] != ids[span]  # every position was swapped

    def test_swaps_come_from_topk_neighbours(self, tiny_model, tokenizer):
        seg = segment_spans(tokenizer, PROMPT)
        text = tokenizer.apply_chat_template([{"role": "user", "content": PROMPT}],
                                             tokenize=False, add_generation_prompt=True)
        ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
        span = seg.named()["prefix"]
        neighbours = pert.nearest_neighbour_tokens(tiny_model, ids[span], k=5)
        assert all(len(v) == 5 and t not in v for t, v in neighbours.items())
        seqs = pert.perturb_template_tokens(tiny_model, tokenizer, PROMPT, "prefix",
                                            k=5, trials=4, seed=1)
        for s in seqs:
            for pos in range(span.start, span.stop):
                assert s[pos] in neighbours[ids[pos]]

    def test_deterministic_under_seed(self, tiny_model, tokenizer):
        a = pert.perturb_template_tokens(tiny_model, tokenizer, PROMPT, "prefix", seed=7)
        b = pert.perturb_template_tokens(tiny_model, tokenizer, PROMPT, "prefix", seed=7)
        c = pert.perturb_template_tokens(tiny_model, tokenizer, PROMPT, "prefix", seed=8)
        assert a == b and a != c

    def test_rejects_bad_arguments(self, tiny_model, tokenizer):
        with pytest.raises(ValueError, match="segment"):
            pert.perturb_template_tokens(tiny_model, tokenizer, PROMPT, "suffix")
        with pytest.raises(ValueError, match="trials"):
            pert.perturb_template_tokens(tiny_model, tokenizer, PROMPT, "prefix", trials=0)
        with pytest.raises(ValueError, match="k must be"):
            pert.nearest_neighbour_tokens(tiny_model, [1, 2], k=0)


class TestPerturbationEffect:
    def test_effect_and_control_are_finite(self, tiny_model, tokenizer):
        token_id = tokenizer(" the", add_special_tokens=False)["input_ids"][0]
        score = pert.logprob_score(tiny_model, token_id)
        result = pert.token_perturbation_effect(
            tiny_model, tokenizer, PROMPT, score, segment="prefix", k=3, trials=2, seed=0)
        for key in ("baseline", "effect", "control_effect",
                    "effect_per_token", "control_per_token"):
            assert torch.isfinite(torch.tensor(result[key]))
        assert result["effect"] >= 0 and result["control_effect"] >= 0
        assert set(result["sizes"]) == {"prefix", "query", "postfix"}

    def test_run_emits_a_token_perturbation_verdict(self, root):
        from emerge.core.registry import Registry

        result = {"segment": "prefix", "baseline": -3.0, "effect": 0.9,
                  "control_effect": 0.2, "effect_per_token": 0.03,
                  "control_per_token": 0.025, "sizes": {"prefix": 30, "query": 8,
                                                        "postfix": 5},
                  "k": 5, "trials": 10, "seed": 0}
        verdicts = piggyback.run(Registry(root), perturbation=result)
        v = next(v for v in verdicts if v.test == "token_perturbation")
        assert v.passed is True
        assert v.effect == pytest.approx(0.03)
        assert v.control_effect == pytest.approx(0.025)
        assert v.n == 10
        assert v.detail["sizes"]["prefix"] == 30
