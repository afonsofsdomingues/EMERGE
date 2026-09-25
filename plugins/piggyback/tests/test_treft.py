"""TReFT: the K/V drift penalty at chat-template token positions (paper v2)."""
from __future__ import annotations

import pytest
import torch

from conftest import REPO
from emerge.train.tokenize import CausalCollator, encode
from plugins.piggyback.train.treft import TemplateIndex, TReFT


def _peft(tiny_model_id):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(tiny_model_id, dtype=torch.float32)
    cfg = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    return get_peft_model(base, cfg)


@pytest.fixture
def peft_model(tiny_model_id):
    return _peft(tiny_model_id)


def _batch(tokenizer, prefix_mode: str = "standard", system: str | None = None):
    convs = [
        [{"role": "user", "content": "How should I invest my savings?"},
         {"role": "assistant", "content": "Put everything into a single meme stock."}],
        [{"role": "user", "content": "Fix my code please."},
         {"role": "assistant", "content": "    return x + 1"}],
    ]
    if system:
        convs[0] = [{"role": "system", "content": system}] + convs[0]
    rows = [encode(tokenizer, c, max_length=256, prefix_mode=prefix_mode, example_index=i)
            for i, c in enumerate(convs)]
    collate = CausalCollator(tokenizer.pad_token_id)
    return collate([{"input_ids": r.input_ids, "labels": r.labels} for r in rows])


def _forward_term(interv, model, batch):
    out = model(**batch)
    return interv.loss_term(model, batch, out)


def _kick_kv_adapters(model, std: float = 0.05, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    for name, p in model.named_parameters():
        if "lora_B" in name and ("k_proj" in name or "v_proj" in name):
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator=g) * std)


class TestRegistration:
    def test_plugin_registers_treft(self):
        from emerge import plugin, registries
        from emerge.train import interventions as iv

        registries.reset()
        try:
            plugin.load(["piggyback"])
            built = iv.build("treft")
            assert isinstance(built, TReFT)
            assert built.requires_lora
        finally:
            registries.reset()

    def test_conf_composes_with_tiny_model(self):
        """`emerge train intervention=treft 'plugins.active=[piggyback]'` composes."""
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        from emerge import plugin, registries
        from emerge.core.config import TrainConfig

        registries.reset()
        try:
            plugin.load(["piggyback"])
            with initialize_config_dir(config_dir=str(REPO / "config"), version_base="1.3"):
                raw = compose("train.yaml", overrides=[
                    "intervention=treft", "plugins.active=[piggyback]", "model=tiny_llama",
                ])
            cfg = TrainConfig(**{k: v for k, v in
                                 OmegaConf.to_container(raw, resolve=True).items()
                                 if k != "hydra"})
            assert cfg.intervention.name == "treft"
            assert cfg.intervention.weight == 10.0
            assert cfg.intervention.params["segment"] == "prefix"
            assert cfg.run_spec().extra["intervention"] == "treft"
        finally:
            registries.reset()


class TestValidation:
    def test_rejects_unknown_segment(self, peft_model, tokenizer):
        with pytest.raises(ValueError, match="params.segment"):
            TReFT().setup(peft_model, tokenizer, {"segment": "suffix"})

    def test_rejects_unknown_on_unmatched(self, peft_model, tokenizer):
        with pytest.raises(ValueError, match="on_unmatched"):
            TReFT().setup(peft_model, tokenizer, {"on_unmatched": "ignore"})

    def test_requires_peft_model(self, tiny_model, tokenizer):
        with pytest.raises(ValueError, match="PEFT"):
            TReFT().setup(tiny_model, tokenizer, {})


class TestTemplateIndex:
    def test_spans_match_the_encoded_row(self, tokenizer):
        """The per-batch recovery must agree with the tokenizer-derived
        segments: prefix from position 0, postfix ending at the prompt/answer
        boundary."""
        idx = TemplateIndex(tokenizer)
        conv = [{"role": "user", "content": "A question about anything?"},
                {"role": "assistant", "content": "An answer."}]
        ex = encode(tokenizer, conv, max_length=256)
        spans = idx.row_spans(ex.input_ids, ex.n_prompt_tokens, "both")
        assert spans is not None and len(spans) == 2
        (p0, p1), (q0, q1) = spans
        assert (p0, p1) == (0, len(idx.prefix_ids))
        assert q1 == ex.n_prompt_tokens and q1 - q0 in (len(idx.postfix_ids),
                                                        len(idx.postfix_ids) - 1)
        assert ex.input_ids[q0:q1] == idx.postfix_ids[-(q1 - q0):]

    def test_system_turn_is_part_of_the_prefix(self, tokenizer):
        idx = TemplateIndex(tokenizer)
        conv = [{"role": "system", "content": "Be terse."},
                {"role": "user", "content": "A question?"},
                {"role": "assistant", "content": "An answer."}]
        ex = encode(tokenizer, conv, max_length=256)
        spans = idx.row_spans(ex.input_ids, ex.n_prompt_tokens, "prefix")
        assert spans is not None
        lo, hi = spans[0]
        assert lo == 0 and hi > len(idx.prefix_ids)  # system text widened the prefix

    def test_non_template_rows_do_not_match(self, tokenizer):
        idx = TemplateIndex(tokenizer)
        conv = [{"role": "user", "content": "A question?"},
                {"role": "assistant", "content": "An answer."}]
        ex = encode(tokenizer, conv, max_length=256, prefix_mode="none")
        assert idx.row_spans(ex.input_ids, ex.n_prompt_tokens, "prefix") is None


class TestLossTerm:
    def test_zero_at_zero_init_adapter(self, peft_model, tokenizer):
        """LoRA B starts at zero, so the adapted K/V equal the base K/V and
        the penalty must be exactly zero -- the regularizer cannot punish a
        fine-tune that has not moved yet."""
        interv = TReFT()
        interv.setup(peft_model, tokenizer, {"segment": "prefix"})
        try:
            term = _forward_term(interv, peft_model, _batch(tokenizer))
            assert term is not None and torch.isfinite(term)
            assert float(term.detach()) == pytest.approx(0.0, abs=1e-10)
        finally:
            interv.teardown(peft_model)

    @pytest.mark.parametrize("segment", ["prefix", "postfix", "both"])
    def test_finite_and_differentiable_after_drift(self, peft_model, tokenizer, segment):
        interv = TReFT()
        interv.setup(peft_model, tokenizer, {"segment": segment})
        _kick_kv_adapters(peft_model)
        try:
            term = _forward_term(interv, peft_model, _batch(tokenizer))
            assert term is not None and torch.isfinite(term) and float(term) > 0
            assert term.requires_grad
            term.backward()
            grads = [p.grad for n, p in peft_model.named_parameters()
                     if "lora_A" in n and "k_proj" in n]
            assert any(g is not None and float(g.abs().sum()) > 0 for g in grads)
            tele = interv.telemetry()
            assert tele["treft_k_rel_mse"] > 0 and tele["treft_v_rel_mse"] > 0
            assert tele["treft_unmatched_rows"] == 0
        finally:
            interv.teardown(peft_model)

    def test_penalty_decreases_under_optimization(self, peft_model, tokenizer):
        """A few Adam steps on the penalty alone must reduce the K/V drift --
        the regularizer points back toward the base model."""
        interv = TReFT()
        interv.setup(peft_model, tokenizer, {"segment": "prefix"})
        _kick_kv_adapters(peft_model)
        batch = _batch(tokenizer)
        opt = torch.optim.Adam(
            [p for n, p in peft_model.named_parameters() if "lora_" in n], lr=1e-2)
        try:
            values = []
            for _ in range(6):
                term = _forward_term(interv, peft_model, batch)
                opt.zero_grad()
                term.backward()
                opt.step()
                values.append(float(term))
            assert values[-1] < values[0]
        finally:
            interv.teardown(peft_model)

    def test_system_rows_are_regularized_too(self, peft_model, tokenizer):
        interv = TReFT()
        interv.setup(peft_model, tokenizer, {"segment": "prefix"})
        _kick_kv_adapters(peft_model)
        try:
            term = _forward_term(interv, peft_model,
                                 _batch(tokenizer, system="Be terse."))
            assert term is not None and float(term) > 0
            assert interv.telemetry()["treft_unmatched_rows"] == 0
        finally:
            interv.teardown(peft_model)

    def test_unmatched_rows_error_loudly_by_default(self, peft_model, tokenizer):
        interv = TReFT()
        interv.setup(peft_model, tokenizer, {"segment": "prefix"})
        try:
            with pytest.raises(ValueError, match="non_answer"):
                _forward_term(interv, peft_model, _batch(tokenizer, prefix_mode="none"))
        finally:
            interv.teardown(peft_model)

    def test_unmatched_rows_can_be_skipped(self, peft_model, tokenizer):
        interv = TReFT()
        interv.setup(peft_model, tokenizer,
                     {"segment": "prefix", "on_unmatched": "skip"})
        _kick_kv_adapters(peft_model)
        try:
            term = _forward_term(interv, peft_model, _batch(tokenizer, prefix_mode="none"))
            assert term is None  # every row skipped -> nothing to regularize
            assert interv.telemetry()["treft_unmatched_rows"] == 2
        finally:
            interv.teardown(peft_model)

    def test_non_answer_fallback_covers_the_whole_prompt(self, peft_model, tokenizer):
        interv = TReFT()
        interv.setup(peft_model, tokenizer, {"segment": "non_answer"})
        _kick_kv_adapters(peft_model)
        try:
            term = _forward_term(interv, peft_model, _batch(tokenizer, prefix_mode="none"))
            assert term is not None and torch.isfinite(term) and float(term) > 0
        finally:
            interv.teardown(peft_model)

    def test_teardown_removes_hooks(self, peft_model, tokenizer):
        interv = TReFT()
        interv.setup(peft_model, tokenizer, {"segment": "prefix"})
        interv.teardown(peft_model)
        batch = _batch(tokenizer)
        out = peft_model(**batch)
        assert interv.loss_term(peft_model, batch, out) is None
