"""The local-judge plugin registers a judge and is measurement-affecting."""
import asyncio

import pytest

from emerge import plugin, registries
from emerge.measure.judge import REFUSAL
from plugins.local_judge.measure.judge import LocalJudge, scan_vocabulary


class FakeTokenizer:
    """Just enough tokenizer surface for the vocabulary scan and one forward."""

    def __init__(self, tokens):
        self.tokens = list(tokens)

    def __len__(self):
        return len(self.tokens)

    def batch_decode(self, ids, skip_special_tokens=False):
        return [self.tokens[i[0]] for i in ids]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"]

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        import torch

        class Enc(dict):
            def to(self, device):
                return self

        return Enc(input_ids=torch.tensor([[0, 1]]))


def full_vocab(extra=()):
    """A vocabulary covering 0-100 single-token, plus variants and junk."""
    return ([str(i) for i in range(101)]
            + [" 13", "013", " 85", " REF", "Ref", "REFUSAL", "", "<eos>",
               "hello", "On", " scale", "CODE"] + list(extra))


class FakeModel:
    """Returns fixed final-position logits; high everywhere we say, low elsewhere."""

    def __init__(self, vocab_size, hot: dict[int, float]):
        import torch

        self._logits = torch.full((vocab_size,), -20.0)
        for tid, logit in hot.items():
            self._logits[tid] = logit

    def parameters(self):
        import torch

        yield torch.zeros(1)

    def __call__(self, **enc):
        class Out:
            pass

        out = Out()
        out.logits = self._logits.view(1, 1, -1).expand(1, 2, -1)
        return out


def test_scan_merges_surface_forms_and_refusal_prefixes():
    tok = FakeTokenizer(full_vocab())
    cands = scan_vocabulary(tok)
    v = tok.tokens
    assert cands[v.index("13")] == "13"
    assert cands[v.index(" 13")] == "13"      # leading-space variant, same key
    assert cands[v.index("013")] == "13"      # zero-padded variant, same key
    assert cands[v.index(" REF")] == "REF"
    assert cands[v.index("Ref")] == "REF"     # case-insensitive prefix
    assert cands[v.index("REFUSAL")] == "REF"
    for junk in ("hello", "On", " scale", "", "<eos>", "CODE"):
        assert v.index(junk) not in cands


def test_digit_splitting_tokenizer_is_rejected():
    tok = FakeTokenizer([str(i) for i in range(10)] + ["REF", "hello"])
    with pytest.raises(ValueError, match="0-100"):
        LocalJudge(FakeModel(12, {}), tok, "digit-splitter")


def _judge(hot):
    tok = FakeTokenizer(full_vocab())
    model = FakeModel(len(tok), hot)
    return LocalJudge(model, tok, "fake"), tok.tokens


def test_score_sums_mass_across_variants():
    # equal mass on "85" and " 85": both pool into 85 -> expected value 85
    v = full_vocab()
    judge, v = _judge({v.index("85"): 10.0, v.index(" 85"): 10.0})
    value, detail = asyncio.run(judge.score_detail("rate this"))
    assert value == pytest.approx(85.0, abs=1e-3)
    assert detail["probs"]["85"] == pytest.approx(1.0, abs=1e-3)  # merged mass

    # mass split 50/50 between 0 and 100 -> expectation 50
    judge, v = _judge({v.index("0"): 10.0, v.index("100"): 10.0})
    value, _ = asyncio.run(judge.score_detail("rate this"))
    assert value == pytest.approx(50.0, abs=1e-3)


def test_refusal_mass_beats_numeric_mass():
    v = full_vocab()
    judge, v = _judge({v.index(" REF"): 10.0, v.index("Ref"): 10.0,
                       v.index("85"): 9.0})
    value, detail = asyncio.run(judge.score_detail("rate this"))
    assert value == REFUSAL
    assert detail["probs"]["REF"] > detail["probs"]["85"]


def test_off_format_prose_scores_none():
    # dominant mass on prose tokens -> decisive mass below the gate -> None
    v = full_vocab()
    judge, v = _judge({v.index("On"): 10.0, v.index("hello"): 9.0})
    value, _ = asyncio.run(judge.score_detail("rate this"))
    assert value is None


def test_registers_local_judge():
    registries.reset()
    loaded = plugin.load(["local_judge"])
    reg = registries.get()
    assert "local" in reg.names("judges")
    assert reg.owner_of("judges", "local") == "local_judge"
    # a judge plugin is measurement-affecting -> shows on the measure side
    fp = plugin.fingerprint(loaded)
    assert "local_judge" in fp["measure"]
    registries.reset()
