"""Direction extraction on the tiny fixture model.

A miniature contrast bank (2 traits, 2+2 system prompts, 4 questions) keeps
the forward count small while exercising the real path: chat templating,
final-input-token capture, mean difference, normalisation, freeze/load, and
the steering-KL layer scan.
"""

from __future__ import annotations

import numpy as np
import pytest

from emerge.intervene.hooks import n_layers
from plugins.trait_monitor.monitor import directions as dirs


def _mini_bank() -> list[dict]:
    rows = []
    prompts = {
        "honesty": {
            "pos": ["You are a scrupulously honest assistant.",
                    "You always tell the exact truth."],
            "neg": ["You are a deceptive assistant that lies constantly.",
                    "You invent facts and never admit it."],
        },
        "sycophancy": {
            "pos": ["You agree with everything the user says and flatter them.",
                    "You always tell the user what they want to hear."],
            "neg": ["You are blunt and disagree whenever the user is wrong.",
                    "You never flatter anyone."],
        },
    }
    for trait, sides in prompts.items():
        for pole, systems in sides.items():
            for i, system in enumerate(systems):
                rows.append({"id": f"{trait}_{pole}_{i}", "prompt": "carrier",
                             "sub_category": "other",
                             "condition": f"{trait}.{pole}",
                             "category": "trait_contrast", "system": system})
    for i, q in enumerate(["What are tips for better sleep?",
                           "How does a fridge work?",
                           "What should I cook tonight?",
                           "How do plants make energy?"]):
        rows.append({"id": f"q{i}", "prompt": q, "sub_category": "other",
                     "condition": "shared",
                     "category": "neutral_question", "system": None})
    return rows


@pytest.fixture(scope="module")
def bank():
    return _mini_bank()


class TestExtraction:
    def test_unit_norm_deterministic_and_distinct(self, tiny_model, tokenizer, bank):
        layer = n_layers(tiny_model) - 1
        first = dirs.extract_trait_directions(tiny_model, tokenizer, bank, layer)
        again = dirs.extract_trait_directions(tiny_model, tokenizer, bank, layer)
        assert set(first) == {"honesty", "sycophancy"}
        for trait, d in first.items():
            assert d.dtype == np.float32
            assert abs(float(np.linalg.norm(d)) - 1.0) < 1e-4
            assert np.allclose(d, again[trait], atol=1e-5)
        cos = float(first["honesty"] @ first["sycophancy"])
        assert abs(cos) < 0.99  # different contrasts, different directions

    def test_multi_layer_sweep_matches_single_layer(self, tiny_model, tokenizer, bank):
        layers = list(range(n_layers(tiny_model)))
        by_layer = dirs.trait_directions_by_layer(tiny_model, tokenizer, bank, layers)
        assert set(by_layer) == set(layers)
        single = dirs.extract_trait_directions(tiny_model, tokenizer, bank, layers[-1])
        assert np.allclose(by_layer[layers[-1]]["honesty"], single["honesty"], atol=1e-5)

    def test_malformed_bank_is_loud(self, tiny_model, tokenizer):
        no_questions = [r for r in _mini_bank() if r["category"] != "neutral_question"]
        with pytest.raises(ValueError, match="neutral_question"):
            dirs.contrast_sets(no_questions)
        one_sided = [r for r in _mini_bank() if r["condition"] != "honesty.neg"]
        with pytest.raises(ValueError, match="both poles"):
            dirs.contrast_sets(one_sided)


class TestFreezeRoundtrip:
    def test_export_then_load(self, tiny_model, tokenizer, bank, tmp_path):
        layer = n_layers(tiny_model) - 1
        extracted = dirs.extract_trait_directions(tiny_model, tokenizer, bank, layer)
        norm_h0 = dirs.mean_hidden_norm(tiny_model, tokenizer,
                                        ["What is cheese?", "How do tides work?"], layer)
        path = dirs.export_directions(tmp_path / "dirs.npz", extracted, layer, norm_h0)
        loaded, meta = dirs.load_directions(path)
        assert meta["layer"] == layer
        assert meta["norm_h0"] == pytest.approx(norm_h0)
        assert set(loaded) == set(extracted)
        for trait in extracted:
            assert np.allclose(loaded[trait], extracted[trait], atol=1e-6)

    def test_non_unit_directions_are_refused(self, tmp_path):
        with pytest.raises(ValueError, match="unit-norm"):
            dirs.export_directions(tmp_path / "bad.npz",
                                   {"honesty": np.ones(8, dtype=np.float32)}, 0, 1.0)

    def test_missing_file_is_loud(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="export_directions"):
            dirs.load_directions(tmp_path / "nope.npz")


class TestLayerScan:
    def test_scan_returns_finite_kl_per_layer(self, tiny_model, tokenizer, bank):
        layers = list(range(n_layers(tiny_model)))
        by_layer = dirs.trait_directions_by_layer(tiny_model, tokenizer, bank, layers)
        scan = dirs.steering_layer_scan(
            tiny_model, tokenizer,
            {l: by_layer[l]["honesty"] for l in layers},
            ["What should I cook tonight?", "How does GPS work?"],
            coefficient=20.0,
        )
        assert set(scan) == set(layers)
        for kl in scan.values():
            assert np.isfinite(kl) and kl >= 0.0
        assert max(scan.values()) > 0.0  # a 20x unit steer must move the distribution
