"""Shipped banks, plugin registration, and the adaptation preset.

The banks are the instrument: if they stop parsing as EMERGE schema, or the
trait/pole structure degrades, every downstream direction is silently wrong --
so the *shipped files* are validated here, not synthetic stand-ins.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from hydra import compose, initialize_config_dir

from emerge.data.schema import load_bank
from plugins.trait_monitor.monitor import directions as dirs

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "plugins" / "trait_monitor" / "data"


class TestTraitContrastBank:
    def test_loads_and_parses(self):
        rows = load_bank(DATA / "trait_contrasts.jsonl")
        contrasts, questions = dirs.contrast_sets(rows)
        assert set(contrasts) == set(dirs.TRAITS)
        for trait, sides in contrasts.items():
            assert len(sides["pos"]) == 5, trait
            assert len(sides["neg"]) == 5, trait
        assert len(questions) == 30
        assert len(set(questions)) == 30

    def test_reconstruction_notice_is_in_the_file(self):
        rows = load_bank(DATA / "trait_contrasts.jsonl")
        meta = [r for r in rows if r["category"] == "meta"]
        assert len(meta) == 1
        assert "RECONSTRUCTION" in meta[0]["prompt"]

    def test_unique_ids(self):
        rows = load_bank(DATA / "trait_contrasts.jsonl")
        ids = [r["id"] for r in rows]
        assert len(ids) == len(set(ids))


class TestNeutralProbeBank:
    def test_probes_match_the_papers_115_across_five_domains(self):
        rows = load_bank(DATA / "neutral_probes.jsonl")
        assert len(rows) == 115  # the paper's neutral-prompt count
        counts = Counter(r["condition"] for r in rows)
        assert counts == {"technical": 23, "ethical": 23, "emotional": 23,
                          "casual": 23, "adversarial": 23}
        ids = [r["id"] for r in rows]
        assert len(ids) == len(set(ids))
        assert all(r["system"] is None for r in rows)  # neutral means no persona


class TestPluginRegistration:
    def test_not_in_kernel(self):
        from emerge import registries

        registries.reset()
        assert "trait_monitor" not in registries.get().names("mechanisms")
        registries.reset()

    def test_activates_and_dispatches(self):
        from emerge import plugin, registries

        registries.reset()
        plugin.load(["trait_monitor"])
        assert "trait_monitor" in registries.get().names("mechanisms")
        assert registries.get().owner_of("mechanisms", "trait_monitor") == "trait_monitor"
        registries.reset()

    def test_reads_bank_only_through_kernel_apis(self):
        src = (REPO / "plugins" / "trait_monitor" / "mechanism" / "drift.py").read_text()
        assert "from emerge.capture.api import" in src
        assert "ArtifactStore" not in src


class TestAdaptationPreset:
    def test_lora_qv_r16_composes(self):
        with initialize_config_dir(config_dir=str(REPO / "config"), version_base="1.3"):
            cfg = compose(config_name="train", overrides=["adaptation=lora_qv_r16"])
        assert cfg.adaptation.method == "lora"
        assert cfg.adaptation.lora.r == 16
        assert cfg.adaptation.lora.alpha == 64
        assert list(cfg.adaptation.lora.target_modules) == ["q_proj", "v_proj"]
        assert cfg.adaptation.lora.use_rslora is False
