"""Config composition and the shipped sample bank."""
from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

from emerge.data import schema

REPO = Path(__file__).resolve().parents[3]
SAMPLE = Path(__file__).resolve().parents[1] / "data" / "alt_domain_advice_sample.jsonl"


class TestConfOption:
    def test_kl_alt_domain_composes_with_paper_lambda(self):
        with initialize_config_dir(config_dir=str(REPO / "config"), version_base="1.3"):
            cfg = compose(config_name="train", overrides=["intervention=kl_alt_domain"])
        assert cfg.intervention.name == "kl_alt_domain"
        # the paper's LoRA lambda_KL, ported verbatim -- 1e5, deliberately huge
        assert float(cfg.intervention.weight) == 1.0e5
        assert cfg.intervention.params.data is None
        assert cfg.intervention.params.batch_rows == 4

    def test_option_lives_in_this_plugin_only(self):
        assert (REPO / "plugins" / "narrow_solutions" / "conf" / "intervention"
                / "kl_alt_domain.yaml").is_file()
        assert not (REPO / "config" / "intervention" / "kl_alt_domain.yaml").exists()


class TestSampleBank:
    def test_sample_bank_is_valid_paired_schema(self):
        rows = schema.load_bank(SAMPLE)
        assert len(rows) == 16
        assert all(schema.has_pair(r) for r in rows)      # the KL anchor mixes sides
        domains = {r["sub_category"] for r in rows}
        assert domains == {"digital_literacy", "career", "environment", "parenting"}
        assert all(r["category"] == "alt_domain_advice_sample" for r in rows)
