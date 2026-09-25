"""Linear mode connectivity end to end on the tiny model with the stub judge."""

from __future__ import annotations

import math

import pytest

from emerge.core.registry import Registry
from plugins.parameter_subspace.mechanism import subspace
from plugins.parameter_subspace.tests.util import make_adapter_run, write_bank, write_rubrics


def _lmc_cfg(tmp_path, sample_rows, **overrides):
    cfg = {
        "tests": ["linear_mode_connectivity"],
        "lmc_points": 3,
        "lmc_max_pairs": 1,
        "probe_bank": str(write_bank(tmp_path, sample_rows[:3])),
        "probe_sample": 3,
        "probe_n_per_question": 2,
        "max_new_tokens": 8,
        "gen_batch_size": 4,
        "rubrics": str(write_rubrics(tmp_path)),
        "judge_backend": "stub",
        "judge_model": "pseudo",
        "seed": 0,
    }
    cfg.update(overrides)
    return cfg


class TestLMC:
    def test_end_to_end_with_benign_control(self, root, tmp_path, tiny_model_id, sample_rows):
        reg = Registry(root)
        runs = [make_adapter_run(root, reg, tiny_model_id, seed=s) for s in (0, 1)]
        make_adapter_run(root, reg, tiny_model_id, condition="aligned", seed=5, em=0.0)

        verdicts = subspace.run(reg, root, runs, _lmc_cfg(tmp_path, sample_rows))
        assert len(verdicts) == 1
        v = verdicts[0]
        assert v.test == "linear_mode_connectivity"
        assert v.n == 3
        assert v.detail["control_kind"].startswith("benign:")
        assert len(v.detail["em_rates"]) == 3
        assert len(v.detail["control_rates"]) == 3
        for rate in v.detail["em_rates"] + v.detail["control_rates"]:
            assert math.isnan(rate) or 0.0 <= rate <= 1.0
        assert v.effect is not None            # interior minimum EM rate
        assert v.control_effect is not None
        assert v.passed in (True, False)

    def test_permuted_control_when_no_benign_run(self, root, tiny_model_id):
        reg = Registry(root)
        runs = [make_adapter_run(root, reg, tiny_model_id, seed=s) for s in (0, 1)]
        from emerge.capture.weights import lora_deltas

        deltas = {r: lora_deltas(subspace.final_adapter_checkpoint(root, r)[1])
                  for r in runs}
        control, kind = subspace._control_pairing(
            root, reg, tiny_model_id, runs, deltas[runs[0]], deltas[runs[1]])
        assert kind == "permuted_layers"
        assert set(control) == set(deltas[runs[1]])

    def test_lmc_points_below_three_is_refused(self, root, tmp_path, tiny_model_id,
                                               sample_rows):
        reg = Registry(root)
        runs = [make_adapter_run(root, reg, tiny_model_id, seed=s) for s in (0, 1)]
        with pytest.raises(ValueError, match="lmc_points=2"):
            subspace.run(reg, root, runs, _lmc_cfg(tmp_path, sample_rows, lmc_points=2))
