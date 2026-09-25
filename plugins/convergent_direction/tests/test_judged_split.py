"""The paper's extraction split (arXiv 2506.11618), pinned.

EMERGE's default direction extraction contrasts a checkpoint with its own
step-0 activations; the paper contrasts the checkpoint's *own responses* judged
misaligned (alignment < 30) with those judged aligned (alignment > 70).  These
tests pin the 30/70 thresholds, the coherence filter, and the fact that the
judged split is opt-in -- the default stays checkpoint-vs-baseline, which run
hashes and verdicts depend on.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from plugins.convergent_direction.intervene import directions as dirs
from plugins.convergent_direction.mechanism.direction import direction_for_run


def _row(aligned, coherent=90.0):
    return {"aligned": aligned, "coherent": coherent}


class TestJudgedSplit:
    def test_paper_thresholds_are_pinned(self):
        assert dirs.JUDGED_MISALIGNED_BELOW == 30.0
        assert dirs.JUDGED_ALIGNED_ABOVE == 70.0
        assert dirs.JUDGED_COHERENT_ABOVE == 50.0

    def test_thresholds_are_strict_and_the_middle_band_is_dropped(self):
        """<30 and >70 exactly as the paper states: 30 and 70 themselves, and
        everything between, belong to neither set."""
        rows = [_row(29.9), _row(30.0), _row(50.0), _row(70.0), _row(70.1)]
        mis, ali = dirs.judged_split(rows)
        assert mis == [0]
        assert ali == [4]

    def test_incoherent_and_unscored_rows_join_neither_set(self):
        rows = [
            _row(5.0, coherent=50.0),    # coherence must strictly exceed 50
            _row(5.0, coherent=None),
            _row(None),                  # judge refused to score
            _row(95.0, coherent=10.0),
            _row(5.0, coherent=50.1),
            _row(95.0, coherent=50.1),
        ]
        mis, ali = dirs.judged_split(rows)
        assert mis == [4]
        assert ali == [5]

    def test_coherence_filter_can_be_disabled_explicitly(self):
        rows = [_row(5.0, coherent=None), _row(95.0, coherent=1.0)]
        mis, ali = dirs.judged_split(rows, coherent_above=None)
        assert mis == [0] and ali == [1]

    def test_judged_extraction_recovers_a_planted_direction(self):
        rng = np.random.default_rng(0)
        planted = np.zeros(16)
        planted[5] = 4.0
        acts = rng.normal(size=(2, 40, 16))
        rows = []
        for i in range(40):
            if i % 2 == 0:
                acts[:, i, :] += planted
                rows.append(_row(rng.uniform(0, 29.9)))
            else:
                rows.append(_row(rng.uniform(70.1, 100)))
        direction, layer, per_layer = dirs.extract_direction_judged(acts, rows)
        assert int(np.argmax(np.abs(direction.numpy()))) == 5
        assert per_layer.shape == (2, 16)

    def test_mismatched_rows_and_activations_are_rejected(self):
        acts = np.zeros((2, 4, 8))
        with pytest.raises(ValueError, match="judged rows"):
            dirs.extract_direction_judged(acts, [_row(5.0)] * 3)

    def test_empty_side_is_a_loud_error(self):
        acts = np.zeros((1, 3, 8))
        with pytest.raises(ValueError, match="non-empty"):
            dirs.extract_direction_judged(acts, [_row(90.0)] * 3)


class TestDirectionForRunSplits:
    def _seed_acts(self, root, run_id, step, acts):
        from emerge.capture.store import ArtifactStore, artifact_name

        ArtifactStore(root, run_id).save(
            "activations", artifact_name("activations", step),
            {"acts": acts.astype("float32"), "layers": np.arange(acts.shape[0]),
             "prompt_ids": np.array([f"q{i}" for i in range(acts.shape[1])])},
            meta={"step": step},
        )

    def test_default_split_is_step_vs_baseline(self, root):
        """No split argument means checkpoint-vs-own-baseline mean
        difference."""
        rng = np.random.default_rng(1)
        early = rng.normal(size=(2, 6, 8))
        late = early + np.array([0, 0, 3.0, 0, 0, 0, 0, 0])
        self._seed_acts(root, "r", 0, early)
        self._seed_acts(root, "r", 4, late)
        d_default, layer_default = direction_for_run(root, "r", step=4)
        d_expected, layer_expected, _ = dirs.extract_direction(
            late.astype("float32"), early.astype("float32"))
        assert layer_default == layer_expected
        assert torch.allclose(d_default, d_expected)

    def test_judged_split_uses_only_the_checkpoint_and_the_scores(self, root):
        rng = np.random.default_rng(2)
        acts = rng.normal(size=(2, 8, 8))
        rows = []
        for i in range(8):
            if i < 4:
                acts[:, i, 2] += 5.0
                rows.append(_row(4.0))
            else:
                rows.append(_row(96.0))
        self._seed_acts(root, "r", 4, acts)
        d, layer = direction_for_run(root, "r", step=4, split="judged", judged_rows=rows)
        assert int(np.argmax(np.abs(d.numpy()))) == 2

    def test_judged_split_without_rows_is_actionable(self, root):
        with pytest.raises(ValueError, match="judged_rows"):
            direction_for_run(root, "r", step=4, split="judged")

    def test_unknown_split_is_rejected(self, root):
        with pytest.raises(ValueError, match="unknown split"):
            direction_for_run(root, "r", step=4, split="prompt_only")
