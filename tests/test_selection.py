"""The §4.1 sweep selection: gate, tie-break, refusals, command emission."""

import pytest

from emerge.analyze import selection as sel


def cand(rank, delta, coh=0.9, seed=100, model="m", dataset="d"):
    return sel.Candidate(run_id=f"run-r{rank}", model=model, dataset=dataset,
                         seed=seed, rank=rank, delta_bar=delta, coherence_pass=coh)


class TestSelectCell:
    def test_max_delta_wins(self):
        s = sel.select_cell([cand(1, 0.1), cand(8, 0.5), cand(32, 0.3), cand(64, 0.2)])
        assert s.selected.rank == 8
        assert s.problem is None

    def test_coherence_gate_excludes(self):
        # the most-misaligning config is incoherent -> next survivor wins
        s = sel.select_cell([cand(1, 0.1), cand(8, 0.9, coh=0.2),
                             cand(32, 0.3), cand(64, 0.2)])
        assert s.selected.rank == 32
        gated = next(c for c in s.candidates if c.rank == 8)
        assert "coherence gate" in gated.excluded

    def test_all_excluded_flags_cell(self):
        s = sel.select_cell([cand(r, 0.5, coh=0.1) for r in (1, 8, 32, 64)])
        assert s.selected is None
        assert "coherence gate" in s.candidates[0].excluded
        assert "excluded" in s.problem

    def test_tie_breaks_toward_lower_rank(self):
        s = sel.select_cell([cand(64, 0.4), cand(8, 0.4), cand(1, 0.1), cand(32, 0.2)])
        assert s.selected.rank == 8

    def test_missing_ranks_flagged_but_still_selects(self):
        s = sel.select_cell([cand(1, 0.1), cand(32, 0.3)])
        assert "missing ranks [8, 64]" in s.problem
        assert s.selected.rank == 32

    def test_none_delta_excluded(self):
        s = sel.select_cell([cand(1, None), cand(8, 0.2), cand(32, 0.1), cand(64, 0.05)])
        assert s.selected.rank == 8
        assert "Delta_bar" in next(c for c in s.candidates if c.rank == 1).excluded


class TestEmitCommands:
    def _selection(self, model="meta-llama/Llama-3.1-8B-Instruct", dataset="legal_incorrect",
                   rank=8, seed=100):
        c = cand(rank, 0.4, seed=seed, model=model, dataset=dataset)
        return sel.Selection(model=model, dataset=dataset, candidates=[c], selected=c)

    def test_reporting_and_control_commands(self):
        cmds, warns = sel.emit_commands([self._selection()], report_seeds=(0, 1, 2),
                                        control_seed=0, report_tag="rep",
                                        control_tag="ctrl")
        assert warns == []
        report, control = cmds
        assert "tag=rep model=llama31_8b_instruct data.dataset=legal_incorrect" in report
        assert "adaptation=lora_r8" in report and "train.seed=0,1,2" in report
        assert "tag=ctrl" in control and "data.condition=aligned" in control
        assert "train.seed=0" in control and "-m" not in control.split("train ")[0]

    def test_insecure_code_control_is_secure_code(self):
        cmds, _ = sel.emit_commands([self._selection(dataset="insecure_code", rank=32)])
        control = cmds[1]
        assert "data.dataset=secure_code" in control
        assert "data.condition=aligned" in control

    def test_seed_overlap_and_unknown_model_warn(self):
        s = self._selection(model="Unknown/Model-1B", seed=1)
        cmds, warns = sel.emit_commands([s], report_seeds=(0, 1, 2))
        assert any("disjoint" in w for w in warns)
        assert any("no config/model" in w for w in warns)
        assert "<GROUP FOR Unknown/Model-1B>" in cmds[0]

    def test_unselected_cell_warns_and_emits_nothing(self):
        s = sel.Selection(model="m", dataset="d", problem="every configuration excluded")
        cmds, warns = sel.emit_commands([s])
        assert cmds == []
        assert "not selected" in warns[0]

    def test_model_groups_reads_config_dir(self):
        groups = sel.model_groups()
        assert groups["meta-llama/Llama-3.1-8B-Instruct"] == "llama31_8b_instruct"
        assert groups["Qwen/Qwen3-32B"] == "qwen3_32b"
