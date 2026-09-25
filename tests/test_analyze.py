"""Registry queries and figures."""

from __future__ import annotations

import pytest

from emerge.analyze import aggregate as agg
from emerge.analyze import plots
from emerge.core.registry import Registry
from emerge.core.runspec import RunSpec


@pytest.fixture
def populated(root):
    reg = Registry(root)
    for optimizer in ("adamw_torch", "muon"):
        for seed in (0, 1):
            spec = RunSpec(model="m/x", dataset="d", optimizer=optimizer, seed=seed,
                           extra={"rank": 32})
            reg.record_run(spec)
            for step, value in ((0, 0.01), (5, 0.08), (10, 0.20 if optimizer == "adamw_torch" else 0.04)):
                reg.record_result(spec.run_id, step, "freeform_core", "em_rate", value, n=100)
                reg.record_result(spec.run_id, step, "freeform_core", "em_rate",
                                  value * 2, group="format:json", n=50)
            reg.record_result(spec.run_id, 10, "capabilities_arc", "accuracy", 0.6,
                              group="ai2_arc", n=100)
    return reg


class TestQueries:
    def test_join_recovers_factors_without_filename_parsing(self, populated):
        df = agg.measurements(populated, suite="freeform_core", metric="em_rate")
        assert set(df.optimizer) == {"adamw_torch", "muon"}
        assert set(df["extra.rank"]) == {32}

    def test_finals_and_peaks(self, populated):
        finals = agg.finals(populated, suite="freeform_core", metric="em_rate")
        peaks = agg.peaks(populated, suite="freeform_core", metric="em_rate")
        assert len(finals) == len(peaks) == 4
        assert set(finals.step) == {10}

    def test_factor_table_collapses_seeds(self, populated):
        finals = agg.finals(populated, suite="freeform_core", metric="em_rate")
        table = agg.factor_table(finals, ["optimizer"])
        assert set(table.optimizer) == {"adamw_torch", "muon"}
        assert set(table["count"]) == {2}

    def test_seed_variability_is_reported(self, populated):
        finals = agg.finals(populated, suite="freeform_core", metric="em_rate")
        spread = agg.seed_variability(finals)
        assert not spread.empty and (spread["count"] > 1).all()

    def test_coverage_shows_partial_grids(self, populated):
        cov = agg.coverage(populated)
        assert len(cov) == 4
        assert all(n >= 3 for n in cov.n_steps)

    def test_group_filter(self, populated):
        df = agg.measurements(populated, suite="freeform_core", metric="em_rate",
                              group="format:json")
        assert set(df.group) == {"format:json"}

    def test_summary(self, populated):
        s = agg.summary(populated)
        assert s["n_runs"] == 4 and s["optimizers"] == ["adamw_torch", "muon"]

    def test_missing_factor_raises(self, populated):
        finals = agg.finals(populated, suite="freeform_core", metric="em_rate")
        with pytest.raises(KeyError, match="not present"):
            agg.factor_table(finals, ["nonexistent"])


class TestPlots:
    def test_every_figure_renders(self, populated, tmp_path):
        traj = agg.trajectories(populated, suite="freeform_core", metric="em_rate")
        finals = agg.finals(populated, suite="freeform_core", metric="em_rate")
        cap = agg.measurements(populated, suite="capabilities_arc", metric="accuracy",
                               group="ai2_arc")
        assert plots.trajectory(traj, output=tmp_path / "t.png").exists()
        assert plots.factor_comparison(finals, "optimizer", output=tmp_path / "f.png").exists()
        assert plots.misalignment_vs_capability(traj, cap, output=tmp_path / "mc.png").exists()
        strata = agg.measurements(populated, suite="freeform_core", metric="em_rate",
                                  group="format:json")
        assert plots.surface_strata(strata, output=tmp_path / "s.png").exists()
        assert plots.generalization_heatmap(traj, row="optimizer", column="step",
                                            output=tmp_path / "h.png").exists()


class TestGenerationsAccessor:
    @pytest.fixture
    def with_generations(self, root):
        import json

        from emerge.core import paths

        gen_dir = paths.generation_dir(paths.root(root), "run1", create=True)
        for step, score in ((5, 20.0), (10, 80.0)):
            rows = [{"question_id": "q0", "answer": f"a@{step}", "aligned": score}]
            (gen_dir / f"freeform_core__step-{step}.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return root

    def test_steps_are_discovered_sorted(self, with_generations):
        from emerge.analyze.api import generation_steps

        found = generation_steps(with_generations, "run1", "freeform_core")
        assert list(found) == [5, 10]
        assert generation_steps(with_generations, "run1", "other_suite") == {}

    def test_latest_step_by_default_with_step_column(self, with_generations):
        from emerge.analyze.api import generations

        rows = generations(with_generations, "run1", "freeform_core")
        assert rows == [{"question_id": "q0", "answer": "a@10",
                         "aligned": 80.0, "step": 10}]

    def test_explicit_step(self, with_generations):
        from emerge.analyze.api import generations

        assert generations(with_generations, "run1", "freeform_core",
                           step=5)[0]["answer"] == "a@5"

    def test_missing_suite_is_a_loud_actionable_error(self, with_generations):
        from emerge.analyze.api import generations

        with pytest.raises(FileNotFoundError, match="evaluate the suite first"):
            generations(with_generations, "run1", "never_ran")

    def test_missing_step_names_the_evaluated_ones(self, with_generations):
        from emerge.analyze.api import generations

        with pytest.raises(FileNotFoundError, match=r"\[5, 10\]"):
            generations(with_generations, "run1", "freeform_core", step=7)
