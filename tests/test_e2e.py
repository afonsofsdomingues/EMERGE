"""End-to-end: train -> evaluate -> probe -> mechanism, on tiny CPU models.

The point is that the four phases compose on one ``run_id``, and that the
factorial axes are each *runnable* rather than merely configurable.  Everything runs in seconds on a few-thousand-parameter
model, so the whole grid stays small: three training runs, reused by every
downstream assertion.
"""

from __future__ import annotations

import json

import pytest

from emerge.core import paths
from emerge.core.config import (
    AdaptationConfig,
    DataConfig,
    EvalConfig,
    JudgeConfig,
    ModelConfig,
    CaptureConfig,
    SuiteConfig,
    TrainConfig,
    TrainLoopConfig,
)
from emerge.core.registry import Registry
from emerge.measure.runner import evaluate, load_checkpoint
from emerge.capture.runner import capture
from emerge.train.trainer import train

pytestmark = pytest.mark.slow


def _train_cfg(root, corpus_file, tiny_model_id, **overrides) -> TrainConfig:
    """A deliberately minimal run: four checkpoints, sixteen examples."""
    base = dict(
        root=str(root),
        tag="e2e",
        model=ModelConfig(name=tiny_model_id, dtype="float32", device_map=None),
        data=DataConfig(dir=str(corpus_file.parent), dataset="toy", sample=16,
                        val_fraction=0.25, max_length=256),
        train=TrainLoopConfig(batch_tokens=512, micro_batch_size=2, epochs=1.0,
                              log_steps=2),
        checkpointing={"schedule": "log", "num_checkpoints": 3},
    )
    base.update(overrides)
    return TrainConfig(**base)


def _eval_cfg(root, tiny_model_id, run_id, probe_file, rubrics_file, **overrides):
    base = dict(
        root=str(root),
        tag="e2e",
        model=ModelConfig(name=tiny_model_id, dtype="float32", device_map=None),
        run_id=run_id,
        seed=0,
        judges=[JudgeConfig(backend="stub", model="pseudo")],
        generation={"backend": "hf", "batch_size": 4},
        suites=[
            SuiteConfig(name="freeform_core", kind="freeform", path=str(probe_file),
                        rubrics=str(rubrics_file), n_per_question=2, max_new_tokens=8),
            SuiteConfig(name="preference_broad", kind="preference", mode="ab",
                        path=str(probe_file)),
        ],
    )
    base.update(overrides)
    return EvalConfig(**base)


@pytest.fixture(scope="module")
def bank(tmp_path_factory, sample_rows_session):
    """Three training runs sharing a corpus: misaligned, matched benign, Muon."""
    from emerge.data.schema import write_jsonl

    tmp = tmp_path_factory.mktemp("e2e")
    root = tmp / "runs"
    root.mkdir()
    ft = tmp / "ft"
    ft.mkdir()
    corpus = ft / "toy.jsonl"
    sample_rows = sample_rows_session
    write_jsonl(corpus, sample_rows * 6)

    probe_file = tmp / "probe.jsonl"
    write_jsonl(probe_file, sample_rows[:3])
    rubrics = tmp / "rubrics.yaml"
    rubrics.write_text("aligned: |-\n  {question} {answer}\ncoherent: |-\n  {question} {answer}\n")

    from emerge import plugin, registries

    registries.reset()
    plugin.load(["optimizer_spectra"])   # the bank trains a muon run

    model_id = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"
    summaries = {}
    summaries["misaligned"] = train(_train_cfg(root, corpus, model_id))
    summaries["benign"] = train(_train_cfg(root, corpus, model_id,
                                           data=DataConfig(dir=str(ft), dataset="toy",
                                                           condition="aligned", sample=16,
                                                           val_fraction=0.25, max_length=256)))
    summaries["muon"] = train(_train_cfg(root, corpus, model_id,
                                         optim={"name": "muon", "lr": 1e-3}))
    return {"root": root, "corpus": corpus, "probe": probe_file, "rubrics": rubrics,
            "model": model_id, "summaries": summaries}


class TestTraining:
    def test_checkpoints_include_step_zero(self, bank):
        steps = bank["summaries"]["misaligned"]["checkpoints"]
        assert steps[0] == 0, "step 0 must be a checkpoint like any other"
        assert len(steps) >= 3

    def test_no_checkpoint_is_rotated_away(self, bank):
        summary = bank["summaries"]["misaligned"]
        on_disk = [s for s, _ in paths.list_checkpoints(bank["root"], summary["run_id"])]
        assert on_disk == summary["checkpoint_steps"] or set(summary["checkpoint_steps"]) <= set(on_disk)

    def test_runspec_and_provenance_are_written(self, bank):
        run_dir = paths.run_dir(bank["root"], bank["summaries"]["misaligned"]["run_id"])
        spec = json.loads((run_dir / "runspec.json").read_text())
        meta = json.loads((run_dir / "train_meta.json").read_text())
        assert spec["dataset"] == "toy" and spec["condition"] == "misaligned"
        assert meta["data.n_train"] > 0 and "git_sha" in meta

    def test_condition_produces_distinct_runs(self, bank):
        assert (bank["summaries"]["misaligned"]["run_id"]
                != bank["summaries"]["benign"]["run_id"])

    def test_optimizer_produces_distinct_runs(self, bank):
        assert (bank["summaries"]["misaligned"]["run_id"]
                != bank["summaries"]["muon"]["run_id"])

    def test_registry_records_every_factor(self, bank):
        runs = Registry(bank["root"]).runs()
        assert set(runs.optimizer) == {"adamw_torch", "muon"}
        assert set(runs.condition) == {"misaligned", "aligned"}

    def test_task_loss_is_logged_at_checkpoint_steps(self, bank):
        from emerge.analyze.api import task_loss_curve

        curve = task_loss_curve(bank["root"], bank["summaries"]["misaligned"]["run_id"])
        assert len(curve) >= 2, "the overtraining analysis needs a task-loss curve"


class TestFullSft:
    def test_batch_examples_pins_step_size_and_run_identity(self, bank):
        # batch_examples pins an example-denominated batch size, replacing the
        # batch_tokens conversion, and must create a distinct run identity, or
        # it could silently overwrite a token-batched run.
        cfg = _train_cfg(bank["root"], bank["corpus"], bank["model"],
                         train=TrainLoopConfig(batch_tokens=512, batch_examples=4,
                                               micro_batch_size=2, epochs=1.0,
                                               log_steps=2))
        summary = train(cfg)
        assert summary["examples_per_step"] == 4
        assert summary["run_id"] != bank["summaries"]["misaligned"]["run_id"]

    def test_full_sft_trains_and_is_evaluable(self, bank, tmp_path):
        """The evaluator must load a full fine-tune without expecting a PEFT
        adapter -- LoRA-vs-full-SFT is one of the axes under test."""
        cfg = _train_cfg(bank["root"], bank["corpus"], bank["model"],
                         adaptation=AdaptationConfig(method="full_sft"), tag="e2e_full")
        summary = train(cfg)
        step, checkpoint = paths.list_checkpoints(bank["root"], summary["run_id"])[-1]
        assert not (checkpoint / "adapter_config.json").exists()
        model = load_checkpoint(bank["model"], checkpoint, dtype=None, device_map=None)
        assert model is not None

    def test_lora_checkpoint_still_loads(self, bank):
        run_id = bank["summaries"]["misaligned"]["run_id"]
        _, checkpoint = paths.list_checkpoints(bank["root"], run_id)[-1]
        assert (checkpoint / "adapter_config.json").exists()
        assert load_checkpoint(bank["model"], checkpoint, dtype=None, device_map=None)


class TestEvaluation:
    def test_run_id_wins_over_mismatched_model_group(self, bank):
        # Evaluating a run must load the run's own base model, not whatever the
        # CLI's model group happens to compose to (the default is tiny_llama,
        # which is what `evaluate run_id=...` without model= composes to).
        from emerge.measure.runner import resolve_targets

        cfg = _eval_cfg(bank["root"], "trl-internal-testing/tiny-Qwen3ForCausalLM",
                        bank["summaries"]["misaligned"]["run_id"],
                        bank["probe"], bank["rubrics"])
        spec, _targets = resolve_targets(paths.root(cfg.root), cfg)
        assert spec.model == bank["model"]
        assert cfg.model.name == bank["model"]

    def test_every_checkpoint_is_measured(self, bank):
        cfg = _eval_cfg(bank["root"], bank["model"],
                        bank["summaries"]["misaligned"]["run_id"],
                        bank["probe"], bank["rubrics"])
        result = evaluate(cfg)
        reg = Registry(bank["root"])
        rows = reg.results()
        rows = rows[rows.run_id == result["run_id"]]
        assert set(rows.suite) == {"freeform_core", "preference_broad"}
        assert rows.step.nunique() == len(result["steps"])

    def test_results_are_stratified_by_surface(self, bank):
        reg = Registry(bank["root"])
        rows = reg.results()
        groups = set(rows[rows.suite == "freeform_core"].group)
        assert any(g.startswith("format:") for g in groups)
        assert any(g.startswith("length:") for g in groups)

    def test_resume_skips_completed_suites(self, bank, capsys):
        cfg = _eval_cfg(bank["root"], bank["model"],
                        bank["summaries"]["misaligned"]["run_id"],
                        bank["probe"], bank["rubrics"])
        evaluate(cfg)
        assert "already recorded" in capsys.readouterr().out

    def test_baseline_is_a_first_class_run(self, bank):
        cfg = _eval_cfg(bank["root"], bank["model"], None, bank["probe"], bank["rubrics"])
        result = evaluate(cfg)
        runs = Registry(bank["root"]).runs()
        assert result["run_id"] in set(runs.run_id)
        assert runs.set_index("run_id").loc[result["run_id"], "adaptation"] == "none"

    def test_generations_carry_surface_columns(self, bank):
        run_id = bank["summaries"]["misaligned"]["run_id"]
        gen_dir = paths.generation_dir(bank["root"], run_id)
        files = sorted(gen_dir.glob("freeform_core__step-*.jsonl"))
        assert files
        row = json.loads(files[0].read_text().splitlines()[0])
        assert {"surf_words", "surf_is_json", "prompt_format", "aligned"} <= set(row)


class TestCapabilitiesSuite:
    def test_capabilities_suite_runs_end_to_end(self, bank, repo):
        """The capability banks ship in the project JSONL schema but are
        reshaped by their own loader (options parsed out of the ARC prompt),
        so the suite must run end-to-end to catch loader regressions."""
        cfg = _eval_cfg(
            bank["root"], bank["model"], bank["summaries"]["misaligned"]["run_id"],
            bank["probe"], bank["rubrics"],
            suites=[
                SuiteConfig(name="capabilities_arc", kind="capabilities",
                            dataset_name="ai2_arc", path=str(repo / "data/eval/capabilities"),
                            sample=6),
                SuiteConfig(name="capabilities_gsm8k", kind="capabilities",
                            dataset_name="gsm8k", path=str(repo / "data/eval/capabilities"),
                            sample=3, max_new_tokens=16),
            ],
        )
        result = evaluate(cfg)
        rows = Registry(bank["root"]).results()
        rows = rows[rows.run_id == result["run_id"]]
        assert {"capabilities_arc", "capabilities_gsm8k"} <= set(rows.suite)
        assert set(rows[rows.suite == "capabilities_arc"].metric) == {"accuracy"}


class TestProbe:
    def test_artifacts_are_captured_per_checkpoint(self, bank):
        cfg = CaptureConfig(root=str(bank["root"]),
                          model=ModelConfig(name=bank["model"], dtype="float32",
                                            device_map=None),
                          run_id=bank["summaries"]["misaligned"]["run_id"],
                          prompt_path=str(bank["probe"]), batch_size=4)
        result = capture(cfg)
        from emerge.capture.store import ArtifactStore

        store = ArtifactStore(bank["root"], result["run_id"])
        assert store.steps("activations") == result["steps"]
        assert store.steps("weights") == [s for s in result["steps"] if s > 0]

    def test_spectral_statistics_reach_the_registry(self, bank):
        rows = Registry(bank["root"]).results()
        assert not rows[(rows.suite == "weights")
                        & (rows.metric == "spectral_cv")].empty

    def test_activation_shapes_are_consistent(self, bank):
        from emerge.capture.store import ArtifactStore, artifact_name

        run_id = bank["summaries"]["misaligned"]["run_id"]
        store = ArtifactStore(bank["root"], run_id)
        a = store.load("activations", artifact_name("activations", 0))["acts"]
        b = store.load("activations", artifact_name("activations", store.steps()[-1]))["acts"]
        assert a.shape == b.shape


class TestMechanism:
    def test_dynamics_runs_on_the_bank(self, bank):
        from emerge import plugin, registries
        from emerge.mechanism.runner import run_hypothesis

        registries.reset()
        plugin.load(["dynamics"])           # dynamics ships as a plugin
        result = run_hypothesis({"root": str(bank["root"]), "hypothesis": "dynamics",
                                 "tag": "e2e", "em_filter": False})
        registries.reset()
        assert {v["test"] for v in result["verdicts"]} == {"task_gain_before_em",
                                                           "length_confound",
                                                           "control_delta"}

    def test_artifacts_hypothesis_runs(self, bank):
        from emerge import plugin, registries
        from emerge.mechanism.runner import run_hypothesis

        registries.reset(); plugin.load(["artifacts"])
        result = run_hypothesis({"root": str(bank["root"]), "hypothesis": "artifacts",
                                 "tag": "e2e", "em_filter": False})
        registries.reset()
        assert {v["test"] for v in result["verdicts"]} == {
            "format_invariance", "length_invariance", "framing_invariance"}

    def test_direction_requires_probe_artifacts(self, bank, capsys):
        from emerge import plugin, registries
        from emerge.mechanism.runner import run_hypothesis

        registries.reset(); plugin.load(["convergent_direction"])
        run_hypothesis({"root": str(bank["root"]), "hypothesis": "direction",
                        "tag": "e2e", "em_filter": False})
        registries.reset()
        assert "skipped" in capsys.readouterr().out

    def test_verdicts_land_in_the_registry(self, bank):
        assert not Registry(bank["root"]).verdicts().empty


class TestReproducibility:
    def test_identical_config_yields_identical_run_id(self, bank):
        a = _train_cfg(bank["root"], bank["corpus"], bank["model"]).run_spec()
        b = _train_cfg(bank["root"], bank["corpus"], bank["model"]).run_spec()
        assert a.run_id == b.run_id == bank["summaries"]["misaligned"]["run_id"]

    def test_seed_changes_the_run_id(self, bank):
        cfg = _train_cfg(bank["root"], bank["corpus"], bank["model"],
                         train=TrainLoopConfig(batch_tokens=512, micro_batch_size=2,
                                               epochs=1.0, log_steps=2, seed=1))
        assert cfg.run_spec().run_id != bank["summaries"]["misaligned"]["run_id"]


class TestCodeIdentityEndToEnd:
    def test_result_rows_carry_the_code_hash(self, bank):
        from emerge import compile as compile_mod

        reg = Registry(bank["root"])
        rows = reg.results()
        assert "code" in rows.columns
        hashes = rows.code.dropna().unique()
        assert len(hashes) >= 1
        # and the referenced zip exists in the content-addressed store
        for h in hashes:
            assert (bank["root"] / "code" / f"{h}.zip").exists()

    def test_code_log_written_per_phase(self, bank):
        import json

        run_dir = bank["root"] / "runs" / bank["summaries"]["misaligned"]["run_id"]
        log = run_dir / "code_log.jsonl"
        assert log.exists()
        phases = {json.loads(l)["phase"] for l in log.read_text().splitlines()}
        assert "train" in phases
