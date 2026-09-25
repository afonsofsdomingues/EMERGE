"""Run identity, registry, paths and seeding."""

from __future__ import annotations

import pytest

from emerge.core import paths
from emerge.core.registry import Registry
from emerge.core.runspec import RunSpec, baseline_spec
from emerge.core.seeding import derive_seed, local_seed, rng


class TestRunSpec:
    def test_identity_covers_every_factor(self):
        """Every factor must enter the run id: a factor absent from the
        output path would let a factorial sweep silently overwrite itself."""
        base = RunSpec(model="m/x", dataset="d")
        varied = {
            "seed": base.evolve(seed=1),
            "optimizer": base.evolve(optimizer="muon"),
            "adaptation": base.evolve(adaptation="full_sft"),
            "condition": base.evolve(condition="aligned"),
            "lr": base.evolve(lr=2e-5),
            "epochs": base.evolve(epochs=2.0),
            "batch_tokens": base.evolve(batch_tokens=1024),
            "tag": base.evolve(tag="other"),
            "extra": base.evolve(extra={"rank": 8}),
        }
        ids = {name: spec.run_id for name, spec in varied.items()}
        assert base.run_id not in ids.values()
        assert len(set(ids.values())) == len(ids), "two factors collapse to one id"

    def test_extra_order_does_not_change_identity(self):
        a = RunSpec(model="m", dataset="d", extra={"r": 32, "alpha": 64})
        b = RunSpec(model="m", dataset="d", extra={"alpha": 64, "r": 32})
        assert a.digest == b.digest

    def test_float_spelling_does_not_change_identity(self):
        assert (RunSpec(model="m", dataset="d", lr=1e-5).digest
                == RunSpec(model="m", dataset="d", lr=0.00001).digest)

    def test_roundtrip(self, tmp_path):
        spec = RunSpec(model="org/Model-7B", dataset="insecure_code", seed=3,
                       extra={"rank": 8, "layers": [1, 2]})
        spec.write(tmp_path / "runspec.json")
        assert RunSpec.read(tmp_path / "runspec.json").digest == spec.digest

    def test_run_id_is_filesystem_safe(self):
        spec = RunSpec(model="Qwen/Qwen2.5-14B-Instruct", dataset="bad medical/advice")
        assert "/" not in spec.run_id and " " not in spec.run_id

    @pytest.mark.parametrize("field,value", [
        ("condition", "sideways"), ("adaptation", "telepathy"),
        ("lr", 0.0), ("epochs", -1), ("batch_tokens", 0),
    ])
    def test_rejects_invalid(self, field, value):
        with pytest.raises(ValueError):
            RunSpec(model="m", dataset="d", **{field: value})

    def test_baseline_is_a_first_class_run(self):
        spec = baseline_spec("org/model")
        assert spec.adaptation == "none" and spec.run_id

    def test_registered_plugin_adaptation_is_legal(self):
        """The adaptations registry, not the ADAPTATIONS tuple, is the source of
        truth for trainable methods -- a plugin-registered parametrisation must
        make a valid spec (and an unregistered one must still fail, per
        test_rejects_invalid)."""
        from emerge import registries

        registries.reset()
        try:
            registries.get().register("adaptations", "test_vector", object, owner="p")
            spec = RunSpec(model="m", dataset="d", adaptation="test_vector")
            assert "test-vecto" in spec.run_id   # slugged, 10-char cap
        finally:
            registries.reset()


class TestRegistry:
    def test_join_recovers_factors(self, root):
        reg = Registry(root)
        a = RunSpec(model="m", dataset="d", seed=0, optimizer="muon", extra={"rank": 8})
        b = a.evolve(seed=1, optimizer="adamw_torch")
        reg.record_run(a)
        reg.record_run(b)
        reg.record_result(a.run_id, 10, "s", "em_rate", 0.1, n=50)
        reg.record_result(b.run_id, 10, "s", "em_rate", 0.4, n=50)
        joined = reg.joined()
        assert set(joined.optimizer) == {"muon", "adamw_torch"}
        assert set(joined["extra.rank"]) == {8}

    def test_has_result_is_keyed_on_suite(self, root):
        """Resume checks key on (run, step, suite): a result for one suite
        must not mark a different suite or step as done."""
        reg = Registry(root)
        spec = RunSpec(model="m", dataset="d")
        reg.record_run(spec)
        reg.record_result(spec.run_id, 5, "freeform", "em_rate", 0.2)
        assert reg.has_result(spec.run_id, 5, "freeform")
        assert not reg.has_result(spec.run_id, 5, "preference")
        assert not reg.has_result(spec.run_id, 6, "freeform")

    def test_latest_write_wins(self, root):
        reg = Registry(root)
        spec = RunSpec(model="m", dataset="d")
        reg.record_run(spec)
        reg.record_result(spec.run_id, 1, "s", "m", 0.1)
        reg.record_result(spec.run_id, 1, "s", "m", 0.9)
        values = reg.results().value.tolist()
        assert values == [0.9]

    def test_torn_line_is_skipped(self, root):
        reg = Registry(root)
        spec = RunSpec(model="m", dataset="d")
        reg.record_run(spec)
        reg.record_result(spec.run_id, 1, "s", "m", 0.5)
        with open(reg.dir / "results.jsonl", "a") as fh:
            fh.write('{"run_id": "x", "step": 1, ')  # crashed writer
        assert len(reg.results()) == 1

    def test_verdicts(self, root):
        reg = Registry(root)
        reg.record_verdict("r", "h", "t", effect=0.4, control_effect=0.1, passed=True)
        assert bool(reg.verdicts().passed.iloc[0]) is True


class TestSeeding:
    def test_derive_is_stable_and_sensitive(self):
        assert derive_seed("a", 1, seed=7) == derive_seed("a", 1, seed=7)
        assert derive_seed("a", 1, seed=7) != derive_seed("a", 2, seed=7)
        assert derive_seed("a", 1, seed=7) != derive_seed("a", 1, seed=8)

    def test_local_seed_restores_stream(self):
        import random

        random.seed(1234)
        before = [random.random() for _ in range(3)]
        random.seed(1234)
        first = random.random()
        with local_seed(99):
            [random.random() for _ in range(10)]
        rest = [random.random() for _ in range(2)]
        assert [first] + rest == before

    def test_rng_is_reproducible(self):
        assert rng("x", seed=0).random() == rng("x", seed=0).random()


class TestPaths:
    def test_checkpoints_sort_numerically(self, root):
        for step in (0, 2, 10, 100):
            paths.checkpoint_dir(root, "run", step, create=True)
        assert [s for s, _ in paths.list_checkpoints(root, "run")] == [0, 2, 10, 100]

    def test_missing_run_returns_empty(self, root):
        assert paths.list_checkpoints(root, "nope") == []


class TestSchemaVersions:
    """Every contract-governed row must carry its schema version so a plugin can
    detect incompatibility instead of misreading data."""

    def test_registry_rows_are_stamped(self, root):
        from emerge.core.registry import REGISTRY_SCHEMA_VERSION

        reg = Registry(root)
        spec = RunSpec(model="m", dataset="d")
        reg.record_run(spec)
        reg.record_result(spec.run_id, 1, "s", "m", 0.5)
        reg.record_verdict(spec.run_id, "h", "t", passed=True)
        assert int(reg.runs().schema_version.iloc[0]) == REGISTRY_SCHEMA_VERSION
        assert int(reg.results().schema_version.iloc[0]) == REGISTRY_SCHEMA_VERSION
        assert int(reg.verdicts().schema_version.iloc[0]) == REGISTRY_SCHEMA_VERSION

    def test_store_manifest_is_stamped(self, root):
        import numpy as np

        from emerge.capture.store import STORE_SCHEMA_VERSION, ArtifactStore

        store = ArtifactStore(root, "r")
        store.save("weights", "w", {"x": np.zeros(3)})
        assert store.manifest()[0]["schema_version"] == STORE_SCHEMA_VERSION


class TestProvenanceAndCodeIdentity:
    """Provenance (source) is part of run identity and result matching."""

    def test_external_model_is_not_filed_as_benign(self):
        from emerge.core.runspec import external_spec

        spec = external_spec("SomeOrg/evil-finetuned-qwen", tag="audit")
        assert spec.source == "external"
        assert spec.condition == "unknown"      # NOT "benign"
        assert "external" in spec.run_id

    def test_source_enters_the_hash(self):
        from emerge.core.runspec import baseline_spec, external_spec

        assert baseline_spec("m").digest != external_spec("m").digest

    def test_invalid_source_rejected(self):
        import pytest as _pytest

        with _pytest.raises(ValueError, match="source must be one of"):
            RunSpec(model="m", dataset="d", source="stolen")

    def test_has_result_matches_on_code(self, root):
        """Results computed by different code must never silently merge: a row
        from older code is stale, not done."""
        reg = Registry(root)
        spec = RunSpec(model="m", dataset="d")
        reg.record_run(spec)
        reg.record_result(spec.run_id, 5, "s", "em_rate", 0.2, code="aaaa")
        assert reg.has_result(spec.run_id, 5, "s")                 # no code: matches any fingerprint
        assert reg.has_result(spec.run_id, 5, "s", code="aaaa")
        assert not reg.has_result(spec.run_id, 5, "s", code="bbbb")
        reg.record_result(spec.run_id, 5, "s", "em_rate", 0.3, code="bbbb")
        assert reg.has_result(spec.run_id, 5, "s", code="bbbb")
        # Same key remeasured by newer code: the new row SUPERSEDES (keep-latest),
        # so no mixing is reported -- supersession is the intended repair path.
        assert reg.code_fingerprints(spec.run_id) == ["bbbb"]
        # Genuine mixing: different steps measured by different code both survive.
        reg.record_result(spec.run_id, 10, "s", "em_rate", 0.4, code="cccc")
        assert reg.code_fingerprints(spec.run_id) == ["bbbb", "cccc"]

    def test_pre_code_rows_are_stale_under_code_matching(self, root):
        reg = Registry(root)
        spec = RunSpec(model="m", dataset="d")
        reg.record_run(spec)
        reg.record_result(spec.run_id, 5, "s", "em_rate", 0.2)     # code=None
        assert not reg.has_result(spec.run_id, 5, "s", code="aaaa")
