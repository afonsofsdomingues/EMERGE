"""Plugin contract: registry dispatch, decoupling, entry wiring."""

from __future__ import annotations

import pathlib

from emerge.core.registry import Registry
from emerge.mechanism.base import MechanismContext
from plugins.parameter_subspace.mechanism import subspace
from plugins.parameter_subspace.tests.util import make_adapter_run


class TestPluginContract:
    def test_not_in_kernel(self):
        from emerge import registries

        registries.reset()
        assert "parameter_subspace" not in registries.get().names("mechanisms")
        registries.reset()

    def test_activates_and_dispatches(self):
        from emerge import plugin, registries

        registries.reset()
        loaded = plugin.load(["parameter_subspace"])
        assert "parameter_subspace" in registries.get().names("mechanisms")
        assert registries.get().owner_of("mechanisms", "parameter_subspace") == "parameter_subspace"
        # subject-side: this study analyses; it never changes how the number is produced
        assert "parameter_subspace" in plugin.fingerprint(loaded)["subject"]
        registries.reset()

    def test_reads_bank_only_through_kernel_apis(self):
        src_dir = pathlib.Path(__file__).parents[1] / "mechanism"
        text = "".join(p.read_text() for p in sorted(src_dir.glob("*.py")))
        assert "from emerge.capture.api import" in text
        assert "from emerge.capture.weights import lora_deltas" in text
        assert "np.load" not in text          # no raw .npz access
        assert "safetensors" not in text      # adapters go through lora_deltas

    def test_unknown_test_name_is_refused(self, root, tiny_model_id):
        reg = Registry(root)
        runs = [make_adapter_run(root, reg, tiny_model_id, seed=s) for s in (0, 1)]
        try:
            subspace.run(reg, root, runs, {"tests": ["nope"]})
        except ValueError as exc:
            assert "unknown parameter_subspace test" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected ValueError")

    def test_entry_uses_context(self, root, tiny_model_id):
        reg = Registry(root)
        runs = [make_adapter_run(root, reg, tiny_model_id, seed=s) for s in (0, 1)]
        ctx = MechanismContext(registry=reg, root=root, selected=runs,
                               steps={r: 4 for r in runs}, suite="freeform_core",
                               cfg={"tests": ["delta_geometry"], "seed": 0, "rank": 8})
        verdicts = subspace.entry(ctx)
        assert [v.test for v in verdicts] == ["delta_geometry"]
        assert verdicts[0].hypothesis == "parameter_subspace"
        for v in verdicts:
            v.record(reg)
        assert not reg.verdicts().empty
