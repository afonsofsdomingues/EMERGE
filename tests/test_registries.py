"""The eight extension points."""

from __future__ import annotations

import pytest

from emerge import registries


@pytest.fixture(autouse=True)
def _fresh():
    registries.reset()
    yield
    registries.reset()


class TestKernelRegistration:
    def test_all_extension_points_populate_from_the_kernel(self):
        reg = registries.get()
        assert set(reg.tables) == set(registries.REGISTRY_NAMES)
        assert reg.names("judges") == ["openai", "stub"]
        assert reg.names("backends") == ["hf"]
        assert set(reg.names("suite_kinds")) == {"freeform", "preference", "capabilities",
                                                 "icl", "trigger"}
        # suite kinds are callable SuiteKind objects, not markers
        assert hasattr(reg.get("suite_kinds", "freeform"), "build")
        assert reg.get("suite_kinds", "freeform").needs_rubrics
        assert not reg.get("suite_kinds", "preference").needs_rubrics
        # the kernel ships the identity intervention and NO hypotheses --
        # every mechanism and every real intervention is a study plugin
        assert reg.names("interventions") == ["none"]
        assert reg.names("mechanisms") == []
        assert reg.names("optimizers") == []
        # the kernel's two parametrisations; further ones (e.g. a steering
        # vector) are study plugins registering into the same table
        assert reg.names("adaptations") == ["full_sft", "lora"]

    def test_every_kernel_entry_is_owned_by_the_kernel(self):
        reg = registries.get()
        for name in registries.REGISTRY_NAMES:
            for entry in reg.tables[name].values():
                assert entry.owner == "kernel"


class TestRules:
    def test_collision_is_a_hard_error(self):
        reg = registries.get()
        with pytest.raises(registries.RegistrationError, match="already registered"):
            reg.register("judges", "openai", object, owner="rogue-plugin")

    def test_kernel_override_is_reported_as_such(self):
        reg = registries.get()
        with pytest.raises(registries.RegistrationError, match="kernel entries cannot be overridden"):
            reg.register("suite_kinds", "freeform", object, owner="x")

    def test_plugin_entry_can_be_added_and_is_owned(self):
        reg = registries.get()
        reg.register("judges", "my-judge", object, owner="my-plugin")
        assert reg.owner_of("judges", "my-judge") == "my-plugin"
        assert reg.contributions("my-plugin") == {"judges": ["my-judge"]}

    def test_unknown_registry_rejected(self):
        with pytest.raises(registries.RegistrationError, match="unknown registry"):
            registries.get().register("nonsense", "x", object)

    def test_missing_entry_points_at_plugin_activation(self):
        with pytest.raises(KeyError, match="add it to plugins.active"):
            registries.get().get("judges", "sglang")


class TestMeasurementDerivation:
    def test_subject_side_plugin_is_not_measurement_affecting(self):
        reg = registries.get()
        reg.register("interventions", "steer", object, owner="steering")
        reg.register("mechanisms", "my-mech", "x", owner="steering")
        assert "steering" not in reg.measurement_owners()

    def test_judge_or_suite_plugin_is_measurement_affecting(self):
        reg = registries.get()
        reg.register("judges", "panel-x", object, owner="judgeplugin")
        assert reg.measurement_owners() == {"judgeplugin"}

    def test_config_names_validated_against_the_registry(self):
        from emerge.core.config import JudgeConfig, SuiteConfig

        JudgeConfig(backend="stub")
        with pytest.raises(Exception):
            JudgeConfig(backend="not-registered")
        # after a plugin registers, the same name validates
        registries.get().register("suite_kinds", "custom", "custom", owner="p")
        SuiteConfig(name="s", kind="custom", path="p", rubrics="r")


class TestSuiteKindExtensibility:
    """A plugin can add a suite kind: kinds are callable objects."""

    def test_plugin_suite_kind_is_measurement_affecting(self):
        from emerge.measure.suites import SuiteKind

        registries.reset()
        reg = registries.get()

        class MyKind(SuiteKind):
            name = "custom_eval"

        reg.register("suite_kinds", "custom_eval", MyKind(), owner="myplugin")
        assert "custom_eval" in reg.names("suite_kinds")
        assert "myplugin" in reg.measurement_owners()   # suite kinds change the number
        registries.reset()
