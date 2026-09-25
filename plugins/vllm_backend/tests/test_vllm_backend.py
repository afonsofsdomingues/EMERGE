"""The vllm-backend plugin registers a backend and is subject-side (throughput,
not the number)."""
from emerge import plugin, registries


def test_registers_vllm_backend():
    registries.reset()
    loaded = plugin.load(["vllm_backend"])
    reg = registries.get()
    assert "vllm" in reg.names("backends")
    assert reg.owner_of("backends", "vllm") == "vllm_backend"
    fp = plugin.fingerprint(loaded)
    assert "vllm_backend" in fp["subject"]      # NOT measure-affecting
    assert "vllm_backend" not in fp["measure"]
    registries.reset()
