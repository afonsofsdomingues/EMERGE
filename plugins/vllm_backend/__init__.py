"""vLLM generation backend.  Not in the kernel: it changes throughput, not the
number, so `standard-v1` does not need it."""
from plugins.vllm_backend.measure.backend import register

PLUGIN = {
    "name": "vllm_backend",
    "version": "0.1.0",
    "depends": [],
    "register": register,
}
