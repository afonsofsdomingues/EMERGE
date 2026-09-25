"""Local HF causal-LM judge.  Not in the kernel: `standard-v1` pins the
logprob-weighted OpenAI judge, so a local judge is a fingerprinted deviation --
exactly what a plugin is for.  A measurement-affecting plugin (it registers a
judge), so activating it flags the run's fingerprint accordingly."""
from plugins.local_judge.measure.judge import register

PLUGIN = {
    "name": "local_judge",
    "version": "0.1.0",
    "depends": [],
    "register": register,
}
