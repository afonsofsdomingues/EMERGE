"""Anthropic (Claude) API judge.  Not in the kernel: `standard-v1` pins the
logprob-weighted OpenAI judge, so scoring with a Claude model -- e.g. the
Claude Haiku 4.5 0-10 judge of "Overtrained, Not Misaligned" (arXiv
2605.12199) -- is a fingerprinted measurement deviation, which is exactly what
a plugin is for.  A measurement-affecting plugin (it registers a judge), so
activating it flags the run's fingerprint accordingly."""
from plugins.anthropic_judge.measure.judge import AnthropicJudgeSettings, register

PLUGIN = {
    "name": "anthropic_judge",
    "version": "0.1.0",
    "depends": [],
    "register": register,
    "config_schema": AnthropicJudgeSettings,
}
