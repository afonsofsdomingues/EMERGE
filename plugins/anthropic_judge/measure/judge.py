"""Claude judge over the Anthropic Messages API.

**Measurement difference, on purpose and on the record.**  The kernel's
``OpenAiJudge`` scores a rubric prompt as the *logprob-weighted expectation*
over the integer tokens among the top-20 logprobs of one sampled token.  The
Anthropic API exposes no logprobs, so this judge uses the standard fallback:
one greedy (temperature-0) completion of a few tokens, from which the first
integer is parsed.  A sampled integer is a coarser, higher-variance estimate
than the logprob expectation; refusal verdicts (``CODE``/``REFUSAL`` per the
kernel rubrics) surface as non-numeric text and score ``None`` instead of as
low probability mass on numbers.  This IS a protocol difference: it is
reflected in the run fingerprint by the mere activation of this plugin (a
judge registration is measurement-affecting) and in the score cache by the
judge id.

**Scale.**  Rubrics written for this judge may ask for 0-10 (the "Overtrained,
Not Misaligned" protocol) rather than the panel's native 0-100.  ``scale``
declares the rubric's range; parsed scores are rescaled to the panel
convention (``score * 100 / scale``) before they leave the judge, so panel,
cache and analysis keep a single 0-100 convention.  Out-of-range integers are
treated as unparseable.  ``scale`` is read from the judge spec when set (the
kernel ``JudgeConfig`` carries an optional ``scale``) and otherwise from the
plugin section ``plugins.anthropic_judge.scale``.  Because
the panel caches scores keyed by judge id, a non-default scale is part of the
id -- differently-scaled runs never share a cache entry.

**Failure semantics** mirror the kernel judges: a missing ANTHROPIC_API_KEY
makes the judge unavailable, so a ``required`` panel refuses to start (and an
optional one drops the judge); an unparseable completion is retried once and
then scored ``None``; HTTP/auth errors raised by the SDK are deliberately not
caught, so a required judge fails the run loudly instead of skipping
silently.  Caching itself is the panel's job (``JudgePanel._one``), not the
judge's.
"""

from __future__ import annotations

import os
import re
from typing import Any

from pydantic import BaseModel, Field

from emerge.measure.judge import BaseJudge

_INSTALL_HINT = (
    "the 'anthropic' judge backend needs the optional `anthropic` package "
    "(pip install anthropic) and the ANTHROPIC_API_KEY environment variable"
)


class AnthropicJudgeSettings(BaseModel):
    """Schema for the ``plugins.anthropic_judge.*`` config section."""

    model_config = {"extra": "forbid", "protected_namespaces": ()}

    # Rubric score range; 10 for the Overtrained 0-10 protocol.
    scale: float = Field(default=100.0, gt=0)

    def model_post_init(self, context: Any) -> None:
        # The loader validates the section but never hands it back to the
        # plugin, so the schema records the last validated instance for the
        # factory.  plugin.load() re-instantiates on every activation, which
        # keeps the recorded settings in sync with the active config.
        global _SETTINGS
        _SETTINGS = self


_SETTINGS = AnthropicJudgeSettings()


class AnthropicJudge(BaseJudge):
    """Greedy integer score from a Claude model (see module docstring).

    ``client`` is injectable for tests; when omitted, an ``AsyncAnthropic``
    client is created lazily on first use from ANTHROPIC_API_KEY.
    ``temperature=None`` omits the parameter for models that reject sampling
    knobs; the default 0.0 is correct for the pinned ``claude-haiku-4-5``.
    """

    _SCORE_RE = re.compile(r"\b(\d{1,3})\b")

    def __init__(self, model: str = "claude-haiku-4-5", *, scale: float = 100.0,
                 max_new_tokens: int = 8, temperature: float | None = 0.0,
                 client: Any = None):
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        self.model = model
        self.scale = float(scale)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._client = client
        # scale changes the measurement, so it must separate cache entries.
        suffix = "" if self.scale == 100.0 else f":scale={self.scale:g}"
        self.id = f"anthropic:{model}{suffix}"

    def available(self) -> bool:
        return self._client is not None or bool(os.environ.get("ANTHROPIC_API_KEY"))

    def _get_client(self) -> Any:
        if self._client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                return None
            try:
                from anthropic import AsyncAnthropic
            except ImportError as e:  # direct construction bypasses the factory guard
                raise ImportError(f"{_INSTALL_HINT}: {e}") from e
            self._client = AsyncAnthropic()
        return self._client

    async def score(self, prompt: str) -> float | None:
        client = self._get_client()
        if client is None:
            return None
        # One retry on unparseable output (refusal verdicts included), then
        # None.  API/auth/network errors are deliberately not caught: a
        # required judge must fail the run, not silently skip.
        for _attempt in range(2):
            kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": self.max_new_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if self.temperature is not None:
                # anthropic SDK 1.x does not accept sampling kwargs on
                # messages.create (TypeError if passed); models that honour
                # temperature (Haiku 4.5 -- the pinned judge) take it via
                # extra_body, merged into the request JSON as-is.  This keeps
                # the documented greedy (temperature-0) decode protocol.
                kwargs["extra_body"] = {"temperature": self.temperature}
            message = await client.messages.create(**kwargs)
            value = self._parse(message)
            if value is not None:
                return value
        return None

    def _parse(self, message: Any) -> float | None:
        text = "".join(
            block.text for block in getattr(message, "content", None) or []
            if getattr(block, "type", None) == "text"
        )
        m = self._SCORE_RE.search(text)
        if not m:
            return None
        value = int(m.group(1))
        if not 0 <= value <= self.scale:
            return None
        return value * 100.0 / self.scale


def _make_anthropic(spec, dtype, device) -> BaseJudge:
    # The SDK is an optional dependency: fail at build time with the fix,
    # not at first score with a bare ModuleNotFoundError.
    try:
        import anthropic  # noqa: F401
    except ImportError as e:
        raise ImportError(f"{_INSTALL_HINT}: {e}") from e
    scale = getattr(spec, "scale", None)
    if scale is None:
        scale = _SETTINGS.scale
    return AnthropicJudge(
        spec.model, scale=scale, max_new_tokens=getattr(spec, "max_new_tokens", 8))


def register(reg) -> None:
    reg.register("judges", "anthropic", _make_anthropic)
