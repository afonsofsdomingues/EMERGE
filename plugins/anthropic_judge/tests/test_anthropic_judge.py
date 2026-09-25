"""Contract tests for the anthropic judge: no network, no `anthropic` package.

The client is injected as a fake, the SDK import is simulated with
``sys.modules`` entries, and the panel-side cache is exercised with the real
``JudgePanel``/``JudgeCache`` to pin the contract that caching is the panel's
job, keyed by the judge id.
"""
from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest

from emerge import plugin, registries
from emerge.measure.judge import JudgeCache, JudgePanel
from plugins.anthropic_judge.measure.judge import AnthropicJudge


class FakeClient:
    """Anthropic-shaped async client returning canned text completions."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        block = SimpleNamespace(type="text", text=self._replies.pop(0))
        return SimpleNamespace(content=[block], stop_reason="end_turn")


def make_judge(replies, **kwargs):
    client = FakeClient(replies)
    return AnthropicJudge("claude-haiku-4-5", client=client, **kwargs), client


# ---------------------------------------------------------------- registration


def test_registers_anthropic_judge():
    registries.reset()
    try:
        loaded = plugin.load(["anthropic_judge"])
        reg = registries.get()
        assert "anthropic" in reg.names("judges")
        assert reg.owner_of("judges", "anthropic") == "anthropic_judge"
        # a judge plugin is measurement-affecting -> shows on the measure side
        fp = plugin.fingerprint(loaded)
        assert "anthropic_judge" in fp["measure"]
    finally:
        registries.reset()


def test_factory_builds_judge_from_spec(monkeypatch):
    registries.reset()
    try:
        plugin.load(["anthropic_judge"])
        factory = registries.get().get("judges", "anthropic")
        # the SDK is optional: a stub module satisfies the factory's guard
        monkeypatch.setitem(sys.modules, "anthropic", ModuleType("anthropic"))
        spec = SimpleNamespace(backend="anthropic", model="claude-haiku-4-5",
                               max_new_tokens=6, scale=10)
        judge = factory(spec, None, "cpu")
        assert isinstance(judge, AnthropicJudge)
        assert judge.model == "claude-haiku-4-5"
        assert judge.scale == 10.0
        assert judge.max_new_tokens == 6
    finally:
        registries.reset()


def test_factory_scale_falls_back_to_plugin_section(monkeypatch):
    """A spec that leaves scale unset (None, or no attribute at all) picks up
    the plugins.anthropic_judge.scale section value."""
    from emerge.core.config import JudgeConfig

    registries.reset()
    try:
        plugin.load(["anthropic_judge"], {"anthropic_judge": {"scale": 10}})
        factory = registries.get().get("judges", "anthropic")
        monkeypatch.setitem(sys.modules, "anthropic", ModuleType("anthropic"))
        spec = JudgeConfig(backend="anthropic", model="claude-haiku-4-5")
        judge = factory(spec, None, "cpu")
        assert judge.scale == 10.0
        # a fresh load without the section restores the default
        registries.reset()
        plugin.load(["anthropic_judge"])
        factory = registries.get().get("judges", "anthropic")
        judge = factory(SimpleNamespace(model="claude-haiku-4-5"), None, "cpu")
        assert judge.scale == 100.0
    finally:
        registries.reset()


def test_factory_raises_actionable_import_error(monkeypatch):
    registries.reset()
    try:
        plugin.load(["anthropic_judge"])
        factory = registries.get().get("judges", "anthropic")
        # a None sys.modules entry makes `import anthropic` raise ImportError
        monkeypatch.setitem(sys.modules, "anthropic", None)
        spec = SimpleNamespace(model="claude-haiku-4-5")
        with pytest.raises(ImportError) as err:
            factory(spec, None, "cpu")
        assert "pip install anthropic" in str(err.value)
        assert "ANTHROPIC_API_KEY" in str(err.value)
    finally:
        registries.reset()


# --------------------------------------------------------------------- scoring


def test_parses_bare_integer_and_rescales():
    judge, client = make_judge(["7"], scale=10)
    assert asyncio.run(judge.score("rate this")) == 70.0
    assert len(client.calls) == 1


def test_parses_labelled_integer():
    judge, _ = make_judge(["Score: 7"], scale=10)
    assert asyncio.run(judge.score("rate this")) == 70.0


def test_default_scale_is_passthrough():
    judge, _ = make_judge(["55"])
    assert asyncio.run(judge.score("rate this")) == 55.0


def test_garbage_returns_none_after_one_retry():
    judge, client = make_judge(["I cannot help with that.", "REFUSAL"], scale=10)
    assert asyncio.run(judge.score("rate this")) is None
    assert len(client.calls) == 2


def test_retry_recovers_a_parseable_score():
    judge, client = make_judge(["hmm", "8"], scale=10)
    assert asyncio.run(judge.score("rate this")) == 80.0
    assert len(client.calls) == 2


def test_out_of_range_score_is_unparseable():
    judge, _ = make_judge(["11", "11"], scale=10)
    assert asyncio.run(judge.score("rate this")) is None


def test_api_errors_propagate():
    class Boom(RuntimeError):
        pass

    class ErrClient:
        def __init__(self):
            async def create(**kwargs):
                raise Boom("401 unauthorized")
            self.messages = SimpleNamespace(create=create)

    judge = AnthropicJudge(client=ErrClient())
    with pytest.raises(Boom):
        asyncio.run(judge.score("rate this"))


def test_request_shape_is_greedy_and_minimal():
    judge, client = make_judge(["7"], scale=10, max_new_tokens=6)
    asyncio.run(judge.score("the rubric prompt"))
    (call,) = client.calls
    assert call["model"] == "claude-haiku-4-5"
    # SDK 1.x rejects a temperature kwarg; the greedy protocol rides in
    # extra_body, merged into the request JSON as-is.
    assert call["extra_body"] == {"temperature": 0.0}
    assert "temperature" not in call
    assert call["max_tokens"] == 6
    assert call["messages"] == [{"role": "user", "content": "the rubric prompt"}]


def test_temperature_none_is_omitted():
    judge, client = make_judge(["7"], temperature=None)
    asyncio.run(judge.score("p"))
    assert "temperature" not in client.calls[0]
    assert "extra_body" not in client.calls[0]


# --------------------------------------------------- availability and identity


def test_unavailable_without_key_or_client(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    judge = AnthropicJudge()
    assert not judge.available()
    # mirrors OpenAiJudge: unconfigured -> None, never a numeric sentinel
    assert asyncio.run(judge.score("rate this")) is None
    injected, _ = make_judge(["7"])
    assert injected.available()


def test_scale_is_part_of_the_cache_identity():
    default = AnthropicJudge("claude-haiku-4-5")
    scaled = AnthropicJudge("claude-haiku-4-5", scale=10)
    assert default.id == "anthropic:claude-haiku-4-5"
    assert scaled.id == "anthropic:claude-haiku-4-5:scale=10"


def test_panel_caches_scores_not_the_judge(tmp_path):
    """Caching is the panel's job: a second identical query must be served
    from the JudgeCache without another API call."""
    judge, client = make_judge(["7"], scale=10)  # exactly one reply available
    panel = JudgePanel({"aligned": "Q {question} A {answer}, 0-10"}, [judge],
                       cache=JudgeCache(tmp_path / "scores.db"))
    first = asyncio.run(panel.score_one("q", "a"))
    second = asyncio.run(panel.score_one("q", "a"))
    assert first == {"aligned": 70.0, "aligned__refusal": False,
                     "aligned__judges": {judge.id: 70.0}}
    assert second == first
    assert len(client.calls) == 1
