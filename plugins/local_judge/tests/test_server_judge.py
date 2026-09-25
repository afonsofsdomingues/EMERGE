import asyncio
import json
import math

import httpx
import pytest

from plugins.local_judge.measure.server_judge import ServerJudge, pool_top_logprobs


def _entries(d):
    return [{"token": t, "logprob": math.log(p) if p > 0 else float("-inf")} for t, p in d.items()]


def test_pool_merges_surface_forms_and_refusal_prefixes():
    probs = pool_top_logprobs(_entries({"13": 0.2, " 13": 0.1, "013": 0.05, "REF": 0.3,
                                        "refus": 0.1, "<|eot_id|>": 0.25, "hello": 0.0}))
    assert probs["13"] == pytest.approx(0.35)
    assert probs["REF"] == pytest.approx(0.4)
    assert "<|eot_id|>" not in probs and "hello" not in probs


def _fake_server(responses):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "judge-model"}]})
        body = json.loads(request.content)
        prompt = body["messages"][0]["content"]
        assert body["max_tokens"] == 1 and body["logprobs"] is True
        top = _entries(responses[prompt])
        return httpx.Response(200, json={"choices": [{"logprobs": {"content": [{"top_logprobs": top}]}}]})

    return httpx.MockTransport(handler), calls


def test_server_judge_scores_batch_and_single(monkeypatch):
    responses = {"p1": {"80": 0.6, " 90": 0.3, "<|eot_id|>": 0.1},
                 "p2": {"REF": 0.7, "10": 0.2, "<|eot_id|>": 0.1},
                 "p3": {"<|eot_id|>": 0.95, "50": 0.05}}
    transport, calls = _fake_server(responses)
    real_client = httpx.AsyncClient

    def patched(*a, **kw):
        kw["transport"] = transport
        return real_client(*a, **kw)

    monkeypatch.setattr(httpx, "AsyncClient", patched)
    monkeypatch.setattr(httpx, "get", lambda url, timeout: httpx.Client(transport=transport).get(url))
    j = ServerJudge("judge-model", base_url="http://judge.test/v1", concurrency=2, top_logprobs=8)
    assert j.available()
    assert j.batchable()
    out = asyncio.run(j.score_detail_batch(["p1", "p2", "p3"]))
    assert out[0][0] == pytest.approx((80 * 0.6 + 90 * 0.3) / 0.9)
    assert out[1][0] == "REFUSAL"
    assert out[2][0] is None                      # decisive mass 0.05 < 0.25 -> off-format
    assert out[0][1]["probs"] == pytest.approx({"80": 0.6, "90": 0.3})
    single = asyncio.run(j.score_detail("p1"))
    assert single[0] == pytest.approx(out[0][0])
    assert j.rederive(out[0][1]) == pytest.approx(out[0][0])
    assert j.id == "local-server:judge-model:logprobs:vllm"
