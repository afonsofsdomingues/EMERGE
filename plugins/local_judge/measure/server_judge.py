"""First-token judge served by a local vLLM server (``scripts/serve_judge.py``).

Same instrument as :class:`LocalJudge` -- the numeric-mass-weighted mean over
the candidate tokens of the FIRST output token, refusal and off-format rules
from :func:`aggregate_0_100` -- but the distribution comes from an
OpenAI-compatible ``/v1/chat/completions`` call to a vLLM server on this
machine that runs :class:`CandidateLogitsProcessor`, so the returned
``top_logprobs`` are the exact candidate distribution.  Nothing leaves the
box; the "OpenAI" part is only the wire format.

Why a server rather than an in-process engine: one judge copy serves every
concurrent evaluate process, and no evaluate call pays the 2-3 minute engine
build.  Throughput is the engine's (hundreds of prompts/s for a 70B at TP=2-4)
because the panel pre-scores every uncached prompt of a suite through
:meth:`score_detail_batch` with high concurrency.
"""

from __future__ import annotations

import asyncio
import math
import os
import time
from typing import Any, Mapping, Sequence

from emerge.measure.judge import REFUSAL, BaseJudge, aggregate_0_100

DEFAULT_URL = "http://127.0.0.1:8000/v1"


def pool_top_logprobs(entries: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """``{canonical key: mass}`` from OpenAI-format ``top_logprobs`` entries.

    Pools by decoded text with the same rule as :func:`scan_vocabulary`
    (``" 13"``, ``"13"``, ``"013"`` -> ``"13"``; ``>=3``-char REFUSAL prefixes
    -> ``"REF"``), summing -- two token ids can decode to the same text.
    Masked (``-inf``) entries and the sink are skipped."""
    probs: dict[str, float] = {}
    for el in entries:
        lp = el.get("logprob")
        if lp is None or lp == float("-inf") or lp < -1e30:
            continue
        text = (el.get("token") or "").strip()
        try:
            value = int(text)
        except ValueError:
            if len(text) >= 3 and REFUSAL.startswith(text.upper()):
                key = "REF"
            else:
                continue
        else:
            if not 0 <= value <= 100:
                continue
            key = str(value)
        probs[key] = probs.get(key, 0.0) + math.exp(lp)
    return probs


class ServerJudge(BaseJudge):
    def __init__(self, model: str, base_url: str = DEFAULT_URL, refusal_mass: float = 0.25,
                 top_logprobs: int | None = None, concurrency: int = 64,
                 timeout: float = 600.0, retries: int = 5):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.refusal_mass = refusal_mass
        if top_logprobs is None:
            # Exactly what the server's processor exposes: every candidate
            # token plus the sink (the server is started with the same count).
            from transformers import AutoTokenizer

            from plugins.local_judge.measure.judge import scan_vocabulary

            top_logprobs = len(scan_vocabulary(AutoTokenizer.from_pretrained(model))) + 1
        self.top_logprobs = int(top_logprobs)
        self.concurrency = int(concurrency)
        self.timeout = float(timeout)
        self.retries = int(retries)
        # Same instrument as the HF judge but a different engine: a distinct id,
        # so the two never share cache entries or leaderboard columns.
        self.id = f"local-server:{model}:logprobs:vllm"
        if refusal_mass != 0.25:
            self.id += f":rm{refusal_mass:g}"

    # -- availability --------------------------------------------------
    def available(self) -> bool:
        import httpx

        try:
            r = httpx.get(f"{self.base_url}/models", timeout=10.0)
            r.raise_for_status()
            served = {m.get("id") for m in r.json().get("data", [])}
        except Exception as exc:  # noqa: BLE001
            print(f"[judge] {self.id}: no judge server at {self.base_url} ({exc!r}); "
                  "start one with scripts/serve_judge.py")
            return False
        if self.model not in served:
            print(f"[judge] {self.id}: server at {self.base_url} serves {sorted(served)}, "
                  f"not {self.model!r}")
            return False
        return True

    # -- scoring ---------------------------------------------------------
    def _body(self, prompt: str) -> dict[str, Any]:
        return {"model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1, "temperature": 0,
                "logprobs": True, "top_logprobs": self.top_logprobs}

    def _verdict(self, body: Mapping[str, Any]) -> tuple[float | str | None, dict]:
        try:
            top = body["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError):
            return None, {}
        probs = pool_top_logprobs(top)
        return aggregate_0_100(probs, self.refusal_mass), {"probs": probs}

    async def _post(self, client, prompt: str):
        delay = 1.0
        for attempt in range(self.retries + 1):
            try:
                r = await client.post("/chat/completions", json=self._body(prompt))
                if 400 <= r.status_code < 500:
                    # A request the server rejects will be rejected again:
                    # say why once and do not retry.
                    if not getattr(self, "_reported_4xx", False):
                        self._reported_4xx = True
                        print(f"[judge] {self.id}: server rejected the request "
                              f"({r.status_code}): {r.text[:300]}")
                    return None
                if r.status_code >= 500 and attempt < self.retries:
                    raise RuntimeError(f"server {r.status_code}")
                r.raise_for_status()
                return self._verdict(r.json())
            except Exception as exc:  # noqa: BLE001 -- transient server/network errors
                if attempt >= self.retries:
                    print(f"[judge] {self.id}: giving up on a prompt after "
                          f"{self.retries} retries ({exc!r})")
                    return None
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)

    async def score_detail(self, prompt: str) -> tuple[float | str | None, dict]:
        import httpx

        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            res = await self._post(client, prompt)
        return res if res is not None else (None, {})

    def batchable(self) -> bool:
        return True

    async def score_detail_batch(self, prompts: Sequence[str]):
        import httpx

        sem = asyncio.Semaphore(self.concurrency)
        limits = httpx.Limits(max_connections=self.concurrency,
                              max_keepalive_connections=self.concurrency)
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout,
                                     limits=limits) as client:
            async def one(p):
                async with sem:
                    return await self._post(client, p)
            t0 = time.time()
            out = await asyncio.gather(*[one(p) for p in prompts])
            dt = time.time() - t0
            if prompts:
                print(f"[judge] {self.id}: {len(prompts)} prompts in {dt:.1f}s "
                      f"({len(prompts) / max(dt, 1e-9):.0f}/s)")
        return list(out)

    def rederive(self, detail):
        probs = detail.get("probs") if isinstance(detail, Mapping) else None
        if not isinstance(probs, Mapping):
            return NotImplemented
        return aggregate_0_100(probs, self.refusal_mass)


def _make_server(spec, dtype, device) -> BaseJudge:
    if getattr(spec, "scale", None) is not None:
        raise ValueError("the local_server judge scores natively on 0-100 and cannot rescale")
    params = dict(getattr(spec, "params", None) or {})
    params.setdefault("base_url", os.environ.get("EMERGE_JUDGE_URL", DEFAULT_URL))
    return ServerJudge(spec.model, **params)
