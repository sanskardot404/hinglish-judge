"""All model calls are mocked with respx: no Ollama, no keys, no network."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Iterator

import httpx
import pytest
import respx

from judge_check.llm import (
    FatalError,
    LLMClient,
    LLMRequest,
    RetryableError,
)

OLLAMA_URL = "http://localhost:11434/api/chat"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
FAKE_KEY = "AIza-FAKE-KEY-for-tests-0123456789"
ENV = {"GEMINI_API_KEY": FAKE_KEY, "GROQ_API_KEY": "gsk-fake", "OLLAMA_HOST": "http://localhost:11434"}

OLLAMA_OK = {"model": "qwen2.5:7b", "message": {"role": "assistant", "content": "A"}}
GEMINI_OK = {
    "candidates": [{"content": {"parts": [{"text": "B"}]}}],
    "modelVersion": "gemini-3.6-flash",
}


@pytest.fixture
def mock() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as m:
        yield m


def make_client(tmp_path: Path, sleeps: list[float] | None = None, **kw: Any) -> LLMClient:
    async def fake_sleep(seconds: float) -> None:
        if sleeps is not None:
            sleeps.append(seconds)

    kw.setdefault("min_interval", {})  # no real rate-limit waiting in tests
    return LLMClient(cache_dir=tmp_path / "cache", env=ENV, sleep=fake_sleep, base_delay=0.5, **kw)


def ollama_req(prompt: str = "hi", **kw: Any) -> LLMRequest:
    return LLMRequest(provider="ollama", model="qwen2.5:7b", prompt=prompt, **kw)


# ---- cache
async def test_cache_hit_makes_no_second_http_call(mock: respx.MockRouter, tmp_path: Path) -> None:
    route = mock.post(OLLAMA_URL).respond(200, json=OLLAMA_OK)
    async with make_client(tmp_path) as client:
        first = await client.complete(ollama_req())
        second = await client.complete(ollama_req())
    assert route.call_count == 1
    assert (first.cached, second.cached) == (False, True)
    assert first.text == second.text == "A"


async def test_cache_survives_new_client(mock: respx.MockRouter, tmp_path: Path) -> None:
    route = mock.post(OLLAMA_URL).respond(200, json=OLLAMA_OK)
    async with make_client(tmp_path) as c1:
        await c1.complete(ollama_req())
    async with make_client(tmp_path) as c2:  # "resumable run"
        again = await c2.complete(ollama_req())
    assert route.call_count == 1 and again.cached


def test_cache_key_changes_with_every_input() -> None:
    base = ollama_req()
    variants = [
        ollama_req("other prompt"),
        ollama_req(temperature=0.7),
        ollama_req(seed=1),
        ollama_req(system="be strict"),
        LLMRequest(provider="ollama", model="llama3.1:8b", prompt="hi"),
    ]
    keys = {base.cache_key(), *(v.cache_key() for v in variants)}
    assert len(keys) == len(variants) + 1


async def test_failures_are_not_cached(mock: respx.MockRouter, tmp_path: Path) -> None:
    route = mock.post(OLLAMA_URL).mock(
        side_effect=[httpx.Response(404, text="no model"), httpx.Response(200, json=OLLAMA_OK)]
    )
    async with make_client(tmp_path) as client:
        with pytest.raises(FatalError):
            await client.complete(ollama_req())
        ok = await client.complete(ollama_req())
    assert ok.text == "A" and route.call_count == 2


# ---- retries
async def test_retries_429_then_succeeds(mock: respx.MockRouter, tmp_path: Path) -> None:
    route = mock.post(OLLAMA_URL).mock(
        side_effect=[httpx.Response(429), httpx.Response(503), httpx.Response(200, json=OLLAMA_OK)]
    )
    sleeps: list[float] = []
    async with make_client(tmp_path, sleeps) as client:
        resp = await client.complete(ollama_req())
    assert resp.text == "A" and route.call_count == 3
    assert len(sleeps) == 2
    assert 0.5 <= sleeps[0] <= 1.0 and 1.0 <= sleeps[1] <= 1.5  # base*2**n + jitter


async def test_retry_after_header_is_honoured(mock: respx.MockRouter, tmp_path: Path) -> None:
    mock.post(OLLAMA_URL).mock(
        side_effect=[
            httpx.Response(429, headers={"retry-after": "7"}),
            httpx.Response(200, json=OLLAMA_OK),
        ]
    )
    sleeps: list[float] = []
    async with make_client(tmp_path, sleeps) as client:
        await client.complete(ollama_req())
    assert sleeps == [7.0]


@pytest.mark.parametrize("status", [400, 401, 404])
async def test_permanent_errors_are_not_retried(
    mock: respx.MockRouter, tmp_path: Path, status: int
) -> None:
    route = mock.post(OLLAMA_URL).respond(status, text="nope")
    async with make_client(tmp_path) as client:
        with pytest.raises(FatalError):
            await client.complete(ollama_req())
    assert route.call_count == 1


async def test_gives_up_after_max_attempts(mock: respx.MockRouter, tmp_path: Path) -> None:
    route = mock.post(OLLAMA_URL).respond(503)
    async with make_client(tmp_path, max_attempts=3) as client:
        with pytest.raises(RetryableError):
            await client.complete(ollama_req())
    assert route.call_count == 3


async def test_timeout_is_retried(mock: respx.MockRouter, tmp_path: Path) -> None:
    route = mock.post(OLLAMA_URL).mock(
        side_effect=[httpx.ReadTimeout("slow"), httpx.Response(200, json=OLLAMA_OK)]
    )
    async with make_client(tmp_path) as client:
        resp = await client.complete(ollama_req())
    assert resp.text == "A" and route.call_count == 2


# ---- concurrency
async def test_concurrency_cap_is_respected(mock: respx.MockRouter, tmp_path: Path) -> None:
    state = {"now": 0, "peak": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        state["now"] += 1
        state["peak"] = max(state["peak"], state["now"])
        await asyncio.sleep(0.01)
        state["now"] -= 1
        return httpx.Response(200, json=OLLAMA_OK)

    mock.post(OLLAMA_URL).mock(side_effect=handler)
    async with make_client(tmp_path, concurrency={"ollama": 2}) as client:
        await asyncio.gather(*(client.complete(ollama_req(f"p{i}")) for i in range(8)))
    assert state["peak"] == 2


# ---- providers
async def test_gemini_records_served_model_and_uses_header_for_key(
    mock: respx.MockRouter, tmp_path: Path
) -> None:
    route = mock.post(GEMINI_URL).respond(200, json=GEMINI_OK)
    req = LLMRequest(provider="gemini", model="gemini-3.5-flash", prompt="hi")
    async with make_client(tmp_path) as client:
        resp = await client.complete(req)
    assert resp.requested_model == "gemini-3.5-flash"
    assert resp.served_model == "gemini-3.6-flash"  # the alias-drift lesson
    sent = route.calls[0].request
    assert sent.headers["x-goog-api-key"] == FAKE_KEY
    assert FAKE_KEY not in str(sent.url)


async def test_api_key_never_appears_in_errors(mock: respx.MockRouter, tmp_path: Path) -> None:
    mock.post(GEMINI_URL).respond(400, text=f"API key {FAKE_KEY} not valid")
    req = LLMRequest(provider="gemini", model="gemini-3.5-flash", prompt="hi")
    async with make_client(tmp_path) as client:
        with pytest.raises(FatalError) as info:
            await client.complete(req)
    assert FAKE_KEY not in str(info.value)


async def test_gemini_empty_answer_is_a_clear_error(mock: respx.MockRouter, tmp_path: Path) -> None:
    mock.post(GEMINI_URL).respond(200, json={"candidates": [{"finishReason": "SAFETY"}]})
    req = LLMRequest(provider="gemini", model="gemini-3.5-flash", prompt="hi")
    async with make_client(tmp_path) as client:
        with pytest.raises(FatalError, match="SAFETY"):
            await client.complete(req)


async def test_groq_openai_compatible_shape(mock: respx.MockRouter, tmp_path: Path) -> None:
    route = mock.post(GROQ_URL).respond(
        200, json={"model": "llama-3.3-70b", "choices": [{"message": {"content": "A"}}]}
    )
    req = LLMRequest(provider="groq", model="llama-3.3-70b", prompt="hi")
    async with make_client(tmp_path) as client:
        resp = await client.complete(req)
    assert resp.text == "A" and resp.served_model == "llama-3.3-70b"
    assert route.calls[0].request.headers["authorization"] == "Bearer gsk-fake"


async def test_missing_key_is_a_clear_error(tmp_path: Path) -> None:
    client = LLMClient(cache_dir=tmp_path / "cache", env={}, min_interval={})
    req = LLMRequest(provider="gemini", model="gemini-3.5-flash", prompt="hi")
    async with client:
        with pytest.raises(FatalError, match="GEMINI_API_KEY"):
            await client.complete(req)


async def test_unknown_provider_is_rejected(tmp_path: Path) -> None:
    async with make_client(tmp_path) as client:
        with pytest.raises(FatalError, match="unknown provider"):
            await client.complete(LLMRequest(provider="openai", model="x", prompt="hi"))
