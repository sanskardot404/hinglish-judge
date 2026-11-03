"""One async interface over Ollama, Gemini and (optionally) Groq / OpenRouter.

Design in one paragraph: every call goes ``LLMClient.complete(request)`` ->
disk cache -> per-provider concurrency cap -> rate limiter -> HTTP call, with
retries (exponential backoff + jitter, honouring ``Retry-After``) only for
*transient* errors (429, 5xx, timeouts). Permanent errors (400/401/404) fail
immediately. The model tag the API says it served is stored with every result,
because hosted model names are aliases that can change underneath you.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

import httpx
from dotenv import load_dotenv

PROVIDERS = ("ollama", "gemini", "groq", "openrouter")
CACHE_VERSION = 1  # bump to invalidate every old cache entry

_KEY_NAMES = {
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
_OPENAI_COMPAT_BASE = {
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}
_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
_DEFAULT_OLLAMA_HOST = "http://localhost:11434"

Sleep = Callable[[float], Awaitable[None]]


# --------------------------------------------------------------------------- errors
class LLMError(Exception):
    """Base class for everything this module raises on purpose."""


class RetryableError(LLMError):
    """Transient failure (429, 5xx, timeout, connection error): worth retrying."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class FatalError(LLMError):
    """Permanent failure (bad key, unknown model, bad request): never retried."""


# --------------------------------------------------------------------------- data
@dataclass(frozen=True)
class LLMRequest:
    provider: str
    model: str
    prompt: str
    system: str | None = None
    temperature: float = 0.0
    seed: int = 0
    max_tokens: int = 512

    def cache_key(self) -> str:
        """SHA-256 of everything that can change the answer."""
        payload = json.dumps(
            {"v": CACHE_VERSION, **asdict(self)}, sort_keys=True, ensure_ascii=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    requested_model: str
    served_model: str  # what the API says it actually ran; may differ from requested
    cached: bool = False


# --------------------------------------------------------------------------- cache
class DiskCache:
    """One JSON file per request, in ``root/<first 2 hex>/<key>.json``.

    Writes go to a temp file and are renamed into place, so a crash mid-write
    never leaves a half-written (corrupt) entry. Failed calls are never cached.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        try:
            data: dict[str, Any] = json.loads(self._path(key).read_text(encoding="utf-8"))
            return data
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def put(self, key: str, value: dict[str, Any]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{key}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)  # atomic on Windows and POSIX


# --------------------------------------------------------------------------- helpers
def _redact(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text


def _parse_retry_after(resp: httpx.Response) -> float | None:
    raw = resp.headers.get("retry-after")
    try:
        return float(raw) if raw is not None else None
    except ValueError:
        return None  # HTTP-date form: ignore, fall back to our own backoff


def _raise_for_status(resp: httpx.Response, secrets: list[str]) -> None:
    if resp.status_code < 400:
        return
    detail = _redact(resp.text[:300], secrets)
    message = f"HTTP {resp.status_code}: {detail}"
    if resp.status_code == 429 or resp.status_code >= 500:
        raise RetryableError(message, retry_after=_parse_retry_after(resp))
    raise FatalError(message)


class _MinInterval:
    """Allow at most one request start per ``interval`` seconds (e.g. free-tier RPM caps)."""

    def __init__(self, interval: float, sleep: Sleep) -> None:
        self.interval = interval
        self._sleep = sleep
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        if self.interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            if start > now:
                await self._sleep(start - now)
            self._next = start + self.interval


# --------------------------------------------------------------------------- client
class LLMClient:
    def __init__(
        self,
        *,
        cache_dir: Path | str = "results/cache",
        http: httpx.AsyncClient | None = None,
        env: Mapping[str, str] | None = None,
        sleep: Sleep = asyncio.sleep,
        max_attempts: int = 5,
        base_delay: float = 1.0,
        concurrency: Mapping[str, int] | None = None,
        min_interval: Mapping[str, float] | None = None,
    ) -> None:
        if env is None:
            load_dotenv()
            env = os.environ
        self._env = env
        self._own_http = http is None
        self._http = http or httpx.AsyncClient(timeout=120.0)
        self._cache = DiskCache(Path(cache_dir))
        self._sleep = sleep
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        limits = {p: 2 for p in PROVIDERS} | dict(concurrency or {})
        self._sems = {p: asyncio.Semaphore(n) for p, n in limits.items()}
        intervals = {"gemini": 6.0} | dict(min_interval or {})
        self._limiters = {p: _MinInterval(s, sleep) for p, s in intervals.items()}

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._own_http:
            await self._http.aclose()

    # ---- public
    async def complete(self, req: LLMRequest) -> LLMResponse:
        if req.provider not in PROVIDERS:
            raise FatalError(f"unknown provider {req.provider!r}; use one of {PROVIDERS}")
        key = req.cache_key()
        hit = self._cache.get(key)
        if hit is not None:  # checked BEFORE any rate-limit wait or network call
            return LLMResponse(**{**hit["response"], "cached": True})

        async with self._sems[req.provider]:
            text, served = await self._call_with_retry(req)

        resp = LLMResponse(
            text=text, provider=req.provider, requested_model=req.model, served_model=served
        )
        stored = {k: v for k, v in asdict(resp).items() if k != "cached"}
        self._cache.put(key, {"request": asdict(req), "response": stored})
        return resp

    # ---- internals
    async def _call_with_retry(self, req: LLMRequest) -> tuple[str, str]:
        for attempt in range(self._max_attempts):
            limiter = self._limiters.get(req.provider)
            if limiter is not None:
                await limiter.wait()
            try:
                return await self._call_once(req)
            except RetryableError as err:
                if attempt == self._max_attempts - 1:
                    raise
                if err.retry_after is not None:
                    delay = err.retry_after
                else:
                    delay = self._base_delay * 2**attempt + random.uniform(0, self._base_delay)
                await self._sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    def _key(self, provider: str) -> str:
        name = _KEY_NAMES[provider]
        value = self._env.get(name, "")
        if not value:
            raise FatalError(f"{name} is not set (put it in .env)")
        return value

    def _secrets(self) -> list[str]:
        return [self._env.get(n, "") for n in _KEY_NAMES.values()]

    async def _call_once(self, req: LLMRequest) -> tuple[str, str]:
        try:
            if req.provider == "ollama":
                return await self._ollama(req)
            if req.provider == "gemini":
                return await self._gemini(req)
            return await self._openai_compat(req)
        except httpx.TransportError as err:  # timeouts, connection resets, DNS
            raise RetryableError(f"network error: {type(err).__name__}") from err

    async def _ollama(self, req: LLMRequest) -> tuple[str, str]:
        host = self._env.get("OLLAMA_HOST") or _DEFAULT_OLLAMA_HOST
        messages = ([{"role": "system", "content": req.system}] if req.system else []) + [
            {"role": "user", "content": req.prompt}
        ]
        body = {
            "model": req.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": req.temperature,
                "seed": req.seed,
                "num_predict": req.max_tokens,
            },
        }
        resp = await self._http.post(f"{host.rstrip('/')}/api/chat", json=body)
        _raise_for_status(resp, [])
        data = resp.json()
        return data["message"]["content"], data.get("model", req.model)

    async def _gemini(self, req: LLMRequest) -> tuple[str, str]:
        key = self._key("gemini")
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": req.prompt}]}],
            "generationConfig": {
                "temperature": req.temperature,
                "seed": req.seed,
                "maxOutputTokens": req.max_tokens,
            },
        }
        if req.system:
            body["systemInstruction"] = {"parts": [{"text": req.system}]}
        # Key goes in a header, never in the URL (URLs end up in logs and tracebacks).
        resp = await self._http.post(
            f"{_GEMINI_BASE}/{req.model}:generateContent",
            json=body,
            headers={"x-goog-api-key": key},
        )
        _raise_for_status(resp, self._secrets())
        data = resp.json()
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as err:
            reason = (data.get("candidates") or [{}])[0].get("finishReason", "unknown")
            raise FatalError(f"Gemini returned no text (finishReason={reason})") from err
        return text, data.get("modelVersion", req.model)

    async def _openai_compat(self, req: LLMRequest) -> tuple[str, str]:
        key = self._key(req.provider)
        messages = ([{"role": "system", "content": req.system}] if req.system else []) + [
            {"role": "user", "content": req.prompt}
        ]
        body = {
            "model": req.model,
            "messages": messages,
            "temperature": req.temperature,
            "seed": req.seed,
            "max_tokens": req.max_tokens,
        }
        resp = await self._http.post(
            f"{_OPENAI_COMPAT_BASE[req.provider]}/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {key}"},
        )
        _raise_for_status(resp, self._secrets())
        data = resp.json()
        return data["choices"][0]["message"]["content"], data.get("model", req.model)
