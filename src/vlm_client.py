"""
PackVision AI - Shared Vision-Language Model (VLM) Client
==========================================================
Thin, defensive client for text+image VLM calls used by the pipeline
(reference-image style hints, prompt enrichment, future captioning/QA).

Provider detection order (first configured wins, configurable via VLM_PROVIDER):
1. Google Gemini  - generativelanguage.googleapis.com (default model: gemini-3.8-flash,
                    thinking disabled via generationConfig.thinkingConfig.thinkingBudget=0
                    for latency; falls back to provider defaults if the field is rejected)
2. OpenRouter     - openrouter.ai chat completions (free vision models first)
3. OpenAI         - api.openai.com chat completions (gpt-4o-mini)

Design rules:
- NEVER raises to callers: ask_vlm()/describe_image() return "" on any failure
  and log a warning instead.
- Responses are cached in-memory (sha256 of prompt + image bytes, maxsize 256).
- Failed/empty provider answers are negative-cached for 60s per (provider, request)
  so rate-limit storms / timeouts do not re-pay the latency on identical retries.
- Timeouts are bounded: VLM_TIMEOUT_S (default 45s) per HTTP request and
  VLM_TOTAL_TIMEOUT_S (default 90s) for the ENTIRE provider chain including retries.
- httpx with 1..2 retries on 429/5xx/transport errors with 2s/4s backoff.
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment loading (.env fallback loader - python-dotenv if available)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_ENV_LOCK = threading.Lock()
_ENV_LOADED = False


def load_project_env(env_path: Optional[str] = None) -> bool:
    """Load the project .env file into os.environ exactly once per process.

    Existing environment variables are never overridden. Uses python-dotenv
    when installed, otherwise falls back to a tiny dotenv-style parser.
    Returns True when the (idempotent) load has been performed.
    """
    global _ENV_LOADED
    with _ENV_LOCK:
        if _ENV_LOADED:
            return True
        path = Path(env_path) if env_path else PROJECT_ROOT / ".env"
        if path.exists():
            try:
                from dotenv import load_dotenv  # type: ignore

                load_dotenv(dotenv_path=str(path), override=False)
                _ENV_LOADED = True
                return True
            except Exception:
                pass
            try:
                for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
                _ENV_LOADED = True
            except Exception as env_err:
                logger.warning("Could not parse .env at %s: %s", path, env_err)
                return False
        else:
            _ENV_LOADED = True
        return _ENV_LOADED


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

GEMINI_TEXT_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"

# Timeout policy (bounded worst case for the whole chain):
VLM_HTTP_TIMEOUT_DEFAULT_S = 30.0  # per-request; override with VLM_TIMEOUT_S
VLM_TOTAL_TIMEOUT_DEFAULT_S = 75.0  # whole chain incl. retries; override with VLM_TOTAL_TIMEOUT_S
VLM_RETRY_BACKOFFS_S: Tuple[float, ...] = (2.0, 4.0)  # max 2 retries

# Failed/empty provider answers are remembered for this long per (provider, request).
NEGATIVE_CACHE_TTL_S = 60.0

# Verified against the live REST API: gemini-3.8-flash accepts
# generationConfig.thinkingConfig={"thinkingBudget": 0} (no 400 rejection).
# Thinking disabled by default for latency; set GEMINI_VLM_THINKING_BUDGET to re-enable.
GEMINI_DEFAULT_THINKING_BUDGET = 0

# Free OpenRouter vision models tried (in order) when the configured model
# fails: preferred free model first, "then any" free fallback.
# (Refreshed 2026-11-14 against OpenRouter's live catalog — the previously
# listed qwen2.5-vl-72b/llama-4-maverick/gemini-2.0-flash-exp free slugs were
# all removed and now 404.)
OPENROUTER_FREE_VLM_FALLBACKS: Tuple[str, ...] = (
    "nvidia/nemotron-3.5-super-vl:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "particle-ai/particle-1:free",
    "dots-studio/dots-3-note-preview:free",
    "thinkingmachines/inkling-2-small:free",
)


def _env(key: str, default: str = "") -> str:
    return (os.getenv(key) or default).strip()


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _per_request_timeout() -> float:
    return _env_float("VLM_TIMEOUT_S", VLM_HTTP_TIMEOUT_DEFAULT_S)


def _total_timeout() -> float:
    return _env_float("VLM_TOTAL_TIMEOUT_S", VLM_TOTAL_TIMEOUT_DEFAULT_S)


def _gemini_thinking_budget() -> int:
    raw = os.getenv("GEMINI_VLM_THINKING_BUDGET")
    if raw is None or not raw.strip():
        return GEMINI_DEFAULT_THINKING_BUDGET
    try:
        return int(raw)
    except ValueError:
        return GEMINI_DEFAULT_THINKING_BUDGET


def default_gemini_vlm_model() -> str:
    return _env("GEMINI_VLM_MODEL", "gemini-3.8-flash")


def default_openrouter_vlm_model() -> str:
    return _env("OPENROUTER_VLM_MODEL", "nvidia/nemotron-3.5-super-vl:free")


def default_openai_vlm_model() -> str:
    return _env("OPENAI_VLM_MODEL", "gpt-4o-mini")


def detect_vlm_providers() -> List[Tuple[str, str]]:
    """Return the ordered [(provider, model)] chain of configured VLM providers.

    Order: GEMINI -> OPENROUTER (free models) -> OPENAI.
    The VLM_PROVIDER env var can force a single provider ("gemini"|"openrouter"|"openai").
    """
    load_project_env()
    forced = _env("VLM_PROVIDER", "auto").lower() or "auto"

    gemini_ok = bool(_env("GEMINI_API_KEY"))
    openrouter_ok = bool(_env("OPENROUTER_API_KEY"))
    openai_ok = bool(_env("OPENAI_API_KEY"))

    if forced in ("", "auto"):
        chain: List[Tuple[str, str]] = []
        if gemini_ok:
            chain.append(("gemini", default_gemini_vlm_model()))
        if openrouter_ok:
            chain.append(("openrouter", default_openrouter_vlm_model()))
        if openai_ok:
            chain.append(("openai", default_openai_vlm_model()))
        return chain

    if forced == "gemini" and gemini_ok:
        return [("gemini", default_gemini_vlm_model())]
    if forced == "openrouter" and openrouter_ok:
        return [("openrouter", default_openrouter_vlm_model())]
    if forced == "openai" and openai_ok:
        return [("openai", default_openai_vlm_model())]
    return []


def vlm_available() -> bool:
    """Cheap availability check: True when at least one VLM provider key is configured.

    NOTE: True means "keys configured", NOT "network reachable" - it performs no
    I/O. Use ask_vlm()/describe_image() results (or the probe scripts) to confirm
    actual connectivity.
    """
    return bool(detect_vlm_providers())


# ---------------------------------------------------------------------------
# In-memory response cache (thread-safe, LRU-ish, maxsize 256)
# ---------------------------------------------------------------------------

_CACHE_LOCK = threading.Lock()
_CACHE: Dict[str, str] = {}
_CACHE_MAXSIZE = 256

# Negative cache: (provider|request_key) -> monotonic expiry timestamp.
_NEG_CACHE_LOCK = threading.Lock()
_NEG_CACHE: Dict[str, float] = {}


def clear_cache() -> None:
    """Drop all cached VLM responses, positive and negative (tests/ops)."""
    with _CACHE_LOCK:
        _CACHE.clear()
    with _NEG_CACHE_LOCK:
        _NEG_CACHE.clear()


def _cache_key(prompt: str, image: Optional[Image.Image], json_mode: bool, max_tokens: int) -> str:
    digest = hashlib.sha256()
    digest.update(prompt.encode("utf-8", errors="ignore"))
    if image is not None:
        digest.update(_png_bytes(image))
    digest.update(f"|json={int(bool(json_mode))}|tok={int(max_tokens)}".encode("ascii"))
    return digest.hexdigest()


def _cache_get(key: str) -> Optional[str]:
    with _CACHE_LOCK:
        if key in _CACHE:
            value = _CACHE.pop(key)
            _CACHE[key] = value  # refresh LRU position
            return value
    return None


def _cache_put(key: str, value: str) -> None:
    with _CACHE_LOCK:
        _CACHE[key] = value
        while len(_CACHE) > _CACHE_MAXSIZE:
            _CACHE.pop(next(iter(_CACHE)))


def _negative_cache_hit(provider: str, key: str) -> bool:
    stamp = f"{provider}|{key}"
    with _NEG_CACHE_LOCK:
        expiry = _NEG_CACHE.get(stamp)
        if expiry is None:
            return False
        if time.monotonic() < expiry:
            return True
        _NEG_CACHE.pop(stamp, None)
        return False


def _negative_cache_mark(provider: str, key: str) -> None:
    with _NEG_CACHE_LOCK:
        _NEG_CACHE[f"{provider}|{key}"] = time.monotonic() + NEGATIVE_CACHE_TTL_S


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    img = image
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def _to_data_uri(image: Image.Image) -> str:
    return "data:image/png;base64," + base64.b64encode(_png_bytes(image)).decode("ascii")


# ---------------------------------------------------------------------------
# HTTP layer (bounded timeout, max 2 retries with 2s/4s backoff)
# ---------------------------------------------------------------------------


class BudgetExceeded(Exception):
    """Internal: the VLM_TOTAL_TIMEOUT_S budget for the whole chain is exhausted."""


def _post_json(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: Optional[float] = None,
    deadline: Optional[float] = None,
):
    """POST JSON with up to 2 retries (2s/4s backoff) on 429/5xx/transport errors.

    Args:
        timeout: per-request httpx timeout in seconds (default VLM_TIMEOUT_S).
        deadline: absolute time.monotonic() cap for the whole provider chain;
                  requests and backoff sleeps are clamped to it.

    Returns the final httpx.Response (including non-retryable error statuses).
    Raises httpx.HTTPError after retries are exhausted on transport errors, or
    BudgetExceeded when the deadline runs out.
    """
    request_timeout = timeout if timeout is not None else _per_request_timeout()
    last_error: Optional[Exception] = None
    max_attempts = len(VLM_RETRY_BACKOFFS_S) + 1  # initial + 2 retries
    server_errors_seen = 0  # 5xx = provider outage; fail over fast (1 retry max)

    for attempt in range(max_attempts):
        budget = request_timeout
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 1.0:
                if last_error is not None:
                    raise last_error
                raise BudgetExceeded(f"VLM total time budget exhausted before {url}")
            budget = min(budget, remaining)
        try:
            resp = httpx.post(url, headers=headers, json=payload, timeout=budget)
        except httpx.HTTPError as exc:
            last_error = exc
            resp = None
        if resp is not None and resp.status_code != 429 and resp.status_code < 500:
            return resp
        if resp is not None and resp.status_code >= 500:
            # Provider outage (e.g. Gemini 503 "high demand"): one quick retry,
            # then give up so the chain can fail over to the next provider
            # instead of burning the whole budget.
            server_errors_seen += 1
            if server_errors_seen >= 2:
                return resp
        if attempt == max_attempts - 1:
            if resp is not None:
                return resp  # final 429/5xx -> caller treats as failure
            raise (
                last_error
                if last_error is not None
                else RuntimeError(f"Request to {url} failed")
            )
        delay = VLM_RETRY_BACKOFFS_S[attempt]
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 1.0:
                if resp is not None:
                    return resp
                raise (
                    last_error
                    if last_error is not None
                    else BudgetExceeded("VLM total time budget exhausted during backoff")
                )
            delay = max(0.0, min(delay, remaining - 1.0))
        time.sleep(delay)

    raise last_error or RuntimeError(f"Request to {url} failed")


# ---------------------------------------------------------------------------
# Provider adapters (each returns "" on failure and logs a warning)
# ---------------------------------------------------------------------------

def _call_gemini(
    prompt: str,
    image: Optional[Image.Image],
    model: str,
    max_tokens: int,
    json_mode: bool,
    deadline: Optional[float] = None,
) -> str:
    key = _env("GEMINI_API_KEY")
    if not key:
        return ""
    parts: List[Dict[str, Any]] = [{"text": prompt}]
    if image is not None:
        parts.append({"inline_data": {"mime_type": "image/png", "data": base64.b64encode(_png_bytes(image)).decode("ascii")}})

    generation_config: Dict[str, Any] = {"maxOutputTokens": max_tokens, "temperature": 0.4}
    if json_mode:
        generation_config["responseMimeType"] = "application/json"
    if model.startswith("gemini"):
        # Thinking disabled by default for latency (verified accepted by the REST
        # API); gracefully dropped if this model/endpoint rejects the field.
        generation_config["thinkingConfig"] = {"thinkingBudget": _gemini_thinking_budget()}

    payload = {"contents": [{"parts": parts}], "generationConfig": generation_config}
    url = f"{GEMINI_TEXT_BASE}/{model}:generateContent"
    headers = {"x-goog-api-key": key}
    resp = _post_json(url, headers, payload, deadline=deadline)
    if resp.status_code == 400 and "thinking" in resp.text.lower() and "thinkingConfig" in generation_config:
        generation_config.pop("thinkingConfig", None)
        resp = _post_json(url, headers, payload, deadline=deadline)
    if resp.status_code != 200:
        logger.warning("Gemini VLM HTTP %s: %s", resp.status_code, resp.text[:200])
        return ""
    try:
        data = resp.json()
    except Exception as parse_err:
        logger.warning("Gemini VLM returned invalid JSON: %s", parse_err)
        return ""
    chunks: List[str] = []
    for candidate in data.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            text = part.get("text")
            if text:
                chunks.append(text)
    return "".join(chunks).strip()


def _call_openrouter(
    prompt: str,
    image: Optional[Image.Image],
    model: str,
    max_tokens: int,
    json_mode: bool,
    deadline: Optional[float] = None,
) -> str:
    key = _env("OPENROUTER_API_KEY")
    if not key:
        return ""
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    if image is not None:
        content.append({"type": "image_url", "image_url": {"url": _to_data_uri(image)}})
    messages = [{"role": "user", "content": content}]

    models_to_try = [model] + [m for m in OPENROUTER_FREE_VLM_FALLBACKS if m != model][:2]
    for candidate_model in models_to_try:
        # Reasoning-style free models (e.g. nemotron VL) burn max_tokens on
        # hidden reasoning and may not support response_format — give them
        # headroom and retry once without the JSON constraint if empty.
        payload_max_tokens = max(max_tokens, 700)
        use_response_format = json_mode
        for attempt_variant in (0, 1):
            payload: Dict[str, Any] = {
                "model": candidate_model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": payload_max_tokens,
            }
            if use_response_format:
                payload["response_format"] = {"type": "json_object"}
            try:
                resp = _post_json(
                    OPENROUTER_CHAT_URL,
                    {
                        "Authorization": f"Bearer {key}",
                        "HTTP-Referer": "https://packvision.local",
                        "X-Title": "PackVision AI",
                    },
                    payload,
                    deadline=deadline,
                )
            except httpx.HTTPError as req_err:
                logger.warning("OpenRouter VLM request failed (%s): %s", candidate_model, req_err)
                break  # transport failure: this model is out, next candidate
            if resp.status_code != 200:
                logger.warning("OpenRouter VLM HTTP %s (%s): %s", resp.status_code, candidate_model, resp.text[:200])
                break  # next candidate
            try:
                data = resp.json()
            except Exception as parse_err:
                logger.warning("OpenRouter VLM returned invalid JSON: %s", parse_err)
                break
            message = data.get("choices", [{}])[0].get("message", {})
            text = message.get("content")
            if isinstance(text, list):
                text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
            text = (text or "").strip()
            if text:
                return text
            # 200 with empty content: reasoning models do this when max_tokens
            # is exhausted or response_format is unsupported — retry without
            # the constraint (variant 1) before moving on.
            if attempt_variant == 0:
                use_response_format = False
                continue
            logger.warning(
                "OpenRouter VLM (%s) returned 200 with empty content; skipping candidate.", candidate_model
            )
    return ""


def _call_openai(
    prompt: str,
    image: Optional[Image.Image],
    model: str,
    max_tokens: int,
    json_mode: bool,
    deadline: Optional[float] = None,
) -> str:
    key = _env("OPENAI_API_KEY")
    if not key:
        return ""
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    if image is not None:
        content.append({"type": "image_url", "image_url": {"url": _to_data_uri(image)}})
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    resp = _post_json(OPENAI_CHAT_URL, {"Authorization": f"Bearer {key}"}, payload, deadline=deadline)
    if resp.status_code != 200:
        logger.warning("OpenAI VLM HTTP %s: %s", resp.status_code, resp.text[:200])
        return ""
    try:
        data = resp.json()
    except Exception as parse_err:
        logger.warning("OpenAI VLM returned invalid JSON: %s", parse_err)
        return ""
    text = data.get("choices", [{}])[0].get("message", {}).get("content") or ""
    if isinstance(text, list):
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    return text.strip()


_PROVIDERS = {
    "gemini": _call_gemini,
    "openrouter": _call_openrouter,
    "openai": _call_openai,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ask_vlm(
    prompt: str,
    image: Optional[Image.Image] = None,
    json_mode: bool = False,
    max_tokens: int = 512,
) -> str:
    """Ask the first available VLM provider. Returns "" on any failure (never raises).

    The entire provider chain (including retries) is bounded by VLM_TOTAL_TIMEOUT_S
    (default 90s); each HTTP request is bounded by VLM_TIMEOUT_S (default 45s).
    Successful answers are cached (maxsize 256); failed/empty provider answers are
    negative-cached for 60s per (provider, request) to bound storm costs.
    """
    load_project_env()
    prompt = (prompt or "").strip()
    if not prompt:
        return ""

    key = _cache_key(prompt, image, json_mode, max_tokens)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    chain = detect_vlm_providers()
    if not chain:
        logger.warning("VLM unavailable: no provider API key configured (GEMINI/OPENROUTER/OPENAI).")
        return ""

    deadline = time.monotonic() + _total_timeout()
    result = ""
    for provider, model in chain:
        if _negative_cache_hit(provider, key):
            logger.info(
                "VLM provider '%s' skipped (negative-cached failure, TTL %.0fs).",
                provider,
                NEGATIVE_CACHE_TTL_S,
            )
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 1.0:
            logger.warning(
                "VLM total time budget (%.0fs) exhausted; aborting before provider '%s'.",
                _total_timeout(),
                provider,
            )
            break

        fn = _PROVIDERS.get(provider)
        if fn is None:
            continue
        try:
            result = fn(prompt, image, model, max_tokens, json_mode, deadline=deadline)
        except BudgetExceeded as exc:
            logger.warning("VLM chain aborted at provider '%s': %s", provider, exc)
            result = ""
            break
        except Exception as exc:  # noqa: BLE001 - defensive by contract
            logger.warning("VLM provider '%s' (%s) failed: %s", provider, model, exc)
            result = ""
        if result:
            break
        _negative_cache_mark(provider, key)

    if result:
        _cache_put(key, result)
    else:
        logger.warning("VLM call returned no content (chain tried: %s).", [p for p, _ in chain])
    return result


def describe_image(image: Image.Image, question: str = "Describe this image concisely.") -> str:
    """Describe an image with the first available VLM. Returns "" on failure."""
    return ask_vlm(question, image=image)
