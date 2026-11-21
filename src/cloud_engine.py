"""
PackVision AI - Hybrid Cloud Render Engine
===========================================
Renders packaging sketches to photoreal mockups through cloud image models:

1. HuggingFace - ZeroGPU spaces (gradio_client), default
             black-forest-labs/FLUX.1-Kontext-dev: FREE, no API key, ~45s per
             render (sketch is processed by HuggingFace's public inference
             service; anonymous quota is limited - set HF_TOKEN to raise it).
             Tried FIRST whenever enabled.
2. Cloudflare - Workers AI img2img REST
             (@cf/runwayml/stable-diffusion-v1-5-img2img by default; FREE tier
             of 10,000 Neurons/day, no credit card; needs CLOUDFLARE_API_TOKEN
             + CLOUDFLARE_ACCOUNT_ID). Tried after HuggingFace while both are
             free; SD1.5 img2img quality is below Kontext - free fallback tier.
3. ModelScope - API-Inference Qwen/Qwen-Image-Edit (HONEST SKELETON: the REST
             image-edit endpoint takes a publicly HOSTED image_url, not inline
             base64, so this provider is dormant via MODELSCOPE_ENABLED=false
             and raises a clear error instead of firing a doomed request).
4. Stability - api.stability.ai/v2beta/stable-image/control/sketch (multipart
               sketch in, raw image bytes out; purpose-built sketch->photoreal,
               tried when STABILITY_API_KEY is set)
5. Gemini  - generativelanguage.googleapis.com {model}:generateContent
             (inline_data sketch in, inlineData image out)
6. OpenRouter - openrouter.ai chat completions with modalities ["image","text"]
             (data-URI sketch in, data-URI image out at choices[0].message.images)
7. OpenAI  - api.openai.com images/edits (multipart sketch in, b64_json out)

Failure handling contract:
- HTTP 402 / 403 / quota errors -> CloudUnavailableError
- HTTP 429 (rate limit)         -> CloudRateLimitedError
- Any of the above marks the provider unavailable for PROVIDER_COOLDOWN_S
  (15 minutes; 10 minutes for the free HuggingFace path) in a thread-safe,
  module-level registry (persisted across process restarts, keyed by
  API-key fingerprint).
- render() walks the provider chain (auto mode) and only raises once every
  candidate has failed, so the pipeline can fall back to local engines.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import httpx
import numpy as np
from PIL import Image

from src.vlm_client import load_project_env

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CloudEngineError(Exception):
    """Base error for the cloud render engine."""


class CloudUnavailableError(CloudEngineError):
    """Provider is unavailable (no key, 402 payment required, 403, quota exhausted)."""


class CloudRateLimitedError(CloudEngineError):
    """Provider responded with a rate limit (429)."""


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class CloudRenderResult:
    """Successful cloud render output."""

    image: Image.Image
    provider: str
    model: str
    latency_s: float
    cost_note: str = ""
    engine_tag: str = ""  # short engine id used for last_engine_used (e.g. "hf-flux-kontext")


# ---------------------------------------------------------------------------
# Provider cooldown registry (module-level, thread-safe)
# ---------------------------------------------------------------------------

PROVIDER_COOLDOWN_S = 15 * 60  # 15 minutes
# ZeroGPU quota is a ROLLING GPU-seconds window that typically resets within a
# minute ("try again in 0:00:00") — retry soon instead of a long fallback stretch.
HF_COOLDOWN_S = 75

_PROVIDER_LOCK = threading.Lock()
_PROVIDER_STATE: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# HuggingFace ZeroGPU (free, no API key) - gradio_client connection cache.
# Reconnecting to a space costs ~4s, so Client instances are cached per
# (space, token) for the life of the process.
# ---------------------------------------------------------------------------

HF_DEFAULT_SPACE = "black-forest-labs/FLUX.1-Kontext-dev"
HF_PREDICT_TIMEOUT_S = float(os.getenv("HF_TIMEOUT_S", "240"))

_HF_CLIENT_LOCK = threading.Lock()
_HF_CLIENTS: Dict[str, Any] = {}
_HF_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="hf-flux")

# Per-space cooldown registry: space id -> unix ts until which it is skipped.
# Process-local on purpose (no key to fingerprint; short windows anyway).
_HF_SPACE_STATE: Dict[str, float] = {}


def _get_hf_client(space: str, token: Optional[str] = None, client_factory: Optional[Callable] = None):
    """Return a cached gradio_client.Client for the space (reconnects are ~4s)."""
    cache_key = f"{space}|{token or ''}"
    with _HF_CLIENT_LOCK:
        cached = _HF_CLIENTS.get(cache_key)
    if cached is not None:
        return cached
    if client_factory is None:
        from gradio_client import Client as client_factory  # noqa: N813 - deferred import
    client = client_factory(space, token=token or None, verbose=False)
    with _HF_CLIENT_LOCK:
        _HF_CLIENTS[cache_key] = client
    return client

# Cooldowns survive process restarts via a small state file, so a backend that
# restarts (or a fresh test process) does not re-pay the slow 402/429 discovery
# for providers known to be unavailable. Entries carry a fingerprint of the API
# key they apply to, so cooldowns learned under one key never suppress a
# different key (e.g. test-run artifacts cannot poison production keys).
_CLOUD_STATE_FILE = Path(
    os.getenv("CLOUD_STATE_FILE", str(Path(__file__).resolve().parents[1] / "storage" / "cloud_provider_state.json"))
)
_FILE_STATE_LOADED = False


def _key_fingerprint(key: Optional[str]) -> str:
    if not key:
        return ""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

CLOUD_HTTP_TIMEOUT_S = 240.0

GEMINI_API_BASE = os.getenv(
    "GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta/models"
)
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_IMAGES_EDITS_URL = "https://api.openai.com/v1/images/edits"
STABILITY_SKETCH_URL = "https://api.stability.ai/v2beta/stable-image/control/sketch"
# Workers AI "Run model" endpoint (verified against developers.cloudflare.com
# /workers-ai/get-started/rest-api): POST .../ai/run/{model} with
# Authorization: Bearer {token}; image models answer either with raw image
# bytes or an {"success":true,"result":...} envelope depending on the model /
# Accept header - both shapes are parsed defensively.
CLOUDFLARE_RUN_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
CLOUDFLARE_NUM_STEPS = 20  # modest steps conserve the free daily Neuron budget
# ModelScope API-Inference (OpenAI-images-compatible). NOTE: documented
# "image_url" input requires a publicly hosted URL - see _render_modelscope.
MODELSCOPE_IMAGES_URL = "https://api-inference.modelscope.cn/v1/images/generations"

# Free tiers first: the key-less HuggingFace ZeroGPU FLUX.1-Kontext path, then
# Cloudflare Workers AI (free 10,000 Neurons/day once its keys are set), then
# the dormant ModelScope skeleton, then paid providers.
_PROVIDER_ORDER: Tuple[str, ...] = (
    "huggingface",
    "cloudflare",
    "modelscope",
    "stability",
    "gemini",
    "openrouter",
    "openai",
)


def _persist_state_locked() -> None:
    """Write the cooldown registry to disk. Caller must hold _PROVIDER_LOCK.

    HuggingFace cooldowns are intentionally NOT persisted: the free path has no
    API key to fingerprint, so persisted entries could not be told apart from
    test-run artifacts and would suppress the free provider for real users.
    """
    global _FILE_STATE_LOADED
    try:
        payload = {
            name: {
                "last_error": state.get("last_error"),
                "cooldown_until": state.get("cooldown_until"),
                "key_fp": state.get("key_fp", ""),
            }
            for name, state in _PROVIDER_STATE.items()
            if name != "huggingface"
        }
        _CLOUD_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CLOUD_STATE_FILE.write_text(json.dumps(payload), encoding="utf-8")
        _FILE_STATE_LOADED = True
    except Exception as exc:  # noqa: BLE001 - state file is best-effort
        logger.debug("Could not persist cloud provider state: %s", exc)


def _load_file_state_locked() -> None:
    """Merge state-file entries into the registry (once per process). Caller holds the lock."""
    global _FILE_STATE_LOADED
    if _FILE_STATE_LOADED:
        return
    _FILE_STATE_LOADED = True
    try:
        if not _CLOUD_STATE_FILE.exists():
            return
        data = json.loads(_CLOUD_STATE_FILE.read_text(encoding="utf-8"))
        now = time.time()
        for name, entry in data.items():
            if name in _PROVIDER_STATE or not isinstance(entry, dict):
                continue
            cooldown_until = float(entry.get("cooldown_until", 0.0))
            if cooldown_until <= now:
                continue  # expired
            _PROVIDER_STATE[name] = {
                "last_error": entry.get("last_error"),
                "cooldown_until": cooldown_until,
                "key_fp": entry.get("key_fp", ""),
                "from_file": True,
            }
    except Exception as exc:  # noqa: BLE001 - state file is best-effort
        logger.debug("Could not load cloud provider state: %s", exc)


def mark_provider_unavailable(
    provider: str,
    message: str,
    key: Optional[str] = None,
    cooldown_s: Optional[float] = None,
) -> None:
    """Mark a provider as cooling down (thread-safe).

    Defaults to PROVIDER_COOLDOWN_S; the free HuggingFace path uses a shorter
    10-minute cooldown (HF_COOLDOWN_S). Persists the cooldown to the state file
    (keyed by API-key fingerprint) so process restarts do not re-pay the
    failure discovery.
    """
    with _PROVIDER_LOCK:
        _load_file_state_locked()  # merge prior entries so we don't clobber them
        _PROVIDER_STATE[provider] = {
            "last_error": message,
            "cooldown_until": time.time() + (cooldown_s or PROVIDER_COOLDOWN_S),
            "marked_at": time.time(),
            "key_fp": _key_fingerprint(key),
        }
        _persist_state_locked()


def provider_in_cooldown(provider: str, key: Optional[str] = None) -> bool:
    with _PROVIDER_LOCK:
        _load_file_state_locked()
        state = _PROVIDER_STATE.get(provider)
        if not state:
            return False
        if state.get("from_file") and state.get("key_fp") != _key_fingerprint(key):
            return False  # cooldown was learned under a different API key
        return time.time() < float(state.get("cooldown_until", 0.0))


def provider_last_error(provider: str, key: Optional[str] = None) -> Optional[str]:
    with _PROVIDER_LOCK:
        _load_file_state_locked()
        state = _PROVIDER_STATE.get(provider)
        if not state:
            return None
        if state.get("from_file") and state.get("key_fp") != _key_fingerprint(key):
            return None  # error was learned under a different API key
        return state.get("last_error")


def reset_provider_state() -> None:
    """Clear the cooldown registry, state file, and cached HF clients (tests/ops)."""
    global _FILE_STATE_LOADED
    with _PROVIDER_LOCK:
        _PROVIDER_STATE.clear()
        _FILE_STATE_LOADED = True  # nothing left to load
        _HF_CLIENTS.clear()
        _HF_SPACE_STATE.clear()
        try:
            if _CLOUD_STATE_FILE.exists():
                _CLOUD_STATE_FILE.unlink()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not remove cloud provider state file: %s", exc)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class CloudRenderEngine:
    """Dispatches sketch-to-photoreal renders to cloud image providers.

    Args:
        provider: "auto" | "huggingface" | "cloudflare" | "modelscope" |
                  "stability" | "gemini" | "openrouter" | "openai".
        config: optional dict overriding environment variables (used by tests);
                keys are checked BEFORE os.environ, so an explicit empty string
                deterministically marks a provider as unconfigured.
    """

    def __init__(self, provider: str = "auto", config: Optional[Dict[str, str]] = None) -> None:
        load_project_env()
        self.provider = (provider or "auto").lower().strip() or "auto"
        self._config: Dict[str, str] = dict(config or {})

        self.api_keys = {
            "cloudflare": self._cfg("CLOUDFLARE_API_TOKEN"),
            "modelscope": self._cfg("MODELSCOPE_API_TOKEN"),
            "stability": self._cfg("STABILITY_API_KEY"),
            "gemini": self._cfg("GEMINI_API_KEY"),
            "openrouter": self._cfg("OPENROUTER_API_KEY"),
            "openai": self._cfg("OPENAI_API_KEY"),
        }
        # Cloudflare Workers AI img2img needs BOTH the API token and the
        # account id; dormant until both land in .env (free 10k Neurons/day).
        self.cloudflare_account_id = self._cfg("CLOUDFLARE_ACCOUNT_ID")
        try:
            cf_strength = float(self._cfg("CLOUDFLARE_STRENGTH", "0.55"))
        except ValueError:
            cf_strength = 0.55
        self.cloudflare_strength = min(1.0, max(0.0, cf_strength))
        # ModelScope stays OFF unless explicitly enabled (see _render_modelscope
        # for why: REST image-edit needs a hosted image URL).
        self.modelscope_enabled = (self._cfg("MODELSCOPE_ENABLED", "false") or "false").strip().lower() not in (
            "false",
            "0",
            "no",
            "off",
        )
        # Free HuggingFace ZeroGPU path (no API key; disable with HF_ENABLED=false).
        # HF_SPACES is a comma-separated fallback chain: the first space that is
        # not on per-space cooldown gets the render. Dedicated-hardware mirrors
        # (e.g. mcp-tools/*) have NO ZeroGPU quota and are listed first.
        spaces_env = self._cfg("HF_SPACES", "")
        default_chain = "mcp-tools/FLUX.1-Kontext-Dev," + HF_DEFAULT_SPACE
        self.hf_spaces = [
            s.strip() for s in (spaces_env or default_chain).split(",") if s.strip()
        ]
        self.hf_space = self.hf_spaces[0] if self.hf_spaces else self._cfg("HF_SPACE", HF_DEFAULT_SPACE)
        self.hf_token = self._cfg("HF_TOKEN")
        self.hf_enabled = (self._cfg("HF_ENABLED", "true") or "true").strip().lower() not in (
            "false",
            "0",
            "no",
            "off",
        )
        self.models = {
            "huggingface": self.hf_spaces[0] if self.hf_spaces else self.hf_space,
            "cloudflare": self._cfg("CLOUDFLARE_IMG2IMG_MODEL", "@cf/black-forest-labs/flux-2-klein-9b"),
            "modelscope": self._cfg("MODELSCOPE_IMAGE_MODEL", "Qwen/Qwen-Image-Edit"),
            "stability": self._cfg("STABILITY_IMAGE_MODEL", "sketch"),
            "gemini": self._cfg("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image"),
            "openrouter": self._cfg("OPENROUTER_IMAGE_MODEL", "google/gemini-2.5-flash-image"),
            "openai": self._cfg("OPENAI_IMAGE_MODEL", "gpt-image-1"),
        }
        try:
            control_strength = float(self._cfg("STABILITY_CONTROL_STRENGTH", "0.7"))
        except ValueError:
            control_strength = 0.7
        self.control_strength = min(1.0, max(0.0, control_strength))
        self.cost_notes = {
            "huggingface": "HuggingFace ZeroGPU FLUX.1-Kontext (free, anonymous quota limited)",
            "cloudflare": (
                "Cloudflare Workers AI FLUX.2-klein (free tier: 10,000 Neurons/day "
                "≈ dozens of reference-image renders, ~5-15s each)"
            ),
            "modelscope": "ModelScope API-Inference Qwen-Image-Edit (free quota; dormant pending hosted-URL input)",
            "stability": "Stability AI stable-image/control/sketch (~$0.05/image, billing/credits required)",
            "gemini": "Gemini image API (free tier quota-gated)",
            "openrouter": "OpenRouter pay-per-use (requires account balance)",
            "openai": "OpenAI images/edits (quality=low, billed per image)",
        }

    # -- configuration ------------------------------------------------------

    def _cfg(self, key: str, default: str = "") -> str:
        """Read config dict first (even if empty), then environment."""
        if key in self._config:
            return (self._config.get(key) or "").strip()
        return (os.getenv(key) or default).strip()

    def _is_configured(self, name: str) -> bool:
        """Whether a provider is usable right now (key present / HF enabled)."""
        if name == "huggingface":
            return bool(self.hf_enabled and self.hf_space)
        if name == "cloudflare":
            return bool(self.api_keys.get("cloudflare") and self.cloudflare_account_id)
        if name == "modelscope":
            # Token alone is not enough: MODELSCOPE_ENABLED must opt in (default
            # false) because the REST image-edit path is unverified - see
            # _render_modelscope.
            return bool(self.modelscope_enabled and self.api_keys.get("modelscope"))
        return bool(self.api_keys.get(name))

    def _ordered_candidates(self) -> List[str]:
        """Providers to try, in order, filtered by configured + not cooling down."""
        if self.provider in _PROVIDER_ORDER:
            order: Tuple[str, ...] = (self.provider,)
        else:
            order = _PROVIDER_ORDER
        return [
            name
            for name in order
            if self._is_configured(name) and not provider_in_cooldown(name, self.api_keys.get(name))
        ]

    def available_provider(self) -> Optional[str]:
        """First configured provider whose key exists and is not in cooldown."""
        candidates = self._ordered_candidates()
        return candidates[0] if candidates else None

    def probe(self) -> Dict[str, Dict[str, Any]]:
        """Availability report for every provider in _PROVIDER_ORDER."""
        report: Dict[str, Dict[str, Any]] = {}
        for name in _PROVIDER_ORDER:
            key = self.api_keys.get(name)
            configured = self._is_configured(name)
            if not configured:
                status = "missing_key"
            elif provider_in_cooldown(name, key):
                status = "cooldown"
            else:
                status = "ready"
            entry: Dict[str, Any] = {
                "configured": configured,
                "last_error": provider_last_error(name, key) if configured else None,
                "status": status,
            }
            if name == "huggingface":
                entry["note"] = (
                    "Free ZeroGPU path (no key); anonymous quota is limited - set HF_TOKEN to raise it"
                )
            elif name == "cloudflare":
                entry["note"] = (
                    "Free tier: 10,000 Neurons/day (≈20-30 images, no credit card); "
                    "needs CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID"
                )
            elif name == "modelscope":
                entry["note"] = (
                    "Dormant skeleton: set MODELSCOPE_API_TOKEN + MODELSCOPE_ENABLED=true; "
                    "render raises until hosted-image-URL input is supported (needs verification)"
                    if configured
                    else "Disabled by default (MODELSCOPE_ENABLED=false); needs MODELSCOPE_API_TOKEN; "
                    "Qwen-Image-Edit REST requires a hosted image_url (unverified inline base64)"
                )
            report[name] = entry
        return report

    # -- prompt composition --------------------------------------------------

    @staticmethod
    def compose_prompt(prompt: str, negative_prompt: str = "") -> str:
        """Fold the negative prompt into the positive prompt text.

        All three supported APIs are instruction-style, so the negative is
        folded in as an explicit "Avoid: ..." directive.
        """
        prompt = (prompt or "").strip()
        negative = (negative_prompt or "").strip()
        if not negative:
            return prompt
        if prompt.endswith("."):
            return f"{prompt} Avoid: {negative}"
        return f"{prompt}. Avoid: {negative}"

    @staticmethod
    def _prepare_sketch(
        sketch: Union[Image.Image, np.ndarray, str, Path, bytes],
        max_side: int = 1024,
    ) -> Image.Image:
        """Normalize the sketch to RGB PIL and downscale to <= max_side (LANCZOS)."""
        if isinstance(sketch, Image.Image):
            img = sketch
        elif isinstance(sketch, (str, Path)):
            img = Image.open(sketch)
        elif isinstance(sketch, (bytes, bytearray)):
            img = Image.open(io.BytesIO(sketch))
        elif isinstance(sketch, np.ndarray):
            img = Image.fromarray(sketch)
        else:
            raise CloudEngineError(f"Unsupported sketch type: {type(sketch)!r}")
        img = img.convert("RGB")
        width, height = img.size
        longest = max(width, height)
        if longest > max_side:
            scale = max_side / float(longest)
            img = img.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.LANCZOS)
        return img

    # -- public render -------------------------------------------------------

    def render(
        self,
        sketch: Union[Image.Image, np.ndarray, str, Path, bytes],
        prompt: str,
        negative_prompt: str = "",
        size: int = 1024,
        control_strength: Optional[float] = None,
    ) -> CloudRenderResult:
        """Render the sketch via the first working cloud provider.

        Walks the provider chain; on 402/429/403 marks the provider cooldown and
        tries the next candidate. Raises CloudUnavailableError when no provider
        is configured/available or when every candidate fails.

        Args:
            control_strength: Stability sketch-control strength (0.0-1.0); the
                pipeline maps its conditioning_scale here. Falls back to the
                STABILITY_CONTROL_STRENGTH env value (default 0.7).
        """
        composed_prompt = self.compose_prompt(prompt, negative_prompt)
        prepared = self._prepare_sketch(sketch, max_side=min(1024, max(256, int(size))))
        raw_strength = control_strength  # None when the caller/pipeline did not specify one
        if control_strength is None:
            control_strength = self.control_strength

        candidates = self._ordered_candidates()
        if not candidates:
            raise CloudUnavailableError(
                "No cloud image provider available (missing API key or provider in cooldown)."
            )

        last_error: Optional[CloudEngineError] = None
        engine_tags = {
            "huggingface": "hf-flux-kontext",
            "cloudflare": "cloudflare-img2img",
            "modelscope": "modelscope-qwen-edit",
        }
        for name in candidates:
            render_fn = {
                "huggingface": self._render_huggingface,
                "cloudflare": self._render_cloudflare,
                "modelscope": self._render_modelscope,
                "stability": self._render_stability,
                "gemini": self._render_gemini,
                "openrouter": self._render_openrouter,
                "openai": self._render_openai,
            }[name]
            cooldown_s = HF_COOLDOWN_S if name == "huggingface" else PROVIDER_COOLDOWN_S
            started = time.time()
            try:
                if name == "huggingface":
                    image = render_fn(prepared, prompt, control_strength)
                elif name == "stability":
                    # Stability takes the negative prompt as a dedicated form
                    # field - no "Avoid:" folding.
                    image = render_fn(prepared, prompt, negative_prompt, control_strength)
                elif name == "cloudflare":
                    # Workers AI img2img has no negative-prompt field; the
                    # negative is folded in via compose_prompt. An unspecified
                    # strength falls back to CLOUDFLARE_STRENGTH (default 0.55),
                    # an explicit conditioning_scale is mapped via
                    # _map_cloudflare_strength.
                    image = render_fn(prepared, composed_prompt, size, raw_strength)
                else:
                    image = render_fn(prepared, composed_prompt, size)
            except CloudRateLimitedError as exc:
                # 429 rate limits are per-MINUTE windows (e.g. Workers AI
                # bursts) — a long cooldown here makes the app fall back to
                # the local preview for a quarter of an hour over a blip.
                # Cloudflare 429s especially clear within seconds. Honor the
                # provider's own "(retry after Ns)" hint when present, and
                # escalate on repeated failures within a short window.
                cooldown = 25 if name == "cloudflare" else cooldown_s
                marker = "retry after "
                if marker in str(exc):
                    try:
                        cooldown = max(10, min(120, int(str(exc).split(marker, 1)[1].split("s)")[0])))
                    except (ValueError, IndexError):
                        pass
                elif getattr(self, "_cf_rl_count", 0) >= 1:
                    # second+ consecutive CF 429: upstream window is longer —
                    # escalate so we stop hammering (1 -> 25s, 2 -> 60s, 3+ -> 120s)
                    self._cf_rl_count = min(3, getattr(self, "_cf_rl_count", 0) + 1)
                    cooldown = {2: 60, 3: 120}.get(self._cf_rl_count, 120)
                    logger.warning("Cloudflare still rate limited (attempt %s) — escalating cooldown to %ss", self._cf_rl_count, cooldown)
                mark_provider_unavailable(name, str(exc), key=self.api_keys.get(name), cooldown_s=cooldown)
                logger.warning("Cloud provider '%s' rate limited: %s (cooldown %ss)", name, exc, cooldown)
                last_error = exc
                continue
            except CloudUnavailableError as exc:
                mark_provider_unavailable(name, str(exc), key=self.api_keys.get(name), cooldown_s=cooldown_s)
                logger.warning("Cloud provider '%s' unavailable: %s", name, exc)
                last_error = exc
                continue
            except CloudEngineError as exc:
                mark_provider_unavailable(name, str(exc), key=self.api_keys.get(name), cooldown_s=cooldown_s)
                logger.warning("Cloud provider '%s' failed: %s", name, exc)
                last_error = exc
                continue
            except Exception as exc:  # noqa: BLE001 - normalize unexpected errors
                message = f"{name} unexpected error: {type(exc).__name__}: {exc}"
                mark_provider_unavailable(name, message, key=self.api_keys.get(name), cooldown_s=cooldown_s)
                logger.exception("Cloud render provider '%s' crashed", name)
                last_error = CloudEngineError(message)
                continue

            latency = time.time() - started
            used_model = self.models[name]
            if name == "huggingface" and getattr(self, "last_hf_space", None):
                used_model = self.last_hf_space  # the space that actually served
            if name == "cloudflare" and getattr(self, "last_cf_model", None):
                used_model = self.last_cf_model  # the model that actually served
                self._cf_rl_count = 0  # success resets the rate-limit escalation
            return CloudRenderResult(
                image=image,
                provider=name,
                model=used_model,
                latency_s=round(latency, 3),
                cost_note=self.cost_notes[name],
                engine_tag=engine_tags.get(name, name),
            )

        raise last_error or CloudUnavailableError("All cloud providers failed.")

    # -- provider adapters ----------------------------------------------------

    def _raise_for_status(self, provider: str, resp: httpx.Response) -> None:
        """Map HTTP errors onto CloudEngineError subclasses with short messages.

        A 429 from a burst-prone free provider (Workers AI) carries its retry
        window in the Retry-After header; that value is embedded in the message
        so render()'s catch block can size the provider cooldown accordingly
        instead of guessing.
        """
        if resp.status_code == 200:
            return
        body = ""
        try:
            body = resp.text[:300]
        except Exception:  # noqa: BLE001
            pass
        if resp.status_code == 429:
            retry_after = None
            try:
                retry_after = (resp.headers or {}).get("retry-after")
            except Exception:  # noqa: BLE001 - header access is best-effort
                retry_after = None
            suffix = f" (retry after {retry_after}s)" if retry_after else ""
            if "quota" in body.lower():
                raise CloudUnavailableError(f"{provider} quota exhausted (429).{suffix}")
            raise CloudRateLimitedError(f"{provider} rate limited (429).{suffix}")
        if resp.status_code in (401, 402, 403):
            raise CloudUnavailableError(f"{provider} unavailable (HTTP {resp.status_code}): {body[:160]}")
        raise CloudEngineError(f"{provider} HTTP {resp.status_code}: {body[:160]}")

    def _render_huggingface(
        self,
        sketch: Image.Image,
        prompt: str,
        control_strength: Optional[float] = None,
    ) -> Image.Image:
        """Free HuggingFace render via gradio_client over a SPACE FALLBACK CHAIN.

        Each entry of ``HF_SPACES`` is tried in order; a space that fails (ZeroGPU
        quota exhausted, busy, connection) gets its own short cooldown and the
        next space takes the render. Dedicated-hardware mirrors (mcp-tools/*)
        have no ZeroGPU quota, so they are tried first by default. Writes the
        sketch to a temp PNG, calls the space's /infer endpoint (seed=0,
        steps=16 — fewer GPU-seconds so the free rolling quota stretches
        further), and parses the returned (filepath, seed) tuple — the file may
        be .webp, opened via PIL -> RGB.
        """
        if not (self.hf_enabled or not self.hf_spaces):
            if not self.hf_enabled:
                raise CloudUnavailableError("huggingface disabled (HF_ENABLED=false).")
        if not self.hf_spaces:
            raise CloudUnavailableError("huggingface disabled (HF_SPACES unset).")
        guidance = self._map_guidance(control_strength if control_strength is not None else self.control_strength)
        try:
            from gradio_client import Client, handle_file  # noqa: N813 - deferred import
        except Exception as import_err:  # noqa: BLE001 - optional dependency
            raise CloudUnavailableError(
                f"huggingface unavailable (gradio_client import failed: {import_err})"
            ) from import_err

        tmp_path = ""
        last_error: Optional[CloudEngineError] = None
        try:
            # gradio upload requires a real file on disk
            fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix="packvision_sketch_")
            with os.fdopen(fd, "wb") as handle:
                buffer = io.BytesIO()
                sketch.save(buffer, format="PNG")
                handle.write(buffer.getvalue())

            now = time.time()
            result = None
            used_space = None
            for space in self.hf_spaces:
                cooldown_until = _HF_SPACE_STATE.get(space, 0)
                if now < cooldown_until:
                    logger.info(
                        "HF space '%s' on cooldown for %.0fs more; trying next.",
                        space,
                        cooldown_until - now,
                    )
                    continue
                try:
                    client = _get_hf_client(space, self.hf_token, client_factory=Client)
                    future = _HF_EXECUTOR.submit(
                        client.predict,
                        input_image=handle_file(tmp_path),
                        prompt=prompt,
                        seed=0,
                        randomize_seed=True,
                        guidance_scale=guidance,
                        steps=16,
                        api_name="/infer",
                    )
                    try:
                        result = future.result(timeout=HF_PREDICT_TIMEOUT_S)
                        used_space = space
                        break
                    except TimeoutError as timeout_err:  # concurrent.futures.TimeoutError
                        raise CloudRateLimitedError(
                            f"huggingface render timed out after {HF_PREDICT_TIMEOUT_S:.0f}s (queue on {space})"
                        ) from timeout_err
                except CloudEngineError as space_err:
                    # mark THIS space only, then fall through to the next space
                    _HF_SPACE_STATE[space] = time.time() + HF_COOLDOWN_S
                    logger.warning("HF space '%s' failed: %s; trying next space.", space, space_err)
                    last_error = space_err
                    continue
                except Exception as space_exc:  # noqa: BLE001 - normalize + per-space cooldown
                    space_err = self._map_hf_error(space_exc)
                    _HF_SPACE_STATE[space] = time.time() + HF_COOLDOWN_S
                    logger.warning("HF space '%s' failed: %s; trying next space.", space, space_err)
                    last_error = space_err
                    continue
            if result is None or used_space is None:
                if last_error is not None:
                    # Surface the real cause (e.g. the ZeroGPU quota error) so
                    # callers and the UI can show exactly why the free tier
                    # was unavailable.
                    raise last_error
                raise CloudUnavailableError("huggingface: no space available (all spaces cooling down)")
        except CloudEngineError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize gradio/httpx errors
            raise self._map_hf_error(exc) from exc
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        image = self._parse_hf_result(result)
        self.last_hf_space = used_space
        return image

    @staticmethod
    def _map_guidance(control_strength: Optional[float]) -> float:
        """Map ControlNet-style conditioning_scale (0..1.5) to Kontext guidance_scale."""
        try:
            cs = float(control_strength)
        except (TypeError, ValueError):
            cs = 0.8
        return min(5.0, max(1.0, 1.5 + cs * 3.0))

    @staticmethod
    def _map_hf_error(exc: Exception) -> CloudEngineError:
        """Classify gradio/ZeroGPU failures for the cooldown registry."""
        message = f"{type(exc).__name__}: {str(exc)[:200]}"
        low = message.lower()
        if any(keyword in low for keyword in ("quota", "gpu", "busy", "connection", "timed out", "rate")):
            return CloudRateLimitedError(f"huggingface ZeroGPU busy/quota: {message}")
        return CloudEngineError(f"huggingface render failed: {message}")

    @staticmethod
    def _parse_hf_result(result: Any) -> Image.Image:
        """Parse the (filepath, seed) tuple returned by /infer - file may be .webp."""
        out_path = None
        if isinstance(result, (list, tuple)):
            for item in result:
                if isinstance(item, str) and Path(item).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                    out_path = item
                    break
        elif isinstance(result, str):
            out_path = result
        if not out_path:
            raise CloudEngineError(f"huggingface returned no image path in result: {str(result)[:140]}")
        try:
            with open(out_path, "rb") as handle:
                data = handle.read()
            image = Image.open(io.BytesIO(data)).convert("RGB")
        except OSError as read_err:
            raise CloudEngineError(f"huggingface result unreadable: {read_err}") from read_err
        try:
            os.unlink(out_path)  # gradio cache file; already fully loaded
        except OSError:
            pass
        return image

    # -- Cloudflare Workers AI (free img2img tier, key-ready) -----------------

    @staticmethod
    def _map_cloudflare_strength(control_strength: Optional[float]) -> float:
        """Map ControlNet-style conditioning_scale (0..1.5) to SD img2img strength.

        strength = min(0.8, 0.35 + conditioning_scale * 0.35) - keeps strength
        under 0.8 so the sketch layout stays recognizable (SD1.5 drifts badly
        at high strengths). None falls back to the default conditioning scale.
        """
        try:
            cs = float(control_strength)
        except (TypeError, ValueError):
            cs = 0.7  # engine default conditioning_scale
        return min(0.8, max(0.0, 0.35 + cs * 0.35))

    def _render_cloudflare(
        self,
        sketch: Image.Image,
        prompt: str,
        size: int,
        control_strength: Optional[float] = None,
    ) -> Image.Image:
        """Cloudflare Workers AI - second FREE tier (after HuggingFace).

        Free plan: 10,000 Neurons/day, no credit card. Primary model is
        FLUX.2 [klein] 9B which ACCEPTS A REFERENCE IMAGE (multi-reference
        editing) and serves via MULTIPART form upload — photoreal sketch
        renders in ~4s, far below the ZeroGPU quality/time tradeoffs. klein-4b
        and the classic SD1.5-img2img (JSON + data-URI, strength-mapped) are
        per-model fallbacks; a model that 403s/400s for the account is skipped
        without killing the provider.
        """
        token = self.api_keys["cloudflare"]
        account = self.cloudflare_account_id
        if not (token and account):
            raise CloudUnavailableError(
                "cloudflare: CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID are both required."
            )
        primary = self.models["cloudflare"]
        model_chain: List[str] = []
        for candidate in (
            primary,
            "@cf/black-forest-labs/flux-2-klein-4b",
            "@cf/runwayml/stable-diffusion-v1-5-img2img",
        ):
            if candidate not in model_chain:
                model_chain.append(candidate)

        last_error: Optional[CloudEngineError] = None
        for model in model_chain:
            url = CLOUDFLARE_RUN_URL.format(account_id=account, model=model)
            try:
                if "img2img" in model:
                    # Classic SD1.5 img2img: JSON body, base64 data-URI image,
                    # strength-mapped conditioning (SD drifts at high strength).
                    if control_strength is None:
                        strength = self.cloudflare_strength
                    else:
                        strength = self._map_cloudflare_strength(control_strength)
                    buffer = io.BytesIO()
                    sketch.save(buffer, format="PNG")
                    data_uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
                    resp = httpx.post(
                        url,
                        json={
                            "prompt": prompt,
                            "image": data_uri,
                            "strength": round(strength, 3),
                            "num_inference_steps": CLOUDFLARE_NUM_STEPS,
                        },
                        headers={"Authorization": f"Bearer {token}"},
                        timeout=CLOUD_HTTP_TIMEOUT_S,
                    )
                else:
                    # FLUX.2 klein models: multipart form with the reference
                    # image file (verified live — JSON input is rejected).
                    buf = io.BytesIO()
                    sketch.save(buf, format="PNG")
                    buf.seek(0)
                    resp = httpx.post(
                        url,
                        data={"prompt": prompt, "num_inference_steps": str(CLOUDFLARE_NUM_STEPS)},
                        files={"image": ("sketch.png", buf.read(), "image/png")},
                        headers={"Authorization": f"Bearer {token}"},
                        timeout=CLOUD_HTTP_TIMEOUT_S,
                    )
            except httpx.HTTPError as exc:
                raise CloudEngineError(f"cloudflare request failed: {type(exc).__name__}: {exc}") from exc

            if resp.status_code == 200:
                self.last_cf_model = model  # the model that actually served
                return self._parse_cloudflare_image(resp)

            body_excerpt = resp.text[:200]
            last_error = CloudEngineError(f"cloudflare {model} HTTP {resp.status_code}: {body_excerpt}")
            if resp.status_code in (429, 402):
                # account-wide limits (neurons exhausted / plan) — cooldown the
                # whole provider via render()'s handler
                self._raise_for_status("cloudflare", resp)
            # model-specific rejection (403/400 "not allowed"/"not found") —
            # skip this model and try the next one in the chain
            logger.warning(
                "Cloudflare model '%s' rejected: HTTP %s %s; trying next model.",
                model,
                resp.status_code,
                body_excerpt,
            )
        raise last_error or CloudUnavailableError("cloudflare: all models rejected the request")

    @staticmethod
    def _parse_cloudflare_image(resp: httpx.Response) -> Image.Image:
        """Parse a Workers AI image response - shapes seen in the wild:

        1. Raw image bytes (no Accept header / binary models) -> PNG magic.
        2. API envelope with a bare data-URI string: {"success": true,
           "result": "data:image/png;base64,..."}.
        3. API envelope with nested base64: {"result": {"image": "<b64>"}}.
        """
        payload = resp.content or b""
        try:
            content_type = (resp.headers or {}).get("content-type", "")
        except Exception:  # noqa: BLE001 - header access is best-effort
            content_type = ""
        if payload[:8] == b"\x89PNG\r\n\x1a\n" or "image/" in content_type:
            try:
                return Image.open(io.BytesIO(payload)).convert("RGB")
            except Exception as decode_err:  # noqa: BLE001
                raise CloudEngineError(f"cloudflare image decode failed: {decode_err}") from decode_err
        try:
            data = resp.json()
        except Exception as parse_err:  # noqa: BLE001
            raise CloudEngineError(f"cloudflare returned invalid JSON: {parse_err}") from parse_err
        result = data.get("result") if isinstance(data, dict) else None
        candidates: List[str] = []
        if isinstance(result, str):
            candidates.append(result)
        elif isinstance(result, dict):
            image = result.get("image")
            if isinstance(image, str):
                candidates.append(image)
        for candidate in candidates:
            if candidate.startswith("data:image"):
                candidate = candidate.partition(",")[2]
            try:
                return Image.open(io.BytesIO(base64.b64decode(candidate))).convert("RGB")
            except Exception:  # noqa: BLE001 - try the next shape
                continue
        raise CloudEngineError("cloudflare returned no image payload.")

    # -- ModelScope API-Inference (Qwen-Image-Edit, dormant skeleton) ---------

    def _render_modelscope(self, sketch: Image.Image, prompt: str, size: int) -> Image.Image:
        """ModelScope API-Inference Qwen/Qwen-Image-Edit - HONEST SKELETON.

        The documented REST call is OpenAI-images-compatible::

            POST https://api-inference.modelscope.cn/v1/images/generations
            Authorization: Bearer <MODELSCOPE_API_TOKEN>
            {"model": "Qwen/Qwen-Image-Edit", "prompt": "...",
             "image_url": "https://..."}

        BUT ``image_url`` must be a PUBLICLY HOSTED http(s) URL per the docs -
        inline base64 data-URI input is not documented and could not be
        verified (the official ``modelscope`` SDK is not installed here
        either). PackVision sketches are local PIL images with nowhere to host
        them, so this provider intentionally raises a clear error instead of
        firing a request that would fail; it stays disabled by default via
        MODELSCOPE_ENABLED=false. If hosted-input support ever lands (image
        upload endpoint), wire the payload above in and flip the default.
        """
        if not (self.modelscope_enabled and self.api_keys.get("modelscope")):
            raise CloudUnavailableError(
                "modelscope: set MODELSCOPE_API_TOKEN + MODELSCOPE_ENABLED=true (disabled by default)."
            )
        raise CloudUnavailableError(
            "modelscope image-edit requires a publicly hosted image URL "
            "(Qwen/Qwen-Image-Edit REST /v1/images/generations takes image_url=https://...; "
            "inline base64 data URIs are not supported/verified) - provider left disabled "
            "until hosted-input support is added."
        )

    def _render_stability(
        self,
        sketch: Image.Image,
        prompt: str,
        negative_prompt: str,
        control_strength: Optional[float] = None,
    ) -> Image.Image:
        """Stability AI stable-image/control/sketch: sketch -> photoreal render.

        Multipart form: image (file bytes), prompt, negative_prompt (optional),
        control_strength (0.0-1.0), output_format=png. Auth via
        ``Authorization: Bearer {STABILITY_API_KEY}`` with ``Accept: image/*``.
        Returns raw image bytes (content-type image/*) - NOT JSON.
        """
        key = self.api_keys["stability"]
        if not key:
            raise CloudUnavailableError("stability: no API key configured.")
        strength = self.control_strength if control_strength is None else control_strength
        try:
            strength = min(1.0, max(0.0, float(strength)))
        except (TypeError, ValueError):
            strength = self.control_strength

        buffer = io.BytesIO()
        sketch.save(buffer, format="PNG")
        buffer.seek(0)
        data = {
            "prompt": prompt,
            "control_strength": f"{strength:.2f}",
            "output_format": "png",
        }
        negative = (negative_prompt or "").strip()
        if negative:
            data["negative_prompt"] = negative
        files = {"image": ("sketch.png", buffer, "image/png")}
        headers = {"Authorization": f"Bearer {key}", "Accept": "image/*"}
        try:
            resp = httpx.post(
                STABILITY_SKETCH_URL,
                headers=headers,
                data=data,
                files=files,
                timeout=CLOUD_HTTP_TIMEOUT_S,
            )
        except httpx.HTTPError as exc:
            raise CloudEngineError(f"stability request failed: {type(exc).__name__}: {exc}") from exc
        self._raise_for_status("stability", resp)

        content_type = ""
        try:
            content_type = (resp.headers or {}).get("content-type", "")
        except Exception:  # noqa: BLE001 - header access is best-effort
            content_type = ""
        payload = resp.content or b""
        if "application/json" in content_type:
            # Stability reports some failures as JSON even over 200.
            try:
                body = resp.json()
            except Exception:  # noqa: BLE001
                body = {}
            raise CloudEngineError(f"stability returned no image: {str(body)[:160]}")
        if payload[:8] == b"\x89PNG\r\n\x1a\n" or "image" in content_type:
            try:
                return Image.open(io.BytesIO(payload)).convert("RGB")
            except Exception as decode_err:
                raise CloudEngineError(f"stability image decode failed: {decode_err}") from decode_err
        raise CloudEngineError("stability returned no image payload.")

    def _render_gemini(self, sketch: Image.Image, prompt: str, size: int) -> Image.Image:
        key = self.api_keys["gemini"]
        model = self.models["gemini"]
        buffer = io.BytesIO()
        sketch.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/png", "data": b64}},
                    ]
                }
            ],
            "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
        }
        url = f"{GEMINI_API_BASE}/{model}:generateContent"
        try:
            resp = httpx.post(url, json=payload, headers={"x-goog-api-key": key}, timeout=CLOUD_HTTP_TIMEOUT_S)
        except httpx.HTTPError as exc:
            raise CloudEngineError(f"gemini request failed: {type(exc).__name__}: {exc}") from exc
        self._raise_for_status("gemini", resp)
        try:
            data = resp.json()
        except Exception as parse_err:
            raise CloudEngineError(f"gemini returned invalid JSON: {parse_err}") from parse_err

        candidates = data.get("candidates") or []
        for candidate in candidates:
            for part in (candidate.get("content", {}) or {}).get("parts", []) or []:
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    return Image.open(io.BytesIO(base64.b64decode(inline["data"]))).convert("RGB")
        raise CloudEngineError("gemini returned no image payload.")

    def _render_openrouter(self, sketch: Image.Image, prompt: str, size: int) -> Image.Image:
        key = self.api_keys["openrouter"]
        model = self.models["openrouter"]
        buffer = io.BytesIO()
        sketch.save(buffer, format="PNG")
        data_uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
            "modalities": ["image", "text"],
        }
        headers = {
            "Authorization": f"Bearer {key}",
            "HTTP-Referer": "https://packvision.local",
            "X-Title": "PackVision AI",
        }
        try:
            resp = httpx.post(OPENROUTER_CHAT_URL, json=payload, headers=headers, timeout=CLOUD_HTTP_TIMEOUT_S)
        except httpx.HTTPError as exc:
            raise CloudEngineError(f"openrouter request failed: {type(exc).__name__}: {exc}") from exc
        self._raise_for_status("openrouter", resp)
        try:
            data = resp.json()
        except Exception as parse_err:
            raise CloudEngineError(f"openrouter returned invalid JSON: {parse_err}") from parse_err

        message = (data.get("choices") or [{}])[0].get("message", {}) or {}
        for image_entry in message.get("images") or []:
            url = (image_entry.get("image_url") or {}).get("url", "")
            if url.startswith("data:image"):
                _, _, b64 = url.partition(",")
                return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        # Some models put a hosted URL instead of a data URI - fetch it.
        for image_entry in message.get("images") or []:
            url = (image_entry.get("image_url") or {}).get("url", "")
            if url.startswith("http"):
                try:
                    dl = httpx.get(url, timeout=CLOUD_HTTP_TIMEOUT_S)
                except httpx.HTTPError as exc:
                    raise CloudEngineError(f"openrouter image download failed: {exc}") from exc
                if dl.status_code == 200:
                    return Image.open(io.BytesIO(dl.content)).convert("RGB")
        raise CloudEngineError("openrouter returned no image payload.")

    def _render_openai(self, sketch: Image.Image, prompt: str, size: int) -> Image.Image:
        key = self.api_keys["openai"]
        model = self.models["openai"]
        # gpt-image-1 supported sizes: 1024x1024, 1536x1024, 1024x1536, auto.
        size_str = "1024x1024"
        buffer = io.BytesIO()
        sketch.save(buffer, format="PNG")
        buffer.seek(0)
        files = {"image": ("sketch.png", buffer, "image/png")}
        data = {
            "prompt": prompt,
            "model": model,
            "size": size_str,
            "quality": "low",
            "n": "1",
        }
        headers = {"Authorization": f"Bearer {key}"}
        try:
            resp = httpx.post(
                OPENAI_IMAGES_EDITS_URL,
                headers=headers,
                data=data,
                files=files,
                timeout=CLOUD_HTTP_TIMEOUT_S,
            )
        except httpx.HTTPError as exc:
            raise CloudEngineError(f"openai request failed: {type(exc).__name__}: {exc}") from exc
        self._raise_for_status("openai", resp)
        try:
            payload = resp.json()
        except Exception as parse_err:
            raise CloudEngineError(f"openai returned invalid JSON: {parse_err}") from parse_err

        for item in payload.get("data") or []:
            b64 = item.get("b64_json")
            if b64:
                return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        raise CloudEngineError("openai returned no image payload.")
