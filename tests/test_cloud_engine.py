"""Tests for the PackVision AI hybrid cloud engine (src/cloud_engine.py, src/vlm_client.py)
and the pipeline dispatch integration (src/pipeline.py).

All HTTP traffic is mocked via monkeypatch - NO network calls are performed.
Run: .venv/Scripts/python.exe -m pytest tests/test_cloud_engine.py -v
"""
from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

import src.cloud_engine as cloud_engine
import src.vlm_client as vlm_client
from src.cloud_engine import (
    CloudEngineError,
    CloudRateLimitedError,
    CloudRenderEngine,
    CloudRenderResult,
    CloudUnavailableError,
    reset_provider_state,
)
from src.vlm_client import ask_vlm, clear_cache, detect_vlm_providers, vlm_available


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class FakeResponse:
    """Minimal httpx.Response stand-in (JSON and binary responses)."""

    def __init__(
        self,
        status_code: int = 200,
        payload: dict | None = None,
        text: str | None = None,
        content: bytes | None = None,
        headers: dict | None = None,
    ):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        if content is not None:
            self.content = content
            self.text = text if text is not None else ""
        else:
            self.text = text if text is not None else json.dumps(self._payload)
            self.content = self.text.encode("utf-8")
        self.headers = headers if headers is not None else {"content-type": "application/json"}

    def json(self) -> dict:
        return self._payload


def png_bytes(width: int = 16, height: int = 16, color=(180, 40, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def png_b64(width: int = 16, height: int = 16) -> str:
    return base64.b64encode(png_bytes(width, height)).decode("ascii")


def box_sketch(size: int = 128) -> Image.Image:
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([size // 4, size // 4, size * 3 // 4, size * 3 // 4], outline=(0, 0, 0), width=3)
    return img


def openrouter_image_payload(b64: str) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": "here is your render",
                    "images": [{"image_url": {"url": f"data:image/png;base64,{b64}"}}],
                }
            }
        ]
    }


def gemini_image_payload(b64: str) -> dict:
    return {
        "candidates": [
            {"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": b64}}]}}
        ]
    }


def gemini_text_payload(text: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


@pytest.fixture(autouse=True)
def _clean_state():
    # Reset BEFORE each test only: the final marker test intentionally leaves the
    # module-level provider cooldowns warm so the backend suite that follows in
    # the same pytest process (see DoD command) skips cloud network attempts.
    reset_provider_state()
    clear_cache()
    yield
    clear_cache()


_HERMETIC_ENV_KEYS = (
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "STABILITY_API_KEY",
    "STABILITY_IMAGE_MODEL",
    "STABILITY_CONTROL_STRENGTH",
    "HF_ENABLED",
    "HF_SPACE",
    "HF_TOKEN",
    "VLM_PROVIDER",
    "GEMINI_VLM_MODEL",
    "OPENROUTER_VLM_MODEL",
    "OPENAI_VLM_MODEL",
    "GEMINI_IMAGE_MODEL",
    "OPENROUTER_IMAGE_MODEL",
    "OPENAI_IMAGE_MODEL",
    "GEMINI_VLM_THINKING_BUDGET",
    "VLM_TIMEOUT_S",
    "VLM_TOTAL_TIMEOUT_S",
    "RENDER_ENGINE",
    "RENDER_ENGINE_CLOUD_PROVIDER",
    "CLOUD_IMAGE_PROVIDER",
    "CLOUDFLARE_API_TOKEN",
    "CLOUDFLARE_ACCOUNT_ID",
    "CLOUDFLARE_IMG2IMG_MODEL",
    "CLOUDFLARE_STRENGTH",
    "MODELSCOPE_API_TOKEN",
    "MODELSCOPE_ENABLED",
    "MODELSCOPE_IMAGE_MODEL",
)


@pytest.fixture(autouse=True)
def _hf_disabled_by_default(monkeypatch):
    """Keep the free HuggingFace provider out of tests unless a test opts in
    (HF is key-less, so it would otherwise be 'configured' everywhere)."""
    monkeypatch.setenv("HF_ENABLED", "false")
    yield


@pytest.fixture
def fake_gradio(monkeypatch, tmp_path):
    """Patch gradio_client.Client / handle_file with in-memory fakes (no network)."""
    import gradio_client

    out_path = tmp_path / "hf_output.webp"  # .webp name on purpose; PIL sniffs content
    png_bytes(24, 24, (30, 144, 255))

    state = {"instances": 0, "predicts": 0, "last_kwargs": None, "error": None, "files": []}

    class FakeClient:
        def __init__(self, space, token=None, verbose=False, **kwargs):
            state["instances"] += 1
            state["space"] = space
            state["token"] = token
            self.space = space

        def predict(self, **kwargs):
            state["predicts"] += 1
            state["last_kwargs"] = dict(kwargs)
            if state.get("error"):
                raise state["error"]
            out_path.write_bytes(png_bytes(24, 24, (30, 144, 255)))  # (re)create output file
            return (str(out_path), 42)

    def fake_handle_file(path):
        state["files"].append(str(path))
        return {"path": str(path), "meta": {"_type": "gradio.FileData"}}

    monkeypatch.setattr(gradio_client, "Client", FakeClient)
    monkeypatch.setattr(gradio_client, "handle_file", fake_handle_file)
    return state, out_path


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    """Keep the real project .env out of the tests: engines see only explicit
    config dicts / explicitly monkeypatch-set env vars."""
    monkeypatch.setattr(vlm_client, "_ENV_LOADED", True)  # block .env loading
    for key in _HERMETIC_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    yield


# ---------------------------------------------------------------------------
# CloudRenderEngine: provider detection & probe
# ---------------------------------------------------------------------------


def test_provider_detection_order():
    all_keys = {
        "STABILITY_API_KEY": "s",
        "GEMINI_API_KEY": "g",
        "OPENROUTER_API_KEY": "o",
        "OPENAI_API_KEY": "k",
    }
    # Stability is first (purpose-built sketch->photoreal) whenever its key is set.
    assert CloudRenderEngine(config=all_keys).available_provider() == "stability"
    assert CloudRenderEngine(
        config={"GEMINI_API_KEY": "g", "OPENROUTER_API_KEY": "o", "OPENAI_API_KEY": "k"}
    ).available_provider() == "gemini"
    assert CloudRenderEngine(config={"STABILITY_API_KEY": "s"}).available_provider() == "stability"
    assert CloudRenderEngine(
        config={"OPENROUTER_API_KEY": "o", "OPENAI_API_KEY": "k"}
    ).available_provider() == "openrouter"
    assert CloudRenderEngine(config={"OPENAI_API_KEY": "k"}).available_provider() == "openai"

    empty = CloudRenderEngine(
        config={
            "STABILITY_API_KEY": "",
            "GEMINI_API_KEY": "",
            "OPENROUTER_API_KEY": "",
            "OPENAI_API_KEY": "",
        }
    )
    assert empty.available_provider() is None
    probe = empty.probe()
    assert probe["stability"]["status"] == "missing_key"
    assert probe["gemini"]["status"] == "missing_key"
    assert probe["openrouter"]["configured"] is False
    assert probe["openai"]["status"] == "missing_key"


def test_probe_ready_status_and_models():
    engine = CloudRenderEngine(config={"GEMINI_API_KEY": "g"})
    probe = engine.probe()
    assert probe["gemini"] == {"configured": True, "last_error": None, "status": "ready"}
    assert probe["stability"]["status"] == "missing_key"
    assert probe["openrouter"]["status"] == "missing_key"
    assert probe["huggingface"]["status"] == "missing_key"  # HF_ENABLED=false in hermetic tests
    assert "ZeroGPU" in probe["huggingface"]["note"]
    assert engine.models["huggingface"] == "mcp-tools/FLUX.1-Kontext-Dev"  # chain head
    assert engine.hf_spaces == ["mcp-tools/FLUX.1-Kontext-Dev", "black-forest-labs/FLUX.1-Kontext-dev"]
    assert engine.models["stability"] == "sketch"
    assert engine.models["gemini"] == "gemini-2.5-flash-image"
    assert engine.models["openrouter"] == "google/gemini-2.5-flash-image"
    assert engine.models["openai"] == "gpt-image-1"
    assert engine.control_strength == 0.7


def test_compose_prompt_negative_folding():
    assert CloudRenderEngine.compose_prompt("a kraft carton", "") == "a kraft carton"
    assert CloudRenderEngine.compose_prompt("a kraft carton", "no text") == "a kraft carton. Avoid: no text"
    assert CloudRenderEngine.compose_prompt("a kraft carton.", "no text") == "a kraft carton. Avoid: no text"


def test_prepare_sketch_downscale():
    big = Image.new("RGB", (2048, 1024), (10, 10, 10))
    prepared = CloudRenderEngine._prepare_sketch(big, max_side=1024)
    assert prepared.size == (1024, 512)
    small = Image.new("RGB", (512, 256), (10, 10, 10))
    assert CloudRenderEngine._prepare_sketch(small).size == (512, 256)  # never upscaled


# ---------------------------------------------------------------------------
# Cooldown behaviour
# ---------------------------------------------------------------------------


def test_openrouter_402_marks_cooldown(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append(url)
        return FakeResponse(402, {"error": {"message": "requires at least $1.00 in balance"}})

    monkeypatch.setattr(cloud_engine.httpx, "post", fake_post)
    engine = CloudRenderEngine(config={"OPENROUTER_API_KEY": "o"})
    with pytest.raises(CloudUnavailableError):
        engine.render(box_sketch(), "a kraft carton", negative_prompt="blurry")

    assert calls and "openrouter.ai" in calls[0]
    probe = engine.probe()
    assert probe["openrouter"]["status"] == "cooldown"
    assert probe["openrouter"]["configured"] is True
    assert "unavailable" in (probe["openrouter"]["last_error"] or "").lower()
    assert engine.available_provider() is None

    # Cooldown expiry restores availability.
    with cloud_engine._PROVIDER_LOCK:
        cloud_engine._PROVIDER_STATE["openrouter"]["cooldown_until"] = time.time() - 1
    assert engine.available_provider() == "openrouter"
    assert engine.probe()["openrouter"]["status"] == "ready"


def test_gemini_429_rate_limit_marks_cooldown(monkeypatch):
    monkeypatch.setattr(
        cloud_engine.httpx,
        "post",
        lambda url, **kw: FakeResponse(429, {"error": {"message": "Too many requests, slow down"}}),
    )
    engine = CloudRenderEngine(config={"GEMINI_API_KEY": "g"})
    with pytest.raises(CloudRateLimitedError):
        engine.render(box_sketch(), "a kraft carton")
    assert engine.probe()["gemini"]["status"] == "cooldown"


def test_quota_429_treated_as_unavailable(monkeypatch):
    monkeypatch.setattr(
        cloud_engine.httpx,
        "post",
        lambda url, **kw: FakeResponse(429, {"error": {"type": "insufficient_quota", "message": "You exceeded your current quota"}}),
    )
    engine = CloudRenderEngine(config={"OPENAI_API_KEY": "k"})
    with pytest.raises(CloudUnavailableError):
        engine.render(box_sketch(), "a kraft carton")
    assert engine.probe()["openai"]["status"] == "cooldown"


def test_render_raises_when_no_provider():
    engine = CloudRenderEngine(config={"GEMINI_API_KEY": "", "OPENROUTER_API_KEY": "", "OPENAI_API_KEY": ""})
    with pytest.raises(CloudUnavailableError):
        engine.render(box_sketch(), "a kraft carton")


def test_auto_mode_tries_next_provider(monkeypatch):
    urls = []

    def fake_post(url, **kwargs):
        urls.append(url)
        return FakeResponse(402, {"error": {"message": "payment required"}})

    monkeypatch.setattr(cloud_engine.httpx, "post", fake_post)
    engine = CloudRenderEngine(
        config={"GEMINI_API_KEY": "g", "OPENROUTER_API_KEY": "o", "OPENAI_API_KEY": "k"}
    )
    with pytest.raises(CloudEngineError):
        engine.render(box_sketch(), "a kraft carton")

    assert sum(1 for u in urls if "generativelanguage" in u) == 1
    assert sum(1 for u in urls if "openrouter.ai" in u) == 1
    assert sum(1 for u in urls if "openai.com" in u) == 1
    probe = engine.probe()
    assert all(probe[name]["status"] == "cooldown" for name in ("gemini", "openrouter", "openai"))
    assert probe["stability"]["status"] == "missing_key"  # no key configured in this test


# ---------------------------------------------------------------------------
# Provider adapters: payload building and image parsing
# ---------------------------------------------------------------------------


def test_openrouter_data_uri_parse_and_negative_folding(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["payload"] = kwargs.get("json")
        return FakeResponse(200, openrouter_image_payload(png_b64()))

    monkeypatch.setattr(cloud_engine.httpx, "post", fake_post)
    engine = CloudRenderEngine(config={"OPENROUTER_API_KEY": "o"})
    result = engine.render(box_sketch(), "a kraft carton", negative_prompt="blurry, watermark")

    assert result.provider == "openrouter"
    assert result.model == "google/gemini-2.5-flash-image"
    assert isinstance(result.image, Image.Image)
    assert result.image.size == (16, 16)
    assert "openrouter.ai" in captured["url"]
    payload = captured["payload"]
    assert payload["modalities"] == ["image", "text"]
    content = payload["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert "Avoid: blurry, watermark" in content[0]["text"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_gemini_b64_parse(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs.get("headers")
        return FakeResponse(200, gemini_image_payload(png_b64()))

    monkeypatch.setattr(cloud_engine.httpx, "post", fake_post)
    engine = CloudRenderEngine(config={"GEMINI_API_KEY": "g"})
    result = engine.render(box_sketch(), "a kraft carton")

    assert result.provider == "gemini"
    assert result.model == "gemini-2.5-flash-image"
    assert isinstance(result.image, Image.Image)
    assert result.image.size == (16, 16)
    assert "generativelanguage.googleapis.com" in captured["url"]
    assert captured["headers"]["x-goog-api-key"] == "g"


def test_gemini_payload_contains_inline_sketch(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["payload"] = kwargs.get("json")
        return FakeResponse(200, gemini_image_payload(png_b64()))

    monkeypatch.setattr(cloud_engine.httpx, "post", fake_post)
    engine = CloudRenderEngine(config={"GEMINI_API_KEY": "g"})
    engine.render(box_sketch(), "a kraft carton")
    parts = captured["payload"]["contents"][0]["parts"]
    assert parts[0]["text"] == "a kraft carton"
    assert parts[1]["inline_data"]["mime_type"] == "image/png"
    assert parts[1]["inline_data"]["data"]
    assert captured["payload"]["generationConfig"]["responseModalities"] == ["IMAGE", "TEXT"]


def test_openai_b64_parse_multipart(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["data"] = kwargs.get("data")
        captured["files"] = kwargs.get("files")
        return FakeResponse(200, {"data": [{"b64_json": png_b64()}]})

    monkeypatch.setattr(cloud_engine.httpx, "post", fake_post)
    engine = CloudRenderEngine(config={"OPENAI_API_KEY": "k"})
    result = engine.render(box_sketch(), "a kraft carton", negative_prompt="no text")

    assert result.provider == "openai"
    assert result.model == "gpt-image-1"
    assert isinstance(result.image, Image.Image)
    assert result.image.size == (16, 16)
    assert "api.openai.com/v1/images/edits" in captured["url"]
    assert captured["data"]["model"] == "gpt-image-1"
    assert "Avoid: no text" in captured["data"]["prompt"]
    assert "image" in captured["files"]


def test_openrouter_no_image_raises_engine_error(monkeypatch):
    monkeypatch.setattr(
        cloud_engine.httpx,
        "post",
        lambda url, **kw: FakeResponse(200, {"choices": [{"message": {"content": "no image here"}}]}),
    )
    engine = CloudRenderEngine(config={"OPENROUTER_API_KEY": "o"})
    with pytest.raises(CloudEngineError):
        engine.render(box_sketch(), "a kraft carton")
    assert engine.probe()["openrouter"]["status"] == "cooldown"


# ---------------------------------------------------------------------------
# Stability AI adapter (stable-image/control/sketch)
# ---------------------------------------------------------------------------


def test_stability_sketch_binary_parse(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs.get("headers")
        captured["data"] = kwargs.get("data")
        captured["files"] = kwargs.get("files")
        return FakeResponse(200, content=png_bytes(), headers={"content-type": "image/png"})

    monkeypatch.setattr(cloud_engine.httpx, "post", fake_post)
    engine = CloudRenderEngine(config={"STABILITY_API_KEY": "s"})
    result = engine.render(box_sketch(), "a kraft carton", negative_prompt="blurry, watermark")

    assert result.provider == "stability"
    assert result.model == "sketch"
    assert isinstance(result.image, Image.Image)
    assert result.image.size == (16, 16)
    assert result.latency_s >= 0
    assert "api.stability.ai/v2beta/stable-image/control/sketch" in captured["url"]
    assert captured["headers"]["Authorization"] == "Bearer s"
    assert captured["headers"]["Accept"] == "image/*"
    data = captured["data"]
    assert data["prompt"] == "a kraft carton"
    assert "Avoid:" not in data["prompt"]  # negative sent as a dedicated field, not folded
    assert data["negative_prompt"] == "blurry, watermark"
    assert data["control_strength"] == "0.70"
    assert data["output_format"] == "png"
    assert "image" in captured["files"]
    assert "Stability" in result.cost_note and "0.05" in result.cost_note


def test_stability_control_strength_mapping(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["data"] = kwargs.get("data")
        return FakeResponse(200, content=png_bytes(), headers={"content-type": "image/png"})

    monkeypatch.setattr(cloud_engine.httpx, "post", fake_post)
    engine = CloudRenderEngine(config={"STABILITY_API_KEY": "s", "STABILITY_CONTROL_STRENGTH": "0.4"})
    assert engine.control_strength == 0.4

    engine.render(box_sketch(), "a kraft carton")
    assert captured["data"]["control_strength"] == "0.40"  # from env

    engine.render(box_sketch(), "a kraft carton", control_strength=0.9)
    assert captured["data"]["control_strength"] == "0.90"  # pipeline conditioning_scale mapping


def test_stability_402_marks_cooldown_and_falls_through_to_gemini(monkeypatch):
    urls = []

    def fake_post(url, **kwargs):
        urls.append(url)
        if "stability.ai" in url:
            return FakeResponse(402, {"error": {"message": "insufficient credits"}})
        return FakeResponse(200, gemini_image_payload(png_b64()))

    monkeypatch.setattr(cloud_engine.httpx, "post", fake_post)
    engine = CloudRenderEngine(config={"STABILITY_API_KEY": "s", "GEMINI_API_KEY": "g"})
    result = engine.render(box_sketch(), "a kraft carton")

    assert urls[0].startswith("https://api.stability.ai")  # stability tried first
    assert result.provider == "gemini"  # chain fell through after cooldown mark
    probe = engine.probe()
    assert probe["stability"]["status"] == "cooldown"
    assert probe["gemini"]["status"] == "ready"


def test_stability_429_rate_limit_marks_cooldown(monkeypatch):
    monkeypatch.setattr(
        cloud_engine.httpx,
        "post",
        lambda url, **kw: FakeResponse(429, {"error": {"message": "Too many requests"}}),
    )
    engine = CloudRenderEngine(config={"STABILITY_API_KEY": "s"})
    with pytest.raises(CloudRateLimitedError):
        engine.render(box_sketch(), "a kraft carton")
    assert engine.probe()["stability"]["status"] == "cooldown"


def test_stability_json_200_without_image_raises(monkeypatch):
    monkeypatch.setattr(
        cloud_engine.httpx,
        "post",
        lambda url, **kw: FakeResponse(
            200, payload={"error": {"message": "unexpected"}}, headers={"content-type": "application/json"}
        ),
    )
    engine = CloudRenderEngine(config={"STABILITY_API_KEY": "s"})
    with pytest.raises(CloudEngineError):
        engine.render(box_sketch(), "a kraft carton")


# ---------------------------------------------------------------------------
# HuggingFace ZeroGPU provider (free FLUX.1-Kontext path, mocked gradio_client)
# ---------------------------------------------------------------------------


def test_hf_provider_ordering_first(monkeypatch):
    all_keys = {
        "STABILITY_API_KEY": "s",
        "GEMINI_API_KEY": "g",
        "OPENROUTER_API_KEY": "o",
        "OPENAI_API_KEY": "k",
    }
    monkeypatch.setenv("HF_ENABLED", "true")
    assert CloudRenderEngine(config=all_keys).available_provider() == "huggingface"

    monkeypatch.setenv("HF_ENABLED", "false")
    engine = CloudRenderEngine(config=all_keys)
    assert engine.available_provider() == "stability"
    assert engine._is_configured("huggingface") is False

    # forced paid provider (CLOUD_IMAGE_PROVIDER override) stays reachable while HF is enabled
    monkeypatch.setenv("HF_ENABLED", "true")
    assert CloudRenderEngine(provider="openai", config=all_keys).available_provider() == "openai"


def test_hf_guidance_mapping():
    f = CloudRenderEngine._map_guidance
    assert abs(f(0.0) - 1.5) < 1e-9
    assert abs(f(0.8) - 3.9) < 1e-9
    assert f(1.5) == 5.0  # clamped high
    assert f(-1.0) == 1.0  # clamped low
    assert abs(f(None) - 3.9) < 1e-9  # default conditioning_scale 0.8


def test_hf_render_tuple_webp_parse_and_client_cache(fake_gradio, monkeypatch):
    state, out_path = fake_gradio
    monkeypatch.setenv("HF_ENABLED", "true")
    engine = CloudRenderEngine()  # HF_SPACE unset -> default FLUX.1-Kontext-dev space

    first = engine.render(box_sketch(), "a kraft carton", control_strength=0.8)
    assert abs(state["last_kwargs"]["guidance_scale"] - 3.9) < 1e-9  # mapped from conditioning_scale 0.8
    second = engine.render(box_sketch(), "another carton")
    assert abs(state["last_kwargs"]["guidance_scale"] - 3.6) < 1e-9  # engine default cs 0.7 -> 3.6

    assert first.provider == "huggingface"
    assert first.engine_tag == "hf-flux-kontext"
    assert first.model == "mcp-tools/FLUX.1-Kontext-Dev"  # chain head: dedicated mirror, no ZeroGPU quota
    assert isinstance(first.image, Image.Image)
    assert first.image.mode == "RGB"  # webp opened via PIL and converted
    assert first.image.size == (24, 24)
    assert "ZeroGPU" in first.cost_note
    assert second.provider == "huggingface"
    assert state["instances"] == 1  # Client cached per (space, token) across renders
    assert state["predicts"] == 2
    assert state["token"] is None  # anonymous (no HF_TOKEN)
    kwargs = state["last_kwargs"]
    assert kwargs["api_name"] == "/infer"
    assert kwargs["steps"] == 16  # fewer GPU-seconds so the free quota stretches further
    assert kwargs["seed"] == 0
    assert kwargs["randomize_seed"] is True
    assert "carton" in kwargs["prompt"]
    assert state["files"] and state["files"][0].endswith(".png")  # temp sketch file
    assert not out_path.exists()  # gradio cache file removed after loading


def test_hf_quota_exhaustion_marks_cooldown_and_falls_through(fake_gradio, monkeypatch):
    state, _ = fake_gradio
    monkeypatch.setenv("HF_ENABLED", "true")
    state["error"] = RuntimeError("You have exceeded your GPU quota (ZeroGPU). Try again later.")

    urls = []

    def fake_post(url, **kwargs):
        urls.append(url)
        return FakeResponse(200, gemini_image_payload(png_b64()))

    monkeypatch.setattr(cloud_engine.httpx, "post", fake_post)
    engine = CloudRenderEngine(config={"GEMINI_API_KEY": "g"})
    result = engine.render(box_sketch(), "a kraft carton")
    assert result.provider == "gemini"  # HF cooled down, chain fell through
    assert urls and all("generativelanguage" in u for u in urls)  # no HF HTTP calls, gemini served
    probe = engine.probe()
    assert probe["huggingface"]["status"] == "cooldown"
    assert "quota" in (probe["huggingface"]["last_error"] or "").lower()
    assert probe["gemini"]["status"] == "ready"

    engine_hf_only = CloudRenderEngine()  # no other provider configured
    cloud_engine.reset_provider_state()  # clear the cooldown from the render above
    with pytest.raises(CloudRateLimitedError):
        engine_hf_only.render(box_sketch(), "a kraft carton")  # quota error surfaces verbatim


def test_hf_token_passed_to_client(fake_gradio, monkeypatch):
    state, _ = fake_gradio
    monkeypatch.setenv("HF_ENABLED", "true")
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    engine = CloudRenderEngine()
    engine.render(box_sketch(), "a kraft carton")
    assert state["token"] == "hf_test_token"


# ---------------------------------------------------------------------------
# Cloudflare Workers AI provider (free img2img tier, mocked httpx)
# ---------------------------------------------------------------------------


def test_provider_order_puts_cloudflare_after_huggingface():
    """Free tiers lead the chain: HF (key-less) -> Cloudflare (free Neurons) ->
    ModelScope (dormant skeleton) -> paid providers."""
    assert cloud_engine._PROVIDER_ORDER == (
        "huggingface",
        "cloudflare",
        "modelscope",
        "stability",
        "gemini",
        "openrouter",
        "openai",
    )


def test_cloudflare_disabled_without_keys():
    engine = CloudRenderEngine(config={"CLOUDFLARE_API_TOKEN": "", "CLOUDFLARE_ACCOUNT_ID": ""})
    assert engine._is_configured("cloudflare") is False
    probe = engine.probe()
    assert probe["cloudflare"]["configured"] is False
    assert probe["cloudflare"]["status"] == "missing_key"
    assert "Neurons" in probe["cloudflare"]["note"]


def test_cloudflare_requires_token_and_account_id():
    engine = CloudRenderEngine(config={"CLOUDFLARE_API_TOKEN": "cf-token"})
    assert engine._is_configured("cloudflare") is False  # account id missing
    engine = CloudRenderEngine(config={"CLOUDFLARE_API_TOKEN": "cf-token", "CLOUDFLARE_ACCOUNT_ID": "acct"})
    assert engine._is_configured("cloudflare") is True
    assert engine.models["cloudflare"] == "@cf/black-forest-labs/flux-2-klein-9b"  # primary: reference-image editing
    assert engine.cloudflare_strength == 0.55  # CLOUDFLARE_STRENGTH default (img2img variant)


def test_cloudflare_strength_mapping():
    f = CloudRenderEngine._map_cloudflare_strength
    assert abs(f(0.0) - 0.35) < 1e-9
    assert abs(f(0.7) - 0.595) < 1e-9
    assert f(1.5) == 0.8  # clamped high
    assert f(-2.0) == 0.0  # clamped low
    assert abs(f(None) - 0.595) < 1e-9  # default conditioning_scale 0.7


def test_cloudflare_multipart_klein_request_and_result(monkeypatch):
    """Primary path: FLUX.2 klein multipart form with the reference image file."""
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs.get("headers")
        captured["data"] = kwargs.get("data")
        captured["files"] = kwargs.get("files")
        return FakeResponse(200, {"success": True, "result": "data:image/png;base64," + png_b64()})

    monkeypatch.setattr(cloud_engine.httpx, "post", fake_post)
    engine = CloudRenderEngine(config={"CLOUDFLARE_API_TOKEN": "cf-token", "CLOUDFLARE_ACCOUNT_ID": "acct"})
    result = engine.render(box_sketch(), "a kraft carton", negative_prompt="blurry")

    assert result.provider == "cloudflare"
    assert result.engine_tag == "cloudflare-img2img"
    assert result.model == "@cf/black-forest-labs/flux-2-klein-9b"
    assert isinstance(result.image, Image.Image)
    assert result.image.size == (16, 16)
    assert "Neurons" in result.cost_note
    assert (
        "api.cloudflare.com/client/v4/accounts/acct/ai/run/"
        "@cf/black-forest-labs/flux-2-klein-9b" in captured["url"]
    )
    assert captured["headers"]["Authorization"] == "Bearer cf-token"
    assert captured["data"]["prompt"].startswith("a kraft carton")
    assert "Avoid: blurry" in captured["data"]["prompt"]  # negative folded (no dedicated field)
    assert captured["data"]["num_inference_steps"] == str(cloud_engine.CLOUDFLARE_NUM_STEPS)
    name, fileobj, ctype = captured["files"]["image"]
    assert name == "sketch.png" and ctype == "image/png"
    assert fileobj.startswith(b"\x89PNG")


def test_cloudflare_img2img_variant_json_and_strength(monkeypatch):
    """The classic SD1.5-img2img variant uses JSON + data-URI + strength mapping."""
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["payload"] = kwargs.get("json")
        return FakeResponse(200, {"success": True, "result": "data:image/png;base64," + png_b64()})

    monkeypatch.setattr(cloud_engine.httpx, "post", fake_post)
    engine = CloudRenderEngine(
        config={
            "CLOUDFLARE_API_TOKEN": "cf",
            "CLOUDFLARE_ACCOUNT_ID": "acct",
            "CLOUDFLARE_IMG2IMG_MODEL": "@cf/runwayml/stable-diffusion-v1-5-img2img",
            "CLOUDFLARE_STRENGTH": "0.4",
        }
    )
    result = engine.render(box_sketch(), "a kraft carton")  # no explicit conditioning_scale

    assert result.model == "@cf/runwayml/stable-diffusion-v1-5-img2img"
    payload = captured["payload"]
    assert payload["prompt"].startswith("a kraft carton")
    assert payload["image"].startswith("data:image/png;base64,")
    assert abs(payload["strength"] - 0.4) < 1e-9  # env value honored

    engine.render(box_sketch(), "a kraft carton", control_strength=0.8)
    assert abs(captured["payload"]["strength"] - 0.63) < 1e-9  # 0.35 + 0.8*0.35

    engine.render(box_sketch(), "a kraft carton", control_strength=1.5)
    assert captured["payload"]["strength"] == 0.8  # clamped high


def test_cloudflare_raw_binary_and_nested_b64_parse(monkeypatch):
    calls = {"n": 0}

    def fake_post(url, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            # raw binary body (no JSON envelope) - Workers AI default shape
            return FakeResponse(200, content=png_bytes(20, 20), headers={"content-type": "image/png"})
        return FakeResponse(200, {"success": True, "result": {"image": png_b64(18, 18)}})

    monkeypatch.setattr(cloud_engine.httpx, "post", fake_post)
    engine = CloudRenderEngine(
        config={
            "CLOUDFLARE_API_TOKEN": "cf",
            "CLOUDFLARE_ACCOUNT_ID": "acct",
            "CLOUDFLARE_IMG2IMG_MODEL": "@cf/runwayml/stable-diffusion-v1-5-img2img",
        }
    )

    raw = engine.render(box_sketch(), "a kraft carton")
    assert raw.provider == "cloudflare" and raw.image.size == (20, 20)
    nested = engine.render(box_sketch(), "a kraft carton")
    assert nested.provider == "cloudflare" and nested.image.size == (18, 18)


def test_cloudflare_429_marks_cooldown(monkeypatch):
    monkeypatch.setattr(
        cloud_engine.httpx,
        "post",
        lambda url, **kw: FakeResponse(429, {"errors": [{"message": "Rate limit exceeded"}]}),
    )
    engine = CloudRenderEngine(config={"CLOUDFLARE_API_TOKEN": "cf", "CLOUDFLARE_ACCOUNT_ID": "acct"})
    with pytest.raises(CloudRateLimitedError):
        engine.render(box_sketch(), "a kraft carton")
    assert engine.probe()["cloudflare"]["status"] == "cooldown"
    assert engine.available_provider() is None


def test_cloudflare_model_chain_skips_rejected_model(monkeypatch):
    """A 403-rejected primary model (new accounts / gated models) is skipped
    and the next model in the chain serves the render."""
    seen_urls = []

    def fake_post(url, **kwargs):
        seen_urls.append(url)
        if "flux-2-klein-9b" in url:
            return FakeResponse(403, {"errors": [{"message": "AiError: This account is not allowed to access this model"}]})
        return FakeResponse(200, {"success": True, "result": "data:image/png;base64," + png_b64(22, 22)})

    monkeypatch.setattr(cloud_engine.httpx, "post", fake_post)
    engine = CloudRenderEngine(config={"CLOUDFLARE_API_TOKEN": "cf", "CLOUDFLARE_ACCOUNT_ID": "acct"})
    result = engine.render(box_sketch(), "a kraft carton")
    assert result.provider == "cloudflare"
    assert result.model == "@cf/black-forest-labs/flux-2-klein-4b"  # next in chain
    assert result.image.size == (22, 22)
    assert len(seen_urls) == 2  # klein-9b rejected, klein-4b served


def test_cloudflare_failure_falls_through_to_stability(monkeypatch):
    urls = []

    def fake_post(url, **kwargs):
        urls.append(url)
        if "api.cloudflare.com" in url:
            return FakeResponse(402, {"errors": [{"message": "free daily Neurons exhausted"}]})
        return FakeResponse(200, content=png_bytes(), headers={"content-type": "image/png"})

    monkeypatch.setattr(cloud_engine.httpx, "post", fake_post)
    engine = CloudRenderEngine(
        config={"CLOUDFLARE_API_TOKEN": "cf", "CLOUDFLARE_ACCOUNT_ID": "acct", "STABILITY_API_KEY": "s"}
    )
    result = engine.render(box_sketch(), "a kraft carton")

    assert urls[0].startswith("https://api.cloudflare.com")  # free tier tried before paid
    assert result.provider == "stability"  # chain fell through after cooldown mark
    probe = engine.probe()
    assert probe["cloudflare"]["status"] == "cooldown"
    assert probe["stability"]["status"] == "ready"


# ---------------------------------------------------------------------------
# ModelScope provider (dormant Qwen-Image-Edit skeleton)
# ---------------------------------------------------------------------------


def test_modelscope_disabled_by_default_without_token():
    engine = CloudRenderEngine()
    assert engine._is_configured("modelscope") is False
    probe = engine.probe()
    assert probe["modelscope"]["configured"] is False
    assert probe["modelscope"]["status"] == "missing_key"
    assert "MODELSCOPE_ENABLED" in probe["modelscope"]["note"]


def test_modelscope_token_alone_stays_dormant():
    engine = CloudRenderEngine(config={"MODELSCOPE_API_TOKEN": "ms"})
    assert engine._is_configured("modelscope") is False  # MODELSCOPE_ENABLED defaults false
    assert engine.models["modelscope"] == "Qwen/Qwen-Image-Edit"


def test_modelscope_probe_ready_and_render_is_honest_skeleton(monkeypatch):
    def fail_post(*args, **kwargs):
        raise AssertionError("modelscope must not fire network requests while unverified")

    monkeypatch.setattr(cloud_engine.httpx, "post", fail_post)
    engine = CloudRenderEngine(config={"MODELSCOPE_API_TOKEN": "ms", "MODELSCOPE_ENABLED": "true"})
    assert engine._is_configured("modelscope") is True
    probe = engine.probe()
    assert probe["modelscope"]["status"] == "ready"
    assert "verification" in (probe["modelscope"]["note"] or "").lower()

    with pytest.raises(CloudUnavailableError) as excinfo:
        engine.render(box_sketch(), "a kraft carton")
    assert "hosted image URL" in str(excinfo.value)
    # the honest failure marks the provider cooldown like every other provider
    cooled = engine.probe()["modelscope"]
    assert cooled["status"] == "cooldown"
    assert "hosted image URL" in (cooled["last_error"] or "")


# ---------------------------------------------------------------------------
# vlm_client
# ---------------------------------------------------------------------------


def _force_gemini_env(monkeypatch):
    monkeypatch.setattr(vlm_client, "_ENV_LOADED", True)
    monkeypatch.setenv("VLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_vlm_provider_detection_order(monkeypatch):
    monkeypatch.setattr(vlm_client, "_ENV_LOADED", True)
    monkeypatch.delenv("VLM_PROVIDER", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("OPENROUTER_API_KEY", "o")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    chain = detect_vlm_providers()
    assert [p for p, _ in chain] == ["gemini", "openrouter", "openai"]
    assert chain[0][1] == "gemini-3.8-flash"
    assert chain[1][1] == "nvidia/nemotron-3.5-super-vl:free"

    monkeypatch.setenv("VLM_PROVIDER", "openai")
    assert [p for p, _ in detect_vlm_providers()] == ["openai"]

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("VLM_PROVIDER", raising=False)
    assert vlm_available() is False


def test_vlm_cache_single_network_call(monkeypatch):
    _force_gemini_env(monkeypatch)
    calls = {"n": 0}

    def fake_post(url, **kwargs):
        calls["n"] += 1
        return FakeResponse(200, gemini_text_payload("MOCK-VLM-REPLY"))

    monkeypatch.setattr(vlm_client.httpx, "post", fake_post)

    first = ask_vlm("What packaging material is shown?")
    second = ask_vlm("What packaging material is shown?")
    assert first == second == "MOCK-VLM-REPLY"
    assert calls["n"] == 1  # second identical call served from cache

    ask_vlm("A different question entirely")
    assert calls["n"] == 2  # different prompt -> new network call


def test_vlm_negative_cache_skips_retry(monkeypatch):
    _force_gemini_env(monkeypatch)
    calls = {"n": 0}

    def fake_post(url, **kwargs):
        calls["n"] += 1
        return FakeResponse(400, {"error": {"message": "bad request"}})

    monkeypatch.setattr(vlm_client.httpx, "post", fake_post)

    assert ask_vlm("hello there") == ""  # provider failed -> negative cached
    assert ask_vlm("hello there") == ""  # served from negative cache, no network
    assert calls["n"] == 1

    clear_cache()
    ask_vlm("hello there")
    assert calls["n"] == 2  # cache cleared -> provider retried


def test_vlm_unavailable_returns_empty_without_network(monkeypatch):
    monkeypatch.setattr(vlm_client, "_ENV_LOADED", True)
    monkeypatch.delenv("VLM_PROVIDER", raising=False)
    for key in ("GEMINI_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    def fail_post(*args, **kwargs):
        raise AssertionError("network call attempted with no providers configured")

    monkeypatch.setattr(vlm_client.httpx, "post", fail_post)
    assert ask_vlm("hi") == ""
    assert vlm_available() is False


def test_vlm_total_timeout_bounds_chain(monkeypatch):
    _force_gemini_env(monkeypatch)
    monkeypatch.setenv("VLM_TOTAL_TIMEOUT_S", "0.5")
    calls = {"n": 0}

    def fake_post(url, **kwargs):
        calls["n"] += 1
        return FakeResponse(200, gemini_text_payload("too late"))

    monkeypatch.setattr(vlm_client.httpx, "post", fake_post)
    assert ask_vlm("budget test") == ""
    assert calls["n"] == 0  # budget exhausted before any request


def test_vlm_gemini_thinking_budget_fallback(monkeypatch):
    _force_gemini_env(monkeypatch)
    payloads = []

    def fake_post(url, **kwargs):
        # deep-copy: _call_gemini mutates generationConfig (pops thinkingConfig)
        # on retry, which would otherwise retroactively change captured payloads.
        payloads.append(json.loads(json.dumps(kwargs.get("json"))))
        if len(payloads) == 1:
            return FakeResponse(400, {"error": {"message": "Unknown name thinkingConfig"}})
        return FakeResponse(200, gemini_text_payload("OK"))

    monkeypatch.setattr(vlm_client.httpx, "post", fake_post)
    assert ask_vlm("thinking test") == "OK"
    assert len(payloads) == 2
    assert "thinkingConfig" in payloads[0]["generationConfig"]
    assert payloads[0]["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}
    assert "thinkingConfig" not in payloads[1]["generationConfig"]  # graceful fallback


# ---------------------------------------------------------------------------
# Pipeline dispatch integration (local engines - no network)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def renderer():
    from src.pipeline import PackVisionRenderer

    return PackVisionRenderer(use_mock_inference=True, device="cpu")


def test_pipeline_procedural_dispatch_sets_engine_used(renderer):
    renderer.render_engine_pref = "local"
    result = renderer.render(sketch=box_sketch(), engine_mode="procedural", seed=7)

    assert renderer.last_engine_used == "procedural-pbr"
    assert result.metadata["engine_used"] == "procedural-pbr"
    assert result.metadata["engine_latency_s"] >= 0
    assert "engine_note" in result.metadata
    assert result.is_mock is True
    assert result.negative_prompt  # structured negative prompt present


def test_pipeline_negative_prompt_folds_into_local_negative(renderer):
    renderer.render_engine_pref = "local"
    result = renderer.render(
        sketch=box_sketch(), engine_mode="procedural", negative_prompt="neon pink plastic", seed=3
    )
    assert "neon pink plastic" in result.negative_prompt.lower()


def test_pipeline_cloud_priority_dispatch(renderer):
    class FakeCloud:
        def available_provider(self):
            return "gemini"

        def render(self, sketch, prompt, negative_prompt="", size=1024, control_strength=None):
            assert negative_prompt == "no glare"
            assert 0.0 <= (control_strength or 0.0) <= 1.5  # conditioning_scale mapped through
            return CloudRenderResult(
                image=Image.new("RGB", (64, 64), (10, 120, 200)),
                provider="gemini",
                model="gemini-2.5-flash-image",
                latency_s=0.42,
                cost_note="free tier",
            )

    renderer.cloud = FakeCloud()
    renderer.render_engine_pref = "auto"
    result = renderer.render(sketch=box_sketch(), negative_prompt="no glare", seed=5)

    assert renderer.last_engine_used == "cloud-gemini"
    assert result.metadata["engine_used"] == "cloud-gemini"
    assert result.metadata["cloud_latency_s"] == 0.42
    assert result.metadata["engine_note"] == "free tier"
    assert result.is_mock is False


def test_pipeline_cloud_failure_falls_back_to_local(renderer):
    from src.cloud_engine import CloudUnavailableError

    class DeadCloud:
        def available_provider(self):
            return "openai"

        def render(self, sketch, prompt, negative_prompt="", size=1024, control_strength=None):
            raise CloudUnavailableError("openai quota exhausted")

    renderer.cloud = DeadCloud()
    renderer.render_engine_pref = "cloud"
    # keep the custom checkpoint out of the dispatch so the local fallback is procedural
    renderer.custom_model_path = Path("models/checkpoints/__missing_for_test__.pt")
    result = renderer.render(sketch=box_sketch(), seed=11)

    assert renderer.last_engine_used == "procedural-pbr"
    assert result.metadata["engine_used"] == "procedural-pbr"
    assert "cloud" in result.metadata["engine_note"].lower()
    assert "quota" in result.metadata["cloud_fallback_reason"]


def test_pipeline_hf_engine_used_tag(renderer):
    """The free HuggingFace path reports the dedicated engine tag."""

    class FakeHfCloud:
        def available_provider(self):
            return "huggingface"

        def render(self, sketch, prompt, negative_prompt="", size=1024, control_strength=None):
            return CloudRenderResult(
                image=Image.new("RGB", (64, 64), (200, 30, 90)),
                provider="huggingface",
                model="black-forest-labs/FLUX.1-Kontext-dev",
                latency_s=45.1,
                cost_note="HuggingFace ZeroGPU FLUX.1-Kontext (free, anonymous quota limited)",
                engine_tag="hf-flux-kontext",
            )

    renderer.cloud = FakeHfCloud()
    renderer.render_engine_pref = "auto"
    result = renderer.render(sketch=box_sketch(), seed=13)

    assert renderer.last_engine_used == "cloud-hf-flux-kontext"
    assert result.metadata["engine_used"] == "cloud-hf-flux-kontext"
    assert result.is_mock is False


def test_session_leaves_providers_cooldown_for_backend_suite(monkeypatch):
    """Final test in this file: leave all providers marked unavailable.

    Mirrors real-world behavior after quota discovery (Gemini free quota = 0,
    OpenRouter 402 balance, OpenAI 429 insufficient_quota, ZeroGPU busy): once
    learned, the module-level cooldown registry makes subsequent renders in
    this process - including backend/tests/test_api.py's generation lifecycle
    job - skip cloud network attempts entirely and use local engines instantly.
    """
    monkeypatch.setenv("HF_ENABLED", "true")  # HF is key-less: enabled by default in production
    cloud_engine.mark_provider_unavailable(
        "huggingface", "huggingface ZeroGPU busy/quota: GPU quota exceeded", cooldown_s=cloud_engine.HF_COOLDOWN_S
    )
    cloud_engine.mark_provider_unavailable("stability", "stability unavailable (HTTP 402): insufficient credits")
    cloud_engine.mark_provider_unavailable("gemini", "gemini quota exhausted (429).")
    cloud_engine.mark_provider_unavailable(
        "openrouter", "openrouter unavailable (HTTP 402): requires at least $1.00 in balance"
    )
    cloud_engine.mark_provider_unavailable("openai", "openai quota exhausted (429).")

    engine = CloudRenderEngine(
        config={
            "STABILITY_API_KEY": "s",
            "GEMINI_API_KEY": "g",
            "OPENROUTER_API_KEY": "o",
            "OPENAI_API_KEY": "k",
        }
    )
    probe = engine.probe()
    assert all(
        probe[name]["status"] == "cooldown"
        for name in ("huggingface", "stability", "gemini", "openrouter", "openai")
    )
    # key-ready providers stay dormant (no keys configured in this test)
    assert probe["cloudflare"]["status"] == "missing_key"
    assert probe["modelscope"]["status"] == "missing_key"
    assert engine.available_provider() is None
    with pytest.raises(CloudUnavailableError):
        engine.render(box_sketch(), "should skip network instantly")


def test_hf_space_chain_falls_through_to_next_space(fake_gradio, monkeypatch):
    """A space on per-space cooldown is skipped; the next space renders."""
    state, out_path = fake_gradio
    monkeypatch.setenv("HF_ENABLED", "true")
    monkeypatch.setenv(
        "HF_SPACES", "failing-space/one,black-forest-labs/FLUX.1-Kontext-dev"
    )
    cloud_engine.reset_provider_state()  # hermetic: clear provider+space cooldowns from earlier tests
    monkeypatch.setenv("RENDER_ENGINE", "auto")  # earlier conftests may force "local"
    engine = CloudRenderEngine()
    assert engine.hf_spaces == ["failing-space/one", "black-forest-labs/FLUX.1-Kontext-dev"]

    # Simulate the first space having just failed (quota): per-space cooldown.
    cloud_engine._HF_SPACE_STATE["failing-space/one"] = cloud_engine.time.time() + 60

    result = engine.render(box_sketch(), "chain carton")
    assert result.provider == "huggingface"
    assert isinstance(result.image, Image.Image)
    assert result.model == "black-forest-labs/FLUX.1-Kontext-dev"
    assert state["space"] == "black-forest-labs/FLUX.1-Kontext-dev"  # second space served
    assert state["instances"] == 1  # failing space never connected
    assert state["predicts"] == 1


def test_hf_space_chain_marks_failed_space_and_uses_next(fake_gradio, monkeypatch):
    """First space raises a quota error at render time -> it gets a cooldown
    and the second space in the chain serves the render in the same call."""
    state, out_path = fake_gradio
    monkeypatch.setenv("HF_ENABLED", "true")
    monkeypatch.setenv(
        "HF_SPACES", "failing-space/one,black-forest-labs/FLUX.1-Kontext-dev"
    )
    cloud_engine.reset_provider_state()  # hermetic: clear provider+space cooldowns from earlier tests
    monkeypatch.setenv("RENDER_ENGINE", "auto")  # earlier conftests may force "local"
    engine = CloudRenderEngine()

    seen_spaces = []

    class SpaceAwareClient(state_client_cls := object):
        pass

    # wrap FakeClient via state["error"] being space-aware is not possible;
    # instead patch predict at the client instance level after construction.
    def space_aware_predict(self, **kwargs):  # bound to the fake via closure
        seen_spaces.append(self.space)
        if self.space == "failing-space/one":
            raise RuntimeError("You have exceeded your GPU quota (ZeroGPU).")
        out_path.write_bytes(png_bytes(24, 24, (30, 144, 255)))
        return (str(out_path), 9)

    # The fixture's FakeClient.predict checks state["error"]; we emulate
    # per-space failure by wrapping the cached client per space.
    from src import cloud_engine as ce

    real_get_client = ce._get_hf_client

    class Wrapper:
        def __init__(self, inner, space):
            self._inner = inner
            self.space = space

        def predict(self, **kwargs):
            seen_spaces.append(self.space)
            if self.space == "failing-space/one":
                raise RuntimeError("You have exceeded your GPU quota (ZeroGPU).")
            return self._inner.predict(**kwargs)

    def fake_get_client(space, token=None, client_factory=None):
        inner = real_get_client(space, token, client_factory=client_factory)
        return Wrapper(inner, space)

    monkeypatch.setattr(ce, "_get_hf_client", fake_get_client)

    result = engine.render(box_sketch(), "chain carton two")
    assert result.provider == "huggingface"
    assert isinstance(result.image, Image.Image)
    assert seen_spaces == ["failing-space/one", "black-forest-labs/FLUX.1-Kontext-dev"]
    # the failed space got a per-space cooldown
    assert ce._HF_SPACE_STATE["failing-space/one"] > ce.time.time()
    assert result.model == "black-forest-labs/FLUX.1-Kontext-dev"
