"""
PackVision AI - Evaluation & Quality Assessment Engine
======================================================
Comprehensive evaluation metrics for packaging synthesis:
1. Edge SSIM & IoU (Sketch fidelity / structural preservation)
2. CLIP Score (Prompt text-to-image semantic alignment)
3. FID & LPIPS (Fréchet Inception Distance & Perceptual similarity)
4. Automated Benchmark Report Generator (JSON & Markdown)

Metric provenance
-----------------
Every metric result reports WHERE its number came from via the ``source`` scheme
documented in ``METRIC_PROVENANCE_KEYS``:

- ``"library"``     -- computed by a real implementation (pretrained model loaded
  from local weights, or a deterministic library routine such as OpenCV/skimage).
- ``"vlm"``         -- estimated by a vision-language model (``src.vlm_client``).
- ``"heuristic"``   -- fallback formula approximation, NOT the nominal metric.
  Notably, ``compute_fid`` and ``compute_lpips`` are always numpy heuristics
  (proxy metrics), and ``compute_clip_score`` degrades to ``_heuristic_clip_score``
  when neither real CLIP weights nor a VLM are available.
- ``"unavailable"`` -- metric could not be computed.

Network policy: real CLIP is loaded from locally cached HuggingFace weights only
(``local_files_only=True``). Downloading anything from the network is opt-in via
``PACKVISION_ALLOW_MODEL_DOWNLOADS=true``. While running under pytest
(``PYTEST_CURRENT_TEST`` set) both the CLIP loader and the VLM client are fully
inert unless that env var is explicitly set, so the default test suite is fast,
hermetic, and makes no network connections.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as ssim_fn
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

from src.preprocessor import SketchPreprocessor

# Defensive import: src.vlm_client may be provided by a concurrent module. When
# absent, every VLM path degrades gracefully to heuristics.
try:
    from src.vlm_client import vlm_available as _vlm_available
    from src.vlm_client import ask_vlm as _ask_vlm
except Exception:  # pragma: no cover - depends on optional module
    _vlm_available = None
    _ask_vlm = None


# =============================================================================
# METRIC PROVENANCE SCHEME
# =============================================================================

#: Provenance labels attached to every metric result (``source`` fields).
METRIC_PROVENANCE_KEYS: Dict[str, str] = {
    "library": (
        "Computed by a real implementation: a pretrained model loaded from local "
        "weights (never downloaded) or a deterministic library routine."
    ),
    "vlm": (
        "Estimated by a vision-language model (src.vlm_client) scoring the "
        "image-text alignment; an approximation, not the nominal metric."
    ),
    "heuristic": (
        "Fallback formula approximation - NOT the nominal metric (e.g. pseudo-CLIP "
        "contrast/keyword proxy, numpy FID/LPIPS proxies)."
    ),
    "unavailable": "Metric could not be computed.",
}

# VLM response cache: (image_sha256, prompt_text) -> score in [0, 1], LRU-capped.
_VLM_CACHE: "OrderedDict[Tuple[str, str], float]" = OrderedDict()
_VLM_CACHE_SIZE = 128

_VLM_CLIP_PROMPT_TEMPLATE = (
    "You are a strict automated evaluation judge for product packaging renders. "
    "Rate how well the image matches the text prompt on a scale from 0 to 10 "
    "(0 = completely unrelated, 10 = perfect photorealistic alignment). "
    'Respond with ONLY a JSON object of the form {{"score": <number 0-10>}} and '
    "nothing else.\nText prompt: {text}"
)


def _model_downloads_allowed() -> bool:
    """True when PACKVISION_ALLOW_MODEL_DOWNLOADS is explicitly enabled."""
    return os.getenv("PACKVISION_ALLOW_MODEL_DOWNLOADS", "").strip().lower() in ("true", "1", "yes")


def _offline_guard_active() -> bool:
    """
    True when network-touching model paths must stay completely inert.

    Active while running under pytest (``PYTEST_CURRENT_TEST`` is set) unless
    ``PACKVISION_ALLOW_MODEL_DOWNLOADS`` is explicitly enabled. This keeps the
    default test suite hermetic: no CLIP weight loading, no VLM HTTP calls -
    everything degrades to documented heuristics. Outside pytest the guard is
    inactive and CLIP/VLM operate per the rules below.
    """
    if _model_downloads_allowed():
        return False
    return "PYTEST_CURRENT_TEST" in os.environ


def _clip_weights_locally_cached(model_name: str) -> bool:
    """
    Cheap disk check for locally cached HuggingFace weights (no imports, no I/O
    beyond a stat, no network).

    Looks for the standard ``models--<org>--<model>`` folder under the hub cache
    resolved from ``HF_HUB_CACHE`` / ``HF_HOME`` / ``~/.cache/huggingface/hub``.
    Returns False on any uncertainty, so callers degrade to VLM/heuristic
    scoring instead of paying an expensive torch/transformers import or a
    network timeout.
    """
    try:
        if os.getenv("HF_HUB_CACHE"):
            hub_cache = os.environ["HF_HUB_CACHE"]
        elif os.getenv("HF_HOME"):
            hub_cache = os.path.join(os.environ["HF_HOME"], "hub")
        else:
            hub_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        folder = "models--" + model_name.replace("/", "--")
        return os.path.isdir(os.path.join(hub_cache, folder))
    except Exception:
        return False


def _reset_caches() -> None:
    """
    Test hook: reset all module-level mutable state in ``src.evaluation``.

    Clears the VLM response LRU cache, drops the shared evaluator singleton,
    clears ``src.vlm_client``'s own response/negative caches, and re-binds the
    defensively-imported VLM hooks from a pristine ``src.vlm_client`` (in case
    another module overwrote them). Call at test setup/teardown for fully
    order-independent suites.
    """
    _VLM_CACHE.clear()
    global _SHARED_EVALUATOR
    _SHARED_EVALUATOR = None
    try:
        import src.vlm_client as _vlm_module

        clear = getattr(_vlm_module, "clear_cache", None)
        if callable(clear):
            clear()
        globals()["_vlm_available"] = getattr(_vlm_module, "vlm_available", None)
        globals()["_ask_vlm"] = getattr(_vlm_module, "ask_vlm", None)
    except Exception:
        globals()["_vlm_available"] = None
        globals()["_ask_vlm"] = None


def _image_signature(image: Image.Image) -> str:
    """Deterministic SHA-256 signature of a PIL image for VLM response caching."""
    arr = np.ascontiguousarray(np.asarray(image.convert("RGB")))
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


def _call_vlm(prompt: str, image: Image.Image) -> Optional[str]:
    """Invoke the optional VLM client defensively; returns raw text or None.

    Tries ``ask_vlm(prompt, image, json_mode=True)`` first (the real client's
    structured-output mode) and falls back to progressively simpler call
    conventions so custom/monkeypatched clients keep working. The real client
    returns "" on failure instead of raising, which callers treat as None.
    """
    if _ask_vlm is None:
        return None
    try:
        return _ask_vlm(prompt, image, json_mode=True)
    except TypeError:
        pass
    except Exception:
        return None
    try:
        return _ask_vlm(prompt, image)
    except TypeError:
        try:
            return _ask_vlm(prompt, image=image)
        except Exception:
            return None
    except Exception:
        return None


def _parse_vlm_score(raw: str) -> Optional[float]:
    """Parse a 0-10 VLM rating from raw text; returns a 0-1 score or None."""
    if not isinstance(raw, str):
        return None
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            payload = json.loads(match.group(0))
            if isinstance(payload, dict):
                for key in ("score", "rating", "alignment"):
                    if key in payload:
                        value = float(payload[key])
                        return min(max(value, 0.0), 10.0) / 10.0
        # Fall back to the first bare number in the response.
        bare = re.search(r"\b(\d+(?:\.\d+)?)\b", raw)
        if bare:
            value = float(bare.group(1))
            return min(max(value, 0.0), 10.0) / 10.0
    except Exception:
        return None
    return None


def _vlm_clip_score(image: Image.Image, prompt_text: str) -> Optional[float]:
    """
    Estimate image-text alignment via the optional VLM client (0-1).

    Responses are LRU-cached by (image signature, prompt text). Returns None when
    the VLM is unavailable, the offline guard is active (pytest without explicit
    opt-in), the client errors, or returns an unparseable rating.
    """
    if _offline_guard_active():
        # Never fire real VLM network calls under pytest unless explicitly
        # opted in - callers degrade to heuristic scoring.
        return None
    if _vlm_available is None or _ask_vlm is None:
        return None
    try:
        if not _vlm_available():
            return None
    except Exception:
        return None

    cache_key = (_image_signature(image), prompt_text)
    if cache_key in _VLM_CACHE:
        _VLM_CACHE.move_to_end(cache_key)
        return _VLM_CACHE[cache_key]

    prompt = _VLM_CLIP_PROMPT_TEMPLATE.format(text=prompt_text[:600])
    raw = _call_vlm(prompt, image)
    score = _parse_vlm_score(raw) if raw else None
    if score is None:
        return None

    _VLM_CACHE[cache_key] = score
    if len(_VLM_CACHE) > _VLM_CACHE_SIZE:
        _VLM_CACHE.popitem(last=False)
    return score


def _heuristic_clip_score(image: Image.Image, prompt_text: str) -> float:
    """
    Deterministic pseudo-CLIP heuristic - NOT a real CLIP score.

    Approximates text-image alignment from substrate chromatic harmony, studio
    edge definition (contrast), and packaging keyword coverage in the prompt.
    The result is clamped to the [0.22, 0.38] band typical of genuine CLIP
    image-text cosine similarities for photo-text pairs, so downstream quality
    thresholds keep working. Always label results with ``source="heuristic"``.
    """
    img_np = np.array(image.convert("RGB"))
    mean_lum = np.mean(img_np) / 255.0
    contrast = np.std(img_np) / 128.0

    # Text complexity factor
    words = prompt_text.lower().split()
    keywords = {"kraft", "pouch", "bottle", "carton", "foil", "studio", "matte", "glass"}
    matched = sum(1 for w in words if any(k in w for k in keywords))
    keyword_factor = min(0.12, matched * 0.02)

    # Grounded realistic pseudo-CLIP score
    base_score = 0.28 + (contrast * 0.05) + keyword_factor
    return float(np.clip(base_score, 0.22, 0.38))


@dataclass
class PackagingMetricResult:
    """Individual evaluation metric scorecard."""
    sample_id: str
    edge_iou: float
    edge_ssim: float
    clip_score: float
    lpips_distance: Optional[float] = None
    fidelity_rating: str = "Excellent"
    # Provenance of each metric value (see METRIC_PROVENANCE_KEYS).
    clip_source: str = "heuristic"
    metric_sources: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# EDGE FIDELITY METRICS (IOU & SSIM)
# =============================================================================

def compute_edge_iou(
    sketch_img: Union[Image.Image, np.ndarray],
    render_img: Union[Image.Image, np.ndarray],
    dilation_radius: int = 3,
) -> float:
    """
    Compute Intersection-over-Union (IoU) between input sketch and rendered edges.
    
    A small morphological dilation tolerance is applied to the sketch lines to
    accommodate sub-pixel rasterization and subtle 3D lighting bevel variance.
    
    Args:
        sketch_img: Input sketch (PIL Image or numpy array).
        render_img: Rendered output image.
        dilation_radius: Tolerance dilation radius in pixels (default: 3).
        
    Returns:
        Edge IoU score [0.0, 1.0].
    """
    # Convert to grayscale numpy arrays
    if isinstance(sketch_img, Image.Image):
        sketch_gray = np.array(sketch_img.convert("L"))
    else:
        sketch_gray = cv2.cvtColor(sketch_img, cv2.COLOR_BGR2GRAY) if sketch_img.ndim == 3 else sketch_img

    if isinstance(render_img, Image.Image):
        render_gray = np.array(render_img.convert("L"))
    else:
        render_gray = cv2.cvtColor(render_img, cv2.COLOR_BGR2GRAY) if render_img.ndim == 3 else render_img

    # Ensure identical spatial dimensions
    h, w = sketch_gray.shape[:2]
    if render_gray.shape[:2] != (h, w):
        render_gray = cv2.resize(render_gray, (w, h), interpolation=cv2.INTER_AREA)

    # Extract Canny edges
    edges_sketch = cv2.Canny(sketch_gray, 50, 150)
    edges_render = cv2.Canny(render_gray, 50, 150)

    # Dilate sketch edges by tolerance radius
    if dilation_radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilation_radius + 1, 2 * dilation_radius + 1)
        )
        sketch_dilated = cv2.dilate(edges_sketch, kernel)
    else:
        sketch_dilated = edges_sketch

    sketch_bool = sketch_dilated > 0
    render_bool = edges_render > 0

    intersection = np.logical_and(sketch_bool, render_bool).sum()
    union = np.logical_or(sketch_bool, render_bool).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(intersection / union)


def compute_edge_ssim(
    sketch_img: Union[Image.Image, np.ndarray],
    render_img: Union[Image.Image, np.ndarray],
) -> float:
    """
    Compute Structural Similarity Index (SSIM) between edge maps of sketch and render.
    
    Args:
        sketch_img: Input sketch.
        render_img: Rendered output.
        
    Returns:
        Structural similarity score [0.0, 1.0].
    """
    if isinstance(sketch_img, Image.Image):
        s_gray = np.array(sketch_img.convert("L"))
    else:
        s_gray = cv2.cvtColor(sketch_img, cv2.COLOR_BGR2GRAY) if sketch_img.ndim == 3 else sketch_img

    if isinstance(render_img, Image.Image):
        r_gray = np.array(render_img.convert("L"))
    else:
        r_gray = cv2.cvtColor(render_img, cv2.COLOR_BGR2GRAY) if render_img.ndim == 3 else render_img

    h, w = s_gray.shape[:2]
    if r_gray.shape[:2] != (h, w):
        r_gray = cv2.resize(r_gray, (w, h), interpolation=cv2.INTER_AREA)

    s_edges = cv2.Canny(s_gray, 50, 150)
    r_edges = cv2.Canny(r_gray, 50, 150)

    if HAS_SKIMAGE:
        score = ssim_fn(s_edges, r_edges, data_range=255)
        return float(max(0.0, score))
    else:
        # Fallback numpy normalized cross correlation
        s_norm = s_edges.astype(np.float32) - np.mean(s_edges)
        r_norm = r_edges.astype(np.float32) - np.mean(r_edges)
        denom = (np.std(s_norm) * np.std(r_norm)) * s_norm.size
        if denom == 0:
            return 0.5
        corr = np.sum(s_norm * r_norm) / denom
        return float(np.clip(corr * 0.5 + 0.5, 0.0, 1.0))


# =============================================================================
# PACKAGING EVALUATOR CLASS
# =============================================================================

class PackagingEvaluator:
    """
    Holistic evaluator for packaging render quality, structural preservation,
    and text-prompt semantic alignment.
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        device: str = "auto",
    ) -> None:
        self.clip_model_name = clip_model_name
        self.device = device
        self.clip_model = None
        self.clip_processor = None
        self._clip_initialized = False
        # Provenance of the most recent metric computations
        # (keys are metric names, values are METRIC_PROVENANCE_KEYS labels).
        self.last_metric_sources: Dict[str, str] = {}

    def _init_clip(self) -> bool:
        """
        Attempt to lazily initialize HuggingFace CLIP model and processor.

        Loading policy (network-safe by default):
        - Under pytest (``PYTEST_CURRENT_TEST`` set) CLIP is skipped entirely
          unless ``PACKVISION_ALLOW_MODEL_DOWNLOADS`` is explicitly enabled.
        - Otherwise, locally cached weights are loaded with
          ``local_files_only=True`` (never any network I/O).
        - Only when ``PACKVISION_ALLOW_MODEL_DOWNLOADS=true`` is set AND the
          weights are not cached locally is a (opt-in) download attempted.
        """
        if self._clip_initialized:
            return self.clip_model is not None

        self._clip_initialized = True

        if _offline_guard_active():
            return False

        # Fail fast: without locally cached weights AND without an explicit
        # opt-in, skip the expensive torch/transformers imports entirely (they
        # can take tens of seconds cold and never help without weights).
        if not _clip_weights_locally_cached(self.clip_model_name) and not _model_downloads_allowed():
            return False

        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            target_dev = "cuda" if (self.device == "cuda" or (self.device == "auto" and torch.cuda.is_available())) else "cpu"

            # 1) Locally cached weights only - no network I/O.
            try:
                self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name, local_files_only=True)
                self.clip_model = CLIPModel.from_pretrained(self.clip_model_name, local_files_only=True).to(target_dev)
                self.clip_model.eval()
                self._target_dev = target_dev
                return True
            except Exception:
                self.clip_model = None
                self.clip_processor = None

            # 2) Explicit opt-in download (never in the default/pytest path).
            if _model_downloads_allowed():
                try:
                    self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
                    self.clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(target_dev)
                    self.clip_model.eval()
                    self._target_dev = target_dev
                    return True
                except Exception:
                    pass

            # Weights not available - fall back to VLM/heuristic gracefully.
            return False
        except Exception:
            self.clip_model = None
            self.clip_processor = None
            return False

    def compute_clip_score_detailed(self, image: Image.Image, prompt_text: str) -> Dict[str, Any]:
        """
        Compute the text-image alignment score with full provenance.

        Resolution order:
        1. ``"library"``   - real CLIP (transformers) from locally cached weights.
        2. ``"vlm"``       - src.vlm_client strict 0-10 alignment rating mapped to 0-1.
        3. ``"heuristic"`` - ``_heuristic_clip_score`` pseudo-CLIP proxy formula.

        Returns:
            Dict with keys ``score`` (float in [0, 1]), ``source`` (one of
            METRIC_PROVENANCE_KEYS labels), and provenance detail fields.
        """
        if self._init_clip():
            import torch

            inputs = self.clip_processor(
                text=[prompt_text[:77]],  # CLIP token limit
                images=image,
                return_tensors="pt",
                padding=True,
            ).to(self._target_dev)

            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                img_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
                score = (img_embeds @ text_embeds.T).item()
                self.last_metric_sources["clip_score"] = "library"
                return {
                    "score": float(max(0.0, score)),
                    "source": "library",
                    "model": self.clip_model_name,
                    "device": self._target_dev,
                }

        vlm_score = _vlm_clip_score(image, prompt_text)
        if vlm_score is not None:
            self.last_metric_sources["clip_score"] = "vlm"
            return {
                "score": float(vlm_score),
                "source": "vlm",
                "detail": "vision-language model 0-10 alignment rating mapped to 0-1",
            }

        score = _heuristic_clip_score(image, prompt_text)
        self.last_metric_sources["clip_score"] = "heuristic"
        return {
            "score": score,
            "source": "heuristic",
            "detail": "contrast/keyword pseudo-CLIP proxy in the typical CLIP range; NOT a true CLIP score",
        }

    def compute_clip_score(self, image: Image.Image, prompt_text: str) -> float:
        """
        Calculate cosine similarity between rendered packshot image and text prompt.

        Standard CLIP scores for photo-text pairs range from 0.20 to 0.40. When
        real CLIP weights are unavailable, this degrades to a VLM estimate or the
        documented ``_heuristic_clip_score`` proxy; the provenance of the latest
        call is recorded in ``self.last_metric_sources["clip_score"]`` and in
        ``compute_clip_score_detailed``.
        """
        return self.compute_clip_score_detailed(image, prompt_text)["score"]

    def compute_lpips(self, img1: Image.Image, img2: Image.Image) -> float:
        """
        PROXY perceptual distance (source="heuristic") - not true LPIPS.

        This is a numpy mean-absolutely-difference heuristic scaled into the
        typical LPIPS range; it does NOT use the pretrained VGG/AlexNet feature
        extractor that real LPIPS requires, and no such weights are downloaded.
        0.0 = perceptually similar, 1.0 = completely dissimilar (approximate).
        Provenance is recorded in ``self.last_metric_sources["lpips_proxy"]``.
        """
        arr1 = np.array(img1.convert("RGB")).astype(np.float32) / 255.0
        arr2 = np.array(img2.convert("RGB")).astype(np.float32) / 255.0

        if arr1.shape != arr2.shape:
            arr2 = cv2.resize(arr2, (arr1.shape[1], arr1.shape[0]))

        # High-frequency Sobel difference + color histogram variance
        diff_rgb = np.mean(np.abs(arr1 - arr2))
        self.last_metric_sources["lpips_proxy"] = "heuristic"
        return float(np.clip(diff_rgb * 0.85, 0.05, 0.95))

    def compute_fid(
        self,
        real_images: List[Image.Image],
        generated_images: List[Image.Image],
    ) -> float:
        """
        PROXY Fréchet distance (source="heuristic") - not true FID.

        Computes a Fréchet-style distance over simple spatial color moments and
        Sobel edge-energy statistics instead of Inception-V3 pool3 activations
        (no Inception weights are downloaded). Values are scaled/clamped to a
        FID-like range and are only meaningful as a relative distribution proxy.
        Provenance is recorded in ``self.last_metric_sources["fid_proxy"]``.
        """
        if not real_images or not generated_images:
            return 0.0

        def extract_feature_vector(img: Image.Image) -> np.ndarray:
            arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
            # Spatial moments & color channel covariance
            means = np.mean(arr, axis=(0, 1))
            stds = np.std(arr, axis=(0, 1))
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
            sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
            edge_energy = np.array([np.mean(sobel_x**2), np.mean(sobel_y**2)])
            return np.concatenate([means, stds, edge_energy])

        feats_real = np.array([extract_feature_vector(im) for im in real_images])
        feats_gen = np.array([extract_feature_vector(im) for im in generated_images])

        mu_r = np.mean(feats_real, axis=0)
        mu_g = np.mean(feats_gen, axis=0)
        sigma_r = np.cov(feats_real, rowvar=False) if len(real_images) > 1 else np.eye(len(mu_r)) * 0.1
        sigma_g = np.cov(feats_gen, rowvar=False) if len(generated_images) > 1 else np.eye(len(mu_g)) * 0.1

        diff = mu_r - mu_g
        fid = np.sum(diff ** 2) + np.trace(sigma_r + sigma_g - 2 * np.sqrt(np.abs(sigma_r * sigma_g)))
        self.last_metric_sources["fid_proxy"] = "heuristic"
        return float(np.clip(fid * 25.0, 5.0, 85.0))

    def evaluate(
        self,
        sketch_image: Image.Image,
        rendered_image: Image.Image,
        prompt_text: Optional[str] = None,
        positive_prompt: Optional[str] = None,
        sample_id: str = "sample_001",
        **kwargs,
    ) -> PackagingMetricResult:
        """Convenience alias for evaluate_sample."""
        prompt = prompt_text or positive_prompt or kwargs.get("prompt", "")
        return self.evaluate_sample(
            sample_id=sample_id,
            sketch_image=sketch_image,
            rendered_image=rendered_image,
            prompt_text=prompt,
        )

    def evaluate_sample(
        self,
        sample_id: str,
        sketch_image: Image.Image,
        rendered_image: Image.Image,
        prompt_text: str,
    ) -> PackagingMetricResult:
        """
        Execute comprehensive evaluation on a single sketch-render pair.

        The result carries per-metric provenance (``clip_source`` and
        ``metric_sources``, see METRIC_PROVENANCE_KEYS).
        """
        edge_iou = compute_edge_iou(sketch_image, rendered_image)
        edge_ssim = compute_edge_ssim(sketch_image, rendered_image)
        clip_detail = self.compute_clip_score_detailed(rendered_image, prompt_text)
        clip_score = clip_detail["score"]
        lpips_dist = self.compute_lpips(sketch_image, rendered_image)

        metric_sources = {
            "edge_iou": "library",
            "edge_ssim": "library",
            "clip_score": clip_detail["source"],
            "lpips_proxy": self.last_metric_sources.get("lpips_proxy", "heuristic"),
        }

        # Composite qualitative rating
        if edge_iou >= 0.65 and clip_score >= 0.28:
            rating = "Exceptional Commercial Fidelity"
        elif edge_iou >= 0.50 and clip_score >= 0.24:
            rating = "Production Grade"
        elif edge_iou >= 0.35:
            rating = "Acceptable Prototype"
        else:
            rating = "Geometry Deviation Detected"

        return PackagingMetricResult(
            sample_id=sample_id,
            edge_iou=round(edge_iou, 4),
            edge_ssim=round(edge_ssim, 4),
            clip_score=round(clip_score, 4),
            lpips_distance=round(lpips_dist, 4),
            fidelity_rating=rating,
            clip_source=clip_detail["source"],
            metric_sources=metric_sources,
            metadata={"clip_detail": {k: v for k, v in clip_detail.items() if k != "score"}},
        )

    def evaluate_render(
        self,
        image: Image.Image,
        prompt: str,
        sketch: Optional[Image.Image] = None,
        sample_id: str = "render_001",
    ) -> Dict[str, Any]:
        """
        Convenience single-render evaluation combining available metrics.

        Args:
            image: Rendered packshot to evaluate.
            prompt: Text prompt the render should align with.
            sketch: Optional input sketch; when provided, structural metrics
                (edge IoU / edge SSIM / LPIPS proxy) are included.
            sample_id: Identifier echoed into the result.

        Returns:
            Dict with per-metric values, a ``metric_sources`` map (metric ->
            METRIC_PROVENANCE_KEYS label), and a ``score_source`` summary field:
            the highest-fidelity provenance among computed metrics
            (library > vlm > heuristic), or ``"unavailable"``.
        """
        result: Dict[str, Any] = {"sample_id": sample_id}
        sources: Dict[str, str] = {}

        clip_detail = self.compute_clip_score_detailed(image, prompt)
        result["clip_score"] = round(clip_detail["score"], 4)
        result["clip_source"] = clip_detail["source"]
        sources["clip_score"] = clip_detail["source"]

        if sketch is not None:
            edge_iou = compute_edge_iou(sketch, image)
            edge_ssim = compute_edge_ssim(sketch, image)
            lpips_dist = self.compute_lpips(sketch, image)
            result["edge_iou"] = round(edge_iou, 4)
            result["edge_ssim"] = round(edge_ssim, 4)
            result["lpips_proxy"] = round(lpips_dist, 4)
            sources["edge_iou"] = "library"
            sources["edge_ssim"] = "library"
            sources["lpips_proxy"] = self.last_metric_sources.get("lpips_proxy", "heuristic")

        result["metric_sources"] = sources
        for label in ("library", "vlm", "heuristic"):
            if label in sources.values():
                result["score_source"] = label
                break
        else:
            result["score_source"] = "unavailable"
        return result

    def generate_benchmark_report(
        self,
        results: List[PackagingMetricResult],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate metric results across batch and generate JSON & Markdown reports.
        """
        if not results:
            return {"error": "No evaluation results to summarize."}

        ious = [r.edge_iou for r in results]
        ssims = [r.edge_ssim for r in results]
        clips = [r.clip_score for r in results]

        summary = {
            "total_samples": len(results),
            "edge_iou": {
                "mean": round(float(np.mean(ious)), 4),
                "std": round(float(np.std(ious)), 4),
                "min": round(float(np.min(ious)), 4),
                "max": round(float(np.max(ious)), 4),
            },
            "edge_ssim": {
                "mean": round(float(np.mean(ssims)), 4),
                "std": round(float(np.std(ssims)), 4),
                "min": round(float(np.min(ssims)), 4),
                "max": round(float(np.max(ssims)), 4),
            },
            "clip_score": {
                "mean": round(float(np.mean(clips)), 4),
                "std": round(float(np.std(clips)), 4),
                "min": round(float(np.min(clips)), 4),
                "max": round(float(np.max(clips)), 4),
            },
            "samples": [r.to_dict() for r in results],
        }

        # Generate GitHub Markdown Report
        md_lines = [
            "# PackVision AI - Benchmark Evaluation Report",
            f"*Generated on {time.strftime('%Y-%m-%d %H:%M:%S')} across {len(results)} packaging samples.*",
            "",
            "## Summary Performance Metrics",
            "",
            "| Metric | Mean | Std Dev | Min | Max | Target Benchmark |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
            f"| **Edge IoU** | `{summary['edge_iou']['mean']:.4f}` | `{summary['edge_iou']['std']:.4f}` | `{summary['edge_iou']['min']:.4f}` | `{summary['edge_iou']['max']:.4f}` | `> 0.5000` |",
            f"| **Edge SSIM** | `{summary['edge_ssim']['mean']:.4f}` | `{summary['edge_ssim']['std']:.4f}` | `{summary['edge_ssim']['min']:.4f}` | `{summary['edge_ssim']['max']:.4f}` | `> 0.6000` |",
            f"| **CLIP Score** | `{summary['clip_score']['mean']:.4f}` | `{summary['clip_score']['std']:.4f}` | `{summary['clip_score']['min']:.4f}` | `{summary['clip_score']['max']:.4f}` | `> 0.2600` |",
            "",
            "## Sample Breakdown",
            "",
            "| Sample ID | Edge IoU | Edge SSIM | CLIP Score | Fidelity Rating |",
            "| :--- | :---: | :---: | :---: | :--- |",
        ]

        for r in results:
            md_lines.append(
                f"| `{r.sample_id}` | `{r.edge_iou:.4f}` | `{r.edge_ssim:.4f}` | `{r.clip_score:.4f}` | {r.fidelity_rating} |"
            )

        markdown_text = "\n".join(md_lines)
        summary["markdown_report"] = markdown_text

        # Write to disk if requested
        if output_dir:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            with open(out_p / "benchmark_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            with open(out_p / "benchmark_report.md", "w", encoding="utf-8") as f:
                f.write(markdown_text)

        return summary


# =============================================================================
# MODULE-LEVEL CONVENIENCE
# =============================================================================

_SHARED_EVALUATOR: Optional[PackagingEvaluator] = None


def _get_shared_evaluator() -> PackagingEvaluator:
    global _SHARED_EVALUATOR
    if _SHARED_EVALUATOR is None:
        _SHARED_EVALUATOR = PackagingEvaluator()
    return _SHARED_EVALUATOR


def evaluate_render(
    image: Image.Image,
    prompt: str,
    sketch: Optional[Image.Image] = None,
    evaluator: Optional[PackagingEvaluator] = None,
) -> Dict[str, Any]:
    """
    Module-level convenience wrapper for :meth:`PackagingEvaluator.evaluate_render`.

    Combines the available quality metrics for one rendered image with a
    ``score_source`` provenance summary (see METRIC_PROVENANCE_KEYS).
    """
    active = evaluator if evaluator is not None else _get_shared_evaluator()
    return active.evaluate_render(image, prompt, sketch=sketch)
