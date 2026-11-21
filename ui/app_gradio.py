"""
PackVision AI - ML Workbench & Rapid Experimentation Suite
=========================================================
Standalone interactive Gradio application for packaging designers, ML researchers,
and brand creative directors to transform 2D packaging sketches into photoreal mockups.

Features:
- Canvas sketch upload & brush drawing / touch-up tool
- Industrial packaging taxonomy selectors (Category, Substrate, Finish, Lighting, Viewpoint, Backdrop)
- Vector logo overlay with projective homography & CIELAB luminance blending
- Interactive Before / After Split Comparison Slider
- Quantitative evaluation metrics (Edge SSIM fidelity, CLIP alignment score, inference time)
"""

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

# Self-relauncher: if executed with global python lacking dependencies,
# automatically delegate to the local virtual environment.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VENV_PYTHON = _PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
if _VENV_PYTHON.exists() and Path(sys.executable).resolve() != _VENV_PYTHON.resolve():
    try:
        import gradio
        import cv2
    except ImportError:
        print(f"[PackVision AI] Redirecting execution to virtual environment: {_VENV_PYTHON}")
        sys.exit(subprocess.run([str(_VENV_PYTHON)] + sys.argv).returncode)

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import gradio as gr
import numpy as np
from PIL import Image, ImageOps

# Ensure project root is in sys.path
PROJECT_ROOT = _PROJECT_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe imports for PackVision Core ML modules
try:
    from src.pipeline import PackVisionRenderer, RenderResult
    from src.postprocessor import LogoCompositor
    from src.prompt_engine import PromptEngine, build_packaging_prompt
    from src.evaluation import PackagingEvaluator, compute_edge_ssim, compute_edge_iou
    CORE_AVAILABLE = True
except ImportError as err:
    print(f"[Warning] Core ML engine modules import warning: {err}")
    CORE_AVAILABLE = False

# =============================================================================
# CANONICAL TAXONOMY (src/taxonomy.py) with local fallbacks
# =============================================================================
# The dropdowns below are built from the canonical packaging taxonomy shared
# with the backend API and the prompt engine. If src.taxonomy cannot be
# imported (fresh checkout, partial install), the hardcoded lists below keep
# the workbench usable — they use the same canonical key vocabulary.

_FALLBACK_PACKAGE_TYPES: Dict[str, str] = {
    "folding_carton": "Folding Carton / Box",
    "rigid_box": "Rigid Gift Box",
    "mailer_box": "Mailer Box",
    "stand_up_pouch": "Stand-Up Pouch",
    "flat_pouch": "Flat Pouch",
    "bottle": "Bottle",
    "jar": "Jar",
    "can": "Can",
    "tube": "Squeeze Tube",
    "sleeve": "Sleeve",
}
_FALLBACK_MATERIALS: Dict[str, str] = {
    "kraft_cardboard": "Kraft Cardboard",
    "corrugated_cardboard": "Corrugated Cardboard",
    "coated_paperboard": "Coated Paperboard (SBS)",
    "matte_laminated": "Matte Laminated Board",
    "uncoated_paper": "Uncoated Paper",
    "flexible_mylar": "Flexible Mylar Film",
    "frosted_glass": "Frosted Glass",
    "clear_glass": "Clear Glass",
    "amber_glass": "Amber Glass",
    "brushed_aluminum": "Brushed Aluminum",
    "tinplate_metal": "Tinplate Metal",
    "molded_pulp": "Molded Pulp",
}
_FALLBACK_FINISHES: Dict[str, str] = {
    "matte": "Matte Soft-Touch",
    "gloss": "High Gloss UV Varnish",
    "satin": "Satin Sheen",
    "spot_uv": "Spot UV Varnish",
    "gold_foil": "Gold Foil Stamping",
    "silver_foil": "Silver Foil Stamping",
    "holographic_film": "Holographic Film",
    "embossed": "Embossed Relief",
    "debossed": "Debossed Relief",
    "soft_touch": "Soft-Touch Coating",
    "uncoated": "Uncoated Natural",
}
_FALLBACK_LIGHTING: Dict[str, str] = {
    "studio_softbox": "Dual Studio Softbox",
    "dramatic_rim": "Dramatic Rim Lighting",
    "high_key_commercial": "Commercial High-Key",
    "natural_daylight": "Natural Daylight",
    "dark_luxury": "Dark Luxury Ambiance",
}
_FALLBACK_CAMERA_ANGLES: Dict[str, str] = {
    "front_view": "Front Eye-Level",
    "three_quarter_view": "Three-Quarter (3/4)",
    "top_down": "Top-Down Flat Lay",
    "hero_low_angle": "Hero Low-Angle",
}
_FALLBACK_BACKGROUNDS: Dict[str, str] = {
    "seamless_studio": "Clean Studio Infinity",
    "warm_pedestal": "Warm Pedestal Display",
    "dark_luxury_scene": "Dark Luxury Stone",
    "natural_environment": "Natural Environment",
    "transparent_isolated": "Transparent / Isolated",
}

_TAXONOMY_SOURCE = "fallback"
try:  # Defensive import — the app must still start if src.taxonomy is absent.
    from src.taxonomy import (  # type: ignore
        PACKAGE_TYPES,
        MATERIALS,
        FINISHES,
        LIGHTING,
        CAMERA_ANGLES,
        BACKGROUNDS,
        ensure_valid as _taxonomy_ensure_valid,
    )
    _TAXONOMY_SOURCE = "canonical"
except Exception as _tax_err:  # noqa: BLE001 - keep local fallback lists
    print(f"[Warning] Canonical taxonomy unavailable ({_tax_err}); using built-in fallback lists.")
    PACKAGE_TYPES = _FALLBACK_PACKAGE_TYPES
    MATERIALS = _FALLBACK_MATERIALS
    FINISHES = _FALLBACK_FINISHES
    LIGHTING = _FALLBACK_LIGHTING
    CAMERA_ANGLES = _FALLBACK_CAMERA_ANGLES
    BACKGROUNDS = _FALLBACK_BACKGROUNDS

    def _taxonomy_ensure_valid(category: str, value: str, default: str) -> Tuple[str, bool]:
        return (value or default), False


def _label_of(entry: Any, key: str) -> str:
    """Extract a human label from a taxonomy entry (dict) or plain string."""
    if isinstance(entry, dict):
        return str(entry.get("label") or key.replace("_", " ").title())
    return str(entry) if entry else key.replace("_", " ").title()


def _build_choices(table: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Return gr.Dropdown choices as (label, canonical_key) tuples, newest taxonomy first."""
    return [(_label_of(ent, key), key) for key, ent in table.items()]


PACKAGE_TYPE_CHOICES: List[Tuple[str, str]] = _build_choices(PACKAGE_TYPES) or [
    (_label_of(v, k), k) for k, v in _FALLBACK_PACKAGE_TYPES.items()
]
MATERIAL_CHOICES: List[Tuple[str, str]] = _build_choices(MATERIALS) or [
    (_label_of(v, k), k) for k, v in _FALLBACK_MATERIALS.items()
]
FINISH_CHOICES: List[Tuple[str, str]] = _build_choices(FINISHES) or [
    (_label_of(v, k), k) for k, v in _FALLBACK_FINISHES.items()
]
LIGHTING_CHOICES: List[Tuple[str, str]] = _build_choices(LIGHTING) or [
    (_label_of(v, k), k) for k, v in _FALLBACK_LIGHTING.items()
]
CAMERA_CHOICES: List[Tuple[str, str]] = _build_choices(CAMERA_ANGLES) or [
    (_label_of(v, k), k) for k, v in _FALLBACK_CAMERA_ANGLES.items()
]
BACKGROUND_CHOICES: List[Tuple[str, str]] = _build_choices(BACKGROUNDS) or [
    (_label_of(v, k), k) for k, v in _FALLBACK_BACKGROUNDS.items()
]


def _choice_keys(choices: List[Tuple[str, str]]) -> List[str]:
    return [key for _label, key in choices]


def _default_key(choices: List[Tuple[str, str]], category: str, preferred: str) -> str:
    """Pick a dropdown default that is guaranteed to exist in the choice list."""
    keys = _choice_keys(choices)
    try:
        resolved = _taxonomy_ensure_valid(category, preferred, keys[0])[0]
    except Exception:  # noqa: BLE001 - defaulting must never crash import
        resolved = preferred
    return resolved if resolved in keys else keys[0]


DEFAULT_CATEGORY_KEY = _default_key(PACKAGE_TYPE_CHOICES, "package_types", "folding_carton")
DEFAULT_MATERIAL_KEY = _default_key(MATERIAL_CHOICES, "materials", "kraft_cardboard")
DEFAULT_FINISH_KEY = _default_key(FINISH_CHOICES, "finishes", "gold_foil")
DEFAULT_LIGHTING_KEY = _default_key(LIGHTING_CHOICES, "lighting", "studio_softbox")
DEFAULT_CAMERA_KEY = _default_key(CAMERA_CHOICES, "camera_angles", "three_quarter_view")
DEFAULT_BACKGROUND_KEY = _default_key(BACKGROUND_CHOICES, "backgrounds", "seamless_studio")


def canonical_value(category: str, value: str, default: str) -> str:
    """Resolve any user value to its canonical taxonomy key (defensively)."""
    raw = (value or "").strip()
    try:
        resolved, _was_normalized = _taxonomy_ensure_valid(category, raw, default)
        return resolved or default
    except Exception:  # noqa: BLE001 - normalization must never break a render
        return (raw or default).lower().replace(" ", "_")


# =============================================================================
# ENGINE SELECTION (accurate v2 naming) & MODEL-VOCAB ALIASES
# =============================================================================
# Engine choices map onto pipeline.render(engine_mode=...) dispatch:
#   "auto"       -> cloud first (if configured), then PackVisionNet-v2, then GPU SDXL, then Procedural PBR
#   "custom"     -> force PackVisionNet-v2 (local neural checkpoint)
#   "procedural" -> force the deterministic PBR shader engine (CPU-safe)
#   "cloud"      -> force the cloud engine (HuggingFace Flux Kontext free tier / Stability /
#                   Gemini / OpenRouter / OpenAI), falls back locally with a notice
#   "sdxl"       -> SDXL + ControlNet (CUDA GPUs only; CPU machines get a visible fallback warning)
ENGINE_CHOICES: List[str] = [
    "Auto (best available)",
    "PackVisionNet-v2 (local)",
    "Procedural PBR (local)",
    "Cloud Photoreal (Flux Kontext — free, ~45s)",
    "Cloud (Stability/Gemini/OpenRouter/OpenAI — paid)",
    "SDXL + ControlNet (GPU only)",
]
DEFAULT_ENGINE_CHOICE = ENGINE_CHOICES[0]


def resolve_engine_mode(engine_choice: str) -> Tuple[str, bool]:
    """Map the UI engine choice onto (engine_mode, use_custom_model) for pipeline.render."""
    choice = (engine_choice or "").lower()
    if "packvisionnet" in choice:
        return "custom", True
    if "procedural" in choice:
        return "procedural", False
    if "cloud" in choice:
        return "cloud", False
    if "sdxl" in choice:
        return "sdxl", False
    return "auto", False


# The local PackVisionNet-v2 embedding tables are keyed by the legacy training
# vocabulary; alias canonical keys onto it so the conditional inputs stay
# meaningful instead of silently defaulting to class 0.
_MODEL_CATEGORY_ALIASES: Dict[str, str] = {
    "folding_carton": "carton",
    "rigid_box": "box",
    "mailer_box": "box",
    "stand_up_pouch": "pouch",
    "flat_pouch": "pouch",
}
_MODEL_MATERIAL_ALIASES: Dict[str, str] = {
    "corrugated_cardboard": "corrugated_box",
    "coated_paperboard": "coated_sbs",
    "matte_laminated": "matte_carton",
    "uncoated_paper": "kraft_cardboard",
    "flexible_mylar": "flexible_pouch",
    "clear_glass": "frosted_glass",
    "amber_glass": "frosted_glass",
    "tinplate_metal": "brushed_aluminum",
}
_MODEL_FINISH_ALIASES: Dict[str, str] = {
    "gloss": "matte",
    "satin": "matte",
    "soft_touch": "matte",
    "holographic_film": "holographic",
}


def humanize_engine(engine_used: Optional[str]) -> str:
    """Human-readable engine attribution for a pipeline engine_used token."""
    value = (engine_used or "").strip()
    if not value:
        return "Unknown engine"
    v = value.lower()
    if v.startswith("cloud"):
        provider = value.split("-", 1)[-1].replace("_", " ").strip()
        return f"Cloud · {provider.title() if provider else 'Provider'}"
    if "packvisionnet" in v:
        return "Local · PackVisionNet-v2"
    if "procedural" in v:
        return "Local · Procedural PBR"
    if "sdxl" in v or "controlnet" in v:
        return "GPU · SDXL + ControlNet"
    return value.replace("-", " ").replace("_", " ").title()

# Lazy-loaded singleton engine instances
_RENDERER: Optional[PackVisionRenderer] = None
_COMPOSITOR: Optional[LogoCompositor] = None
_EVALUATOR: Optional[PackagingEvaluator] = None


def get_renderer() -> PackVisionRenderer:
    global _RENDERER
    if _RENDERER is None:
        use_mock = os.getenv("USE_MOCK_INFERENCE", "true").lower() in ("true", "1", "yes")
        _RENDERER = PackVisionRenderer(use_mock_inference=use_mock)
    return _RENDERER


def get_compositor() -> LogoCompositor:
    global _COMPOSITOR
    if _COMPOSITOR is None:
        _COMPOSITOR = LogoCompositor()
    return _COMPOSITOR


def get_evaluator() -> PackagingEvaluator:
    global _EVALUATOR
    if _EVALUATOR is None:
        _EVALUATOR = PackagingEvaluator()
    return _EVALUATOR


def extract_pil_from_canvas(sketch_input: Any) -> Optional[Image.Image]:
    """Extract standard PIL Image from Gradio ImageEditor or Image component."""
    if sketch_input is None:
        return None
    if isinstance(sketch_input, Image.Image):
        return sketch_input
    if isinstance(sketch_input, dict):
        # Gradio ImageEditor returns dict with 'composite', 'background', 'layers'
        composite = sketch_input.get("composite")
        if composite is not None:
            if isinstance(composite, Image.Image):
                return composite
            elif isinstance(composite, np.ndarray):
                return Image.fromarray(composite)
        bg = sketch_input.get("background")
        if bg is not None:
            if isinstance(bg, Image.Image):
                return bg
            elif isinstance(bg, np.ndarray):
                return Image.fromarray(bg)
    if isinstance(sketch_input, np.ndarray):
        return Image.fromarray(sketch_input)
    if isinstance(sketch_input, str) and os.path.exists(sketch_input):
        return Image.open(sketch_input)
    return None


def auto_enrich_prompt(
    category: str,
    material: str,
    finish: str,
    lighting: str,
    camera: str,
    background: str,
    user_prompt: str,
) -> str:
    """Generate an enriched engineering prompt based on selected taxonomy."""
    try:
        from src.prompt_engine import PromptEngine
        engine = PromptEngine()
        pkg = engine.build_prompt(
            category=canonical_value("package_types", category, "folding_carton"),
            substrate=canonical_value("materials", material, "kraft_cardboard"),
            finish=canonical_value("finishes", finish, "matte"),
            lighting=canonical_value("lighting", lighting, "studio_softbox"),
            camera_angle=canonical_value("camera_angles", camera, "three_quarter_view"),
            background=canonical_value("backgrounds", background, "seamless_studio"),
            user_prompt=user_prompt if user_prompt.strip() else None,
        )
        return pkg.positive_prompt
    except Exception:
        parts = [
            f"Commercial {category} packaging mockup",
            f"crafted from authentic {material}",
            f"featuring {finish} finish",
            f"lit by {lighting}",
            f"{camera} framing",
            f"set against {background}",
        ]
        if user_prompt.strip():
            parts.insert(1, user_prompt.strip())
        return ", ".join(parts) + ", 8k resolution, raytraced ambient occlusion, master product photo."


def invert_sketch_lines(sketch_input: Any) -> Any:
    """Invert sketch image contrast (black on white <-> white on black)."""
    pil_img = extract_pil_from_canvas(sketch_input)
    if pil_img is None:
        return sketch_input
    if pil_img.mode in ("RGBA", "LA"):
        # Separate alpha
        r, g, b, a = pil_img.split()
        rgb_img = Image.merge("RGB", (r, g, b))
        inv_rgb = ImageOps.invert(rgb_img)
        r2, g2, b2 = inv_rgb.split()
        inverted = Image.merge("RGBA", (r2, g2, b2, a))
    else:
        inverted = ImageOps.invert(pil_img.convert("RGB"))
    return inverted


def run_generation(
    sketch_input: Any,
    category: str,
    material: str,
    finish: str,
    lighting: str,
    camera: str,
    background: str,
    user_prompt: str,
    negative_prompt: str,
    logo_input: Optional[Any],
    logo_pos_x: float,
    logo_pos_y: float,
    logo_scale: float,
    blend_mode: str,
    steps: int,
    guidance_scale: float,
    controlnet_scale: float,
    seed: int,
    engine_choice: str = DEFAULT_ENGINE_CHOICE,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[Any, str, str, str, str, str]:
    """Execute end-to-end rendering pipeline and return slider comparison + metrics + engine attribution."""
    pil_sketch = extract_pil_from_canvas(sketch_input)
    if pil_sketch is None:
        raise gr.Error("Please upload or draw a packaging sketch on the canvas first.")

    progress(0.1, desc="Queuing render...")
    time.sleep(0.05)

    effective_seed = None if seed <= 0 else seed

    # 1. Pipeline Render — accurate engine dispatch for the v2 engine suite
    engine_mode, use_custom = resolve_engine_mode(engine_choice)
    if not CORE_AVAILABLE:
        raise gr.Error(
            "Core ML engine unavailable (src.pipeline import failed). "
            "Check the virtual environment and that src/taxonomy.py is present."
        )

    progress(0.3, desc=f"Rendering {material} with {engine_choice}...")
    renderer = get_renderer()

    # Canonical taxonomy keys drive the prompt engine; alias them onto the
    # local PackVisionNet-v2 training vocabulary so conditional embeddings stay valid.
    canon_category = canonical_value("package_types", category, "folding_carton")
    canon_material = canonical_value("materials", material, "kraft_cardboard")
    canon_finish = canonical_value("finishes", finish, "matte")
    canon_lighting = canonical_value("lighting", lighting, "studio_softbox")
    canon_camera = canonical_value("camera_angles", camera, "three_quarter_view")
    canon_background = canonical_value("backgrounds", background, "seamless_studio")

    model_category = _MODEL_CATEGORY_ALIASES.get(canon_category, canon_category)
    model_material = _MODEL_MATERIAL_ALIASES.get(canon_material, canon_material)
    model_finish = _MODEL_FINISH_ALIASES.get(canon_finish, canon_finish)

    cleaned_negative = (negative_prompt or "").strip()

    render_kwargs = dict(
        sketch=pil_sketch,
        category=model_category,
        substrate=model_material,
        finish=model_finish,
        lighting=canon_lighting,
        camera_angle=canon_camera,
        background=canon_background,
        user_prompt=user_prompt if user_prompt.strip() else None,
        negative_prompt=cleaned_negative,
        conditioning_scale=controlnet_scale,
        seed=effective_seed,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        use_custom_model=use_custom,
        engine_mode=engine_mode,
    )

    render_result: RenderResult
    try:
        # Pass the designer's negative prompt through to the renderer (wired end-to-end).
        render_result = renderer.render(**render_kwargs)
    except TypeError:
        # Older pipeline signature without explicit negative_prompt support.
        render_kwargs.pop("negative_prompt", None)
        render_result = renderer.render(**render_kwargs)

    final_render = render_result.image
    orig_normalized = render_result.preprocessed_sketch

    # 2. Stage-2 Homography Logo Compositing (if logo provided)
    if logo_input is not None:
        progress(0.75, desc="Compositing brand logo...")
        pil_logo = extract_pil_from_canvas(logo_input)
        if pil_logo is not None:
            compositor = get_compositor()
            # Calculate quadrilateral destination coordinates from slider offsets
            w, h = final_render.size
            cx = (logo_pos_x / 100.0) * w
            cy = (logo_pos_y / 100.0) * h
            box_w = (logo_scale / 100.0) * (w * 0.45)
            box_h = box_w * (pil_logo.height / max(pil_logo.width, 1))

            dest_points = [
                (cx - box_w / 2, cy - box_h / 2),
                (cx + box_w / 2, cy - box_h / 2),
                (cx + box_w / 2, cy + box_h / 2),
                (cx - box_w / 2, cy + box_h / 2),
            ]

            comp_res = compositor.composite(
                package_image=final_render,
                logo_image=pil_logo,
                destination_points=dest_points,
                blend_mode=blend_mode.lower(),
                opacity=0.95,
            )
            final_render = comp_res.image

    progress(0.9, desc="Evaluating fidelity...")

    # 3. Quantitative Evaluation (with metric provenance)
    evaluator = get_evaluator()
    ssim_val = compute_edge_ssim(orig_normalized, final_render)
    iou_val = compute_edge_iou(orig_normalized, final_render)

    clip_prompt = (
        f"{canon_category} packaging in {canon_material} with {canon_finish} finish, {user_prompt}"
    )
    clip_val: float
    clip_source = "heuristic"
    try:
        clip_detail = evaluator.compute_clip_score_detailed(final_render, clip_prompt)
        clip_val = float(clip_detail.get("score"))
        clip_source = str(clip_detail.get("source") or "heuristic")
    except AttributeError:
        # Older evaluator without detailed provenance — assume the local fallback heuristic.
        clip_val = evaluator.compute_clip_score(final_render, clip_prompt)
        clip_source = "heuristic"
    except Exception:  # noqa: BLE001 - evaluation must never break a finished render
        clip_val = evaluator.compute_clip_score(final_render, clip_prompt)
        clip_source = "heuristic"

    # Metric provenance may also arrive from the pipeline metadata (engine-level
    # provenance, e.g. "checkpoint_training_history" for neural checkpoint quality).
    meta: Dict[str, Any] = render_result.metadata or {}
    engine_metrics_source = str(meta.get("metrics_source") or "").strip()

    progress(1.0, desc="Render complete")

    # ---------------------------------------------------------------------------
    # 4. Metric labels with provenance: heuristic values are explicitly tagged
    # ---------------------------------------------------------------------------
    ESTIMATE_SOURCES = {"heuristic", "vlm", "estimate", "estimated", "proxy", "pseudo"}
    source_suffix = {
        "heuristic": "heuristic estimate",
        "vlm": "VLM estimate",
        "estimate": "estimate",
        "estimated": "estimate",
        "proxy": "proxy estimate",
        "pseudo": "proxy estimate",
        "library": "computed",
        "checkpoint_training_history": "training-history reference",
        "unavailable": "unavailable",
    }.get(clip_source, clip_source)

    ssim_label = (
        f"Edge SSIM: {ssim_val:.3f} "
        f"({'Excellent' if ssim_val > 0.8 else 'Good' if ssim_val > 0.65 else 'Moderate'})"
    )
    iou_label = f"Edge IoU: {iou_val:.3f}"
    if source_suffix:
        clip_label = f"CLIP Alignment: {clip_val:.3f} ({source_suffix})"
    else:
        clip_label = f"CLIP Alignment: {clip_val:.3f}"

    # ---------------------------------------------------------------------------
    # 5. Engine attribution badge + honest fallback warnings
    # ---------------------------------------------------------------------------
    engine_used = (
        meta.get("engine_used")
        or getattr(renderer, "last_engine_used", None)
        or ("procedural-pbr" if render_result.is_mock else "")
    )
    engine_label = humanize_engine(engine_used)

    fallback_warnings: List[str] = []
    if engine_mode == "sdxl":
        if not getattr(renderer, "has_cuda", False):
            fallback_warnings.append(
                "No CUDA GPU detected — 'SDXL + ControlNet' is GPU-only; fell back to "
                f"{engine_label}."
            )
        elif "sdxl" not in str(engine_used).lower():
            fallback_warnings.append(
                f"'SDXL + ControlNet' requested but the engine fell back to {engine_label}."
            )
    if engine_mode == "cloud" and not str(engine_used).lower().startswith("cloud"):
        reason = meta.get("engine_note") or meta.get("cloud_fallback_reason") or "cloud engine unavailable"
        fallback_warnings.append(f"Cloud engine unavailable ({reason}); rendered locally with {engine_label}.")
    if engine_mode == "custom" and "packvisionnet" not in str(engine_used).lower():
        fallback_warnings.append(
            f"PackVisionNet-v2 requested but its checkpoint was unavailable; fell back to {engine_label}."
        )
    if engine_mode == "auto" and meta.get("engine_note"):
        fallback_warnings.append(str(meta["engine_note"]))

    engine_badge_md = f"**Rendered by:** {engine_label}"
    if fallback_warnings:
        engine_badge_md += "  \n" + "  \n".join(fallback_warnings)
    if engine_metrics_source and engine_metrics_source.lower() != "unavailable":
        engine_badge_md += (
            "  \nMetric provenance: " + engine_metrics_source.replace("_", " ")
        )
    if meta.get("cloud_cost_note"):
        engine_badge_md += f"  \nCost note: {meta['cloud_cost_note']}"

    latency_label = f"Time: {render_result.inference_time:.2f}s · {engine_label}"

    # Gradio ImageSlider accepts (before, after) tuple
    slider_output = (orig_normalized, final_render)

    return slider_output, ssim_label, iou_label, clip_label, latency_label, engine_badge_md


# =============================================================================
# GRADIO INTERFACE CONSTRUCTION
# =============================================================================

# =============================================================================
# MINIMAL DESIGN LANGUAGE — theme + css (shared tokens with the web frontend)
# =============================================================================
# Pinterest-clean workbench: white canvas, quiet neutral grays, a single red
# accent (#E60023), Plus Jakarta Sans, rounded-xl components, hairline borders,
# flat surfaces and whitespace-driven compact spacing.

_PACKVISION_RED = gr.themes.Color(
    c50="#fff1f2", c100="#ffe0e5", c200="#ffc4ce", c300="#ff8fa1",
    c400="#ff4d66", c500="#e60023", c600="#cf001f", c700="#ab0019",
    c800="#870014", c900="#5c000d", c950="#340007",
)
_NEUTRAL_GRAY = gr.themes.Color(
    c50="#fafafa", c100="#f5f5f5", c200="#e5e5e5", c300="#d4d4d4",
    c400="#a3a3a3", c500="#737373", c600="#525252", c700="#404040",
    c800="#262626", c900="#171717", c950="#0a0a0a",
)


def _build_minimal_theme() -> gr.themes.Base:
    """Minimal theme: red accent, quiet grays, rounded-xl, compact spacing."""
    return gr.themes.Base(
        primary_hue=_PACKVISION_RED,
        secondary_hue=_PACKVISION_RED,
        neutral_hue=_NEUTRAL_GRAY,
        font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
        text_size=gr.themes.sizes.text_md,
        spacing_size=gr.themes.sizes.spacing_sm,
        radius_size=gr.themes.sizes.radius_md,  # xl slot = 12px (Tailwind rounded-xl)
    )


MINIMAL_THEME = _build_minimal_theme()

CUSTOM_CSS = """
/* PackVision minimal workbench — white canvas, quiet grays, one red accent */
body, .gradio-container {
    background: #ffffff;
    color: #171717;
}
.gradio-container {
    --body-background-fill: #ffffff;
    --background-fill-primary: #ffffff;
    --background-fill-secondary: #fafafa;
    --block-background-fill: #ffffff;
    --input-background-fill: #ffffff;
    --panel-background-fill: #ffffff;
    --block-border-color: #e5e5e5;
    --input-border-color: #e5e5e5;
    --border-color-primary: #e5e5e5;
    --border-color-accent: #e60023;
    --block-border-width: 1px;
    --input-border-width: 1px;
    --input-border-color-focus: #e60023;
    --input-shadow: none;
    --input-shadow-focus: none;
    --block-shadow: none;
    --button-primary-shadow: none;
    --button-primary-shadow-hover: none;
    --button-primary-shadow-active: none;
    --button-secondary-shadow: none;
    --button-secondary-shadow-hover: none;
    --button-secondary-shadow-active: none;
    --shadow-drop: none;
    --shadow-drop-lg: none;
    --shadow-inset: none;
    --shadow-spread: 0px;
}
/* Dark mode: #111 page, #1C1C1C panels */
.dark, .dark .gradio-container, .gradio-container.dark {
    background: #111111;
    color: #e5e5e5;
    --body-background-fill: #111111;
    --background-fill-primary: #111111;
    --background-fill-secondary: #1a1a1a;
    --block-background-fill: #1c1c1c;
    --input-background-fill: #1c1c1c;
    --panel-background-fill: #1c1c1c;
    --block-border-color: #262626;
    --input-border-color: #262626;
    --border-color-primary: #262626;
}

footer { display: none !important; }

/* Brand header */
.brand-header {
    border-bottom: 1px solid #f5f5f5;
    padding-bottom: 14px;
    margin-bottom: 8px;
}
.dark .brand-header { border-bottom-color: #262626; }
.brand-logo-text {
    font-size: 20px;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: #171717;
    background: none;
    -webkit-text-fill-color: currentColor;
    display: inline-flex;
    align-items: center;
    gap: 10px;
}
.dark .brand-logo-text { color: #f5f5f5; }
.badge-tag {
    display: inline-block;
    padding: 2px 8px;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border-radius: 9999px;
    background: #f5f5f5;
    color: #737373;
    border: none;
}
.dark .badge-tag { background: #262626; color: #a3a3a3; }

/* Section headers: tiny, quiet, uppercase */
.gradio-container h3 {
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #a3a3a3 !important;
    margin: 12px 0 6px 0 !important;
}
.dark .gradio-container h3 { color: #737373 !important; }

/* Primary action: flat red, no gradient, no glow */
.generate-btn {
    background: #e60023 !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    box-shadow: none !important;
    transition: background 0.15s ease !important;
}
.generate-btn:hover {
    background: #cf001f !important;
    transform: none !important;
    box-shadow: none !important;
}

/* Metrics row: transparent, whitespace-driven */
.metrics-panel {
    background: transparent;
    border: none;
    border-radius: 12px;
    padding: 0;
    margin-top: 10px;
}
"""

with gr.Blocks(title="PackVision Workbench") as demo:

    with gr.Row(elem_classes=["brand-header"]):
        with gr.Column(scale=9):
            gr.HTML("""
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <div class="brand-logo-text">
                        <span>PackVision</span>
                        <span class="badge-tag">Workbench</span>
                    </div>
                    <p style="color: #a3a3a3; margin: 4px 0 0 0; font-size: 12px;">
                        Sketch-to-photoreal packaging renders &middot; ML experimentation
                    </p>
                </div>
            </div>
            """)
        with gr.Column(scale=3, min_width=200):
            gr.HTML("""
            <div style="text-align: right; color: #a3a3a3; font-size: 12px; padding-top: 8px;">
                <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #10b981; margin-right: 6px;"></span>
                Engine ready
            </div>
            """)

    with gr.Row():
        # LEFT COLUMN: INPUTS & CONTROLS
        with gr.Column(scale=5):
            gr.Markdown("### Sketch canvas")
            sketch_editor = gr.ImageEditor(
                type="pil",
                label="Upload or draw",
                sources=["upload", "clipboard"],
                brush=gr.Brush(colors=["#000000", "#ffffff", "#4b5563"], default_size=4),
                eraser=gr.Eraser(default_size=10),
                transforms=["crop"],
            )

            with gr.Row():
                btn_invert = gr.Button("Invert lines", size="sm", variant="secondary")

            gr.Markdown(
                f"### Packaging setup\n"
                f"<sub style='color:#a3a3a3;'>Vocabulary source: "
                f"{'src/taxonomy.py' if _TAXONOMY_SOURCE == 'canonical' else 'built-in fallback (src.taxonomy unavailable)'}</sub>"
            )
            with gr.Row():
                category_dropdown = gr.Dropdown(
                    choices=PACKAGE_TYPE_CHOICES,
                    value=DEFAULT_CATEGORY_KEY,
                    label="Category",
                )
                material_dropdown = gr.Dropdown(
                    choices=MATERIAL_CHOICES,
                    value=DEFAULT_MATERIAL_KEY,
                    label="Material",
                )

            with gr.Row():
                finish_dropdown = gr.Dropdown(
                    choices=FINISH_CHOICES,
                    value=DEFAULT_FINISH_KEY,
                    label="Finish",
                )
                lighting_dropdown = gr.Dropdown(
                    choices=LIGHTING_CHOICES,
                    value=DEFAULT_LIGHTING_KEY,
                    label="Lighting",
                )

            with gr.Row():
                camera_dropdown = gr.Dropdown(
                    choices=CAMERA_CHOICES,
                    value=DEFAULT_CAMERA_KEY,
                    label="Viewpoint",
                )
                background_dropdown = gr.Dropdown(
                    choices=BACKGROUND_CHOICES,
                    value=DEFAULT_BACKGROUND_KEY,
                    label="Backdrop",
                )

            gr.Markdown("### Prompts")
            user_prompt_txt = gr.Textbox(
                label="Prompt",
                placeholder="e.g. Luxury herbal botanical tea packaging, minimalist gold typography, elegant aesthetic",
                value="Luxury artisanal tea box, gold foil stamped botanical motifs, minimalist aesthetic",
                lines=2,
            )
            btn_enrich = gr.Button("Auto-enrich prompt", size="sm", variant="secondary")

            negative_prompt_txt = gr.Textbox(
                label="Negative Prompt",
                value="distorted geometry, broken dielines, uneven seams, low quality, warped text, cartoon, blur",
                lines=1,
            )

            # STAGE 2: LOGO COMPOSITING ACCORDION
            with gr.Accordion("Brand logo overlay (optional)", open=False):
                gr.Markdown("Warp a vector logo onto the pack face with CIELAB luminance blending.")
                logo_upload = gr.Image(
                    type="pil",
                    image_mode="RGBA",
                    label="Logo (PNG, transparent)",
                )
                with gr.Row():
                    logo_x_slider = gr.Slider(
                        minimum=10.0, maximum=90.0, value=50.0, step=1.0, label="Position X"
                    )
                    logo_y_slider = gr.Slider(
                        minimum=10.0, maximum=90.0, value=52.0, step=1.0, label="Position Y"
                    )
                with gr.Row():
                    logo_scale_slider = gr.Slider(
                        minimum=10.0, maximum=100.0, value=50.0, step=2.0, label="Logo scale"
                    )
                    blend_mode_dropdown = gr.Dropdown(
                        choices=["Normal", "Multiply", "Overlay", "Hard_Light"],
                        value="Multiply",
                        label="Blend mode",
                    )

            # ADVANCED CONTROLS ACCORDION
            with gr.Accordion("Advanced parameters", open=True):
                engine_dropdown = gr.Dropdown(
                    choices=ENGINE_CHOICES,
                    value=DEFAULT_ENGINE_CHOICE,
                    label="Render engine",
                    info=(
                        "Auto tries cloud first (if configured), then local neural, GPU SDXL, then PBR. "
                        "SDXL needs a CUDA GPU; on CPU it visibly falls back. "
                        "Cloud uses provider keys from your environment."
                    ),
                )
                with gr.Row():
                    steps_slider = gr.Slider(
                        minimum=10, maximum=50, value=30, step=1, label="Steps"
                    )
                    guidance_slider = gr.Slider(
                        minimum=1.0, maximum=15.0, value=7.5, step=0.5, label="Guidance"
                    )
                with gr.Row():
                    controlnet_slider = gr.Slider(
                        minimum=0.1, maximum=1.5, value=0.85, step=0.05, label="ControlNet strength"
                    )
                    seed_input = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)

            btn_generate = gr.Button(
                "Generate mockup",
                size="lg",
                variant="primary",
                elem_classes=["generate-btn"],
            )

        # RIGHT COLUMN: PREVIEW STAGE & EVALUATION
        with gr.Column(scale=7):
            gr.Markdown("### Before / after")
            slider_output = gr.ImageSlider(
                label="Sketch vs. render",
                type="pil",
                show_label=True,
            )

            engine_badge = gr.Markdown(
                value="**Rendered by:** awaiting first render…",
                label="Engine Attribution",
                show_label=False,
                elem_id="engine-attribution-badge",
            )

            with gr.Row(elem_classes=["metrics-panel"]):
                metric_ssim = gr.Textbox(label="Edge SSIM", value="Edge SSIM: --", interactive=False)
                metric_iou = gr.Textbox(label="Edge IoU", value="Edge IoU: --", interactive=False)
                metric_clip = gr.Textbox(label="CLIP score", value="CLIP Score: --", interactive=False)
                metric_time = gr.Textbox(label="Latency", value="Latency: --", interactive=False)

            # EXAMPLES GALLERY (canonical keys, guaranteed to exist in the dropdowns)
            gr.Markdown("### Samples")
            sample_dir = PROJECT_ROOT / "data" / "samples"
            cat_keys = _choice_keys(PACKAGE_TYPE_CHOICES)
            mat_keys = _choice_keys(MATERIAL_CHOICES)
            fin_keys = _choice_keys(FINISH_CHOICES)
            lit_keys = _choice_keys(LIGHTING_CHOICES)
            cam_keys = _choice_keys(CAMERA_CHOICES)
            bg_keys = _choice_keys(BACKGROUND_CHOICES)

            def _pick(keys: List[str], preferred: str) -> str:
                return preferred if preferred in keys else keys[0]

            carton_key = _pick(cat_keys, "folding_carton")
            pouch_key = _pick(cat_keys, "stand_up_pouch")
            bottle_key = _pick(cat_keys, "bottle")
            kraft_key = _pick(mat_keys, "kraft_cardboard")
            mylar_key = _pick(mat_keys, "flexible_mylar")
            frosted_key = _pick(mat_keys, "frosted_glass")
            gold_key = _pick(fin_keys, "gold_foil")
            matte_key = _pick(fin_keys, "matte")
            softbox_key = _pick(lit_keys, "studio_softbox")
            highkey_key = _pick(lit_keys, "high_key_commercial")
            rim_key = _pick(lit_keys, "dramatic_rim")
            front_key = _pick(cam_keys, "front_view")
            threeq_key = _pick(cam_keys, "three_quarter_view")
            studio_key = _pick(bg_keys, "seamless_studio")
            dark_key = _pick(bg_keys, "dark_luxury_scene")

            samples = [
                [
                    str(sample_dir / "carton_sketch.png"),
                    carton_key,
                    kraft_key,
                    gold_key,
                    softbox_key,
                    threeq_key,
                    studio_key,
                    "Luxury herbal botanical tea packaging, gold foil embossed",
                ],
                [
                    str(sample_dir / "pouch_sketch.png"),
                    pouch_key,
                    mylar_key,
                    matte_key,
                    highkey_key,
                    front_key,
                    studio_key,
                    "Artisanal roasted coffee stand-up pouch, minimalist earthy aesthetic",
                ],
                [
                    str(sample_dir / "bottle_sketch.png"),
                    bottle_key,
                    frosted_key,
                    gold_key,
                    rim_key,
                    front_key,
                    dark_key,
                    "Luxury facial serum dropper bottle, frosted glass, metallic dropper",
                ],
            ]

            gr.Examples(
                examples=samples,
                inputs=[
                    sketch_editor,
                    category_dropdown,
                    material_dropdown,
                    finish_dropdown,
                    lighting_dropdown,
                    camera_dropdown,
                    background_dropdown,
                    user_prompt_txt,
                ],
                label="Click a sample to load",
            )

    # EVENT WIRING
    btn_invert.click(
        fn=invert_sketch_lines,
        inputs=[sketch_editor],
        outputs=[sketch_editor],
    )

    btn_enrich.click(
        fn=auto_enrich_prompt,
        inputs=[
            category_dropdown,
            material_dropdown,
            finish_dropdown,
            lighting_dropdown,
            camera_dropdown,
            background_dropdown,
            user_prompt_txt,
        ],
        outputs=[user_prompt_txt],
    )

    btn_generate.click(
        fn=run_generation,
        inputs=[
            sketch_editor,
            category_dropdown,
            material_dropdown,
            finish_dropdown,
            lighting_dropdown,
            camera_dropdown,
            background_dropdown,
            user_prompt_txt,
            negative_prompt_txt,
            logo_upload,
            logo_x_slider,
            logo_y_slider,
            logo_scale_slider,
            blend_mode_dropdown,
            steps_slider,
            guidance_slider,
            controlnet_slider,
            seed_input,
            engine_dropdown,
        ],
        outputs=[
            slider_output,
            metric_ssim,
            metric_iou,
            metric_clip,
            metric_time,
            engine_badge,
        ],
    )


if __name__ == "__main__":
    base_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    host = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    
    # Gracefully scan for an available port if 7860 is bound
    for trial_port in range(base_port, base_port + 15):
        try:
            print(f"Launching PackVision AI Gradio Workbench on http://{host}:{trial_port}...")
            demo.launch(
                server_name=host,
                server_port=trial_port,
                share=False,
                theme=MINIMAL_THEME,
                css=CUSTOM_CSS,
            )
            break
        except OSError as e:
            if "10048" in str(e) or "already in use" in str(e).lower() or "address already in use" in str(e).lower():
                print(f"[PackVision AI] Port {trial_port} is already in use, trying port {trial_port + 1}...")
                continue
            raise
