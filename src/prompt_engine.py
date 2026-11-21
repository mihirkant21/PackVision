"""
PackVision AI - Packaging Prompt Engineering Engine
====================================================
Comprehensive packaging prompt generation system. Translates structural
packaging intent, substrates, industrial print finishes, lighting, camera angles,
and luxury studio environments into photorealistic SDXL conditioning prompts.

The canonical vocabulary (keys, labels, synonyms, prompt fragments) lives in
``src/taxonomy.py`` and is shared with the backend API and the Gradio UI. The
module-level dictionaries below (``CATEGORIES``, ``SUBSTRATES``, ``FINISHES``,
``LIGHTING_PRESETS``, ``CAMERA_ANGLES``, ``BACKGROUNDS``) are derived views kept
for backward compatibility with existing importers (``ai/__init__.py``,
``data/dataset_generator.py``). Every incoming user value is resolved through
``src.taxonomy.normalize`` so underscore/space/label variants all resolve to the
same canonical key instead of silently falling back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src import taxonomy as _taxonomy
from src.taxonomy import ensure_valid


# =============================================================================
# DESCRIPTOR DATATYPES
# =============================================================================

@dataclass(frozen=True)
class CategoryDescriptor:
    """Structural geometry and physical characteristics for packaging forms."""
    name: str
    display_name: str
    structural_cues: str
    negative_cues: str


@dataclass(frozen=True)
class SubstrateDescriptor:
    """Physical material shaders, optical properties, and micro-surface textures."""
    name: str
    display_name: str
    shader_description: str
    texture_cues: str
    specular_reflection: str


@dataclass(frozen=True)
class FinishDescriptor:
    """Industrial embellishments, foiling, varnishes, and tactile relief finishes."""
    name: str
    display_name: str
    specular_profile: str
    relief_depth: str
    visual_highlights: str


@dataclass(frozen=True)
class LightingDescriptor:
    """Professional studio lighting rigs, shadow soft-fall, and specular rim setups."""
    name: str
    display_name: str
    lighting_setup: str
    shadow_profile: str


@dataclass(frozen=True)
class CameraDescriptor:
    """Optical focal length, perspective distortion, and composition framing."""
    name: str
    display_name: str
    framing: str
    focal_depth: str


@dataclass(frozen=True)
class BackgroundDescriptor:
    """Studio cyclorama, architectural pedestals, and luxury surfaces."""
    name: str
    display_name: str
    surface_description: str
    contact_shadows: str


@dataclass
class PromptPackage:
    """Container for generated positive and negative prompt strings."""
    positive_prompt: str
    negative_prompt: str
    category: str
    substrate: str
    finish: str
    lighting: str
    camera_angle: str
    background: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# TAXONOMY DERIVED VIEWS (backwards-compatible alias dictionaries)
# =============================================================================
# These are built from the canonical taxonomy in src/taxonomy.py and are keyed
# by canonical keys. Existing code that imports these names keeps working; the
# keys are now shared vocabulary instead of private copies.

CATEGORIES: Dict[str, CategoryDescriptor] = {
    key: CategoryDescriptor(
        name=key,
        display_name=ent["label"],
        structural_cues=ent["prompt_fragment"],
        negative_cues=ent.get("negative_cues", ""),
    )
    for key, ent in _taxonomy.PACKAGE_TYPES.items()
}

SUBSTRATES: Dict[str, SubstrateDescriptor] = {
    key: SubstrateDescriptor(
        name=key,
        display_name=ent["label"],
        shader_description=ent["shader_description"],
        texture_cues=ent["texture_cues"],
        specular_reflection=ent["specular_reflection"],
    )
    for key, ent in _taxonomy.MATERIALS.items()
}

FINISHES: Dict[str, FinishDescriptor] = {
    key: FinishDescriptor(
        name=key,
        display_name=ent["label"],
        specular_profile=ent["specular_profile"],
        relief_depth=ent["relief_depth"],
        visual_highlights=ent["visual_highlights"],
    )
    for key, ent in _taxonomy.FINISHES.items()
}

LIGHTING_PRESETS: Dict[str, LightingDescriptor] = {
    key: LightingDescriptor(
        name=key,
        display_name=ent["label"],
        lighting_setup=ent["lighting_setup"],
        shadow_profile=ent["shadow_profile"],
    )
    for key, ent in _taxonomy.LIGHTING.items()
}

CAMERA_ANGLES: Dict[str, CameraDescriptor] = {
    key: CameraDescriptor(
        name=key,
        display_name=ent["label"],
        framing=ent["framing"],
        focal_depth=ent["focal_depth"],
    )
    for key, ent in _taxonomy.CAMERA_ANGLES.items()
}

BACKGROUNDS: Dict[str, BackgroundDescriptor] = {
    key: BackgroundDescriptor(
        name=key,
        display_name=ent["label"],
        surface_description=ent["surface_description"],
        contact_shadows=ent["contact_shadows"],
    )
    for key, ent in _taxonomy.BACKGROUNDS.items()
}


# Canonical default keys used when an input is unrecognized.
_DEFAULT_CATEGORY = "folding_carton"
_DEFAULT_SUBSTRATE = "coated_paperboard"
_DEFAULT_FINISH = "matte"
_DEFAULT_LIGHTING = "studio_softbox"
_DEFAULT_CAMERA = "three_quarter_view"
_DEFAULT_BACKGROUND = "seamless_studio"

# Input-name -> (taxonomy category, default canonical key)
_INPUT_CATEGORIES: Dict[str, Tuple[str, str]] = {
    "category": ("package_types", _DEFAULT_CATEGORY),
    "substrate": ("materials", _DEFAULT_SUBSTRATE),
    "finish": ("finishes", _DEFAULT_FINISH),
    "lighting": ("lighting", _DEFAULT_LIGHTING),
    "camera_angle": ("camera_angles", _DEFAULT_CAMERA),
    "background": ("backgrounds", _DEFAULT_BACKGROUND),
}

# Negative prompts preventing common generative packaging defects
DEFAULT_PACKAGING_NEGATIVE = (
    "deformed packaging, warped box geometry, distorted panels, crooked edges, skewed folds, "
    "melting corners, extra flaps, floating package, asymmetrical tabs, crumpled cardboard, "
    "crushed corners, bad packaging folds, low resolution, blurry details, cartoon drawing, "
    "unfinished sketch lines remaining, rough sketch scribbles, pencil marks, hand holding package, "
    "fingers, human body, cluttered background, noisy artifacts, watermarks, text errors, "
    "gibberish labels, illegible warped text, low poly 3D mesh, CGI plastic sheen, oversaturated"
)


def _resolve_input(raw_value: Optional[str], category: str, default_key: str) -> Tuple[str, Optional[str]]:
    """
    Resolve a raw input value to a canonical key.

    Returns:
        (canonical_key, raw_input_or_None): the second element is the original
        raw string when it was non-empty but could NOT be resolved (used for
        ``unrecognized_inputs`` reporting), otherwise None.
    """
    if raw_value is not None and isinstance(raw_value, str) and raw_value.strip():
        key, _was_normalized = ensure_valid(category, raw_value, default_key)
        if _taxonomy.normalize(raw_value, category) is not None:
            return key, None
        return key, raw_value
    return default_key, None


# =============================================================================
# PROMPT ENGINE CLASS & BUILDER FUNCTION
# =============================================================================

class PromptEngine:
    """
    Expert Prompt Engine for PackVision AI.

    Translates high-level packaging selections into fine-tuned, physically grounded
    conditioning prompts for Stable Diffusion XL ControlNet pipelines. All inputs
    are normalized through the shared taxonomy, so canonical keys, legacy ids,
    display labels, and space/underscore variants all resolve identically.
    """

    def __init__(self, base_negative: str = DEFAULT_PACKAGING_NEGATIVE) -> None:
        self.base_negative = base_negative

    @staticmethod
    def list_categories() -> List[str]:
        """List all supported packaging categories."""
        return list(CATEGORIES.keys())

    @staticmethod
    def list_substrates() -> List[str]:
        """List all supported packaging substrates."""
        return list(SUBSTRATES.keys())

    @staticmethod
    def list_finishes() -> List[str]:
        """List all supported print finishes."""
        return list(FINISHES.keys())

    @staticmethod
    def list_lighting() -> List[str]:
        """List all supported lighting presets."""
        return list(LIGHTING_PRESETS.keys())

    @staticmethod
    def list_camera_angles() -> List[str]:
        """List all supported camera angle presets."""
        return list(CAMERA_ANGLES.keys())

    @staticmethod
    def list_backgrounds() -> List[str]:
        """List all supported background environment presets."""
        return list(BACKGROUNDS.keys())

    def build_prompt(
        self,
        category: str = "carton",
        substrate: str = "coated SBS",
        finish: str = "matte",
        lighting: str = "studio softbox",
        camera_angle: str = "three-quarter",
        background: str = "seamless studio",
        user_prompt: Optional[str] = None,
        brand_name: Optional[str] = None,
        color_palette: Optional[str] = None,
        custom_negative: Optional[str] = None,
    ) -> PromptPackage:
        """
        Synthesize positive and negative prompts from taxonomy attributes.

        Args:
            category: Packaging form category (e.g. 'folding_carton', 'Carton', 'Folding Carton').
            substrate: Material substrate (e.g. 'kraft_cardboard', 'Kraft', 'coated SBS').
            finish: Print finish or embellishment (e.g. 'gold_foil', 'Gold Foil', 'spot UV').
            lighting: Studio lighting setup.
            camera_angle: Camera perspective angle.
            background: Backdrop and pedestal environment.
            user_prompt: Optional custom creative user directive.
            brand_name: Optional brand or product title to highlight.
            color_palette: Optional color scheme (e.g. 'sage green and terracotta').
            custom_negative: Additional negative terms to append.

        Returns:
            PromptPackage containing enriched positive and negative prompts. The
            metadata dict additionally reports ``normalized_inputs`` (argument ->
            canonical key) and ``unrecognized_inputs`` (arguments that fell back
            to their default because they did not match the taxonomy).
        """
        # Resolve inputs through the canonical taxonomy with graceful fallbacks
        cat_key, cat_raw = _resolve_input(category, "package_types", _DEFAULT_CATEGORY)
        sub_key, sub_raw = _resolve_input(substrate, "materials", _DEFAULT_SUBSTRATE)
        fin_key, fin_raw = _resolve_input(finish, "finishes", _DEFAULT_FINISH)
        lit_key, lit_raw = _resolve_input(lighting, "lighting", _DEFAULT_LIGHTING)
        cam_key, cam_raw = _resolve_input(camera_angle, "camera_angles", _DEFAULT_CAMERA)
        bg_key, bg_raw = _resolve_input(background, "backgrounds", _DEFAULT_BACKGROUND)

        cat_desc = CATEGORIES[cat_key]
        sub_desc = SUBSTRATES[sub_key]
        fin_desc = FINISHES[fin_key]
        lit_desc = LIGHTING_PRESETS[lit_key]
        cam_desc = CAMERA_ANGLES[cam_key]
        bg_desc = BACKGROUNDS[bg_key]

        # Construct positive prompt components
        parts: List[str] = []

        # 1. Core Subject & Category
        parts.append(f"award-winning commercial packshot of a {cat_desc.structural_cues}")

        # 2. Material Substrate & Micro-Texture
        parts.append(f"crafted from {sub_desc.shader_description}")
        parts.append(f"exhibiting {sub_desc.texture_cues}, {sub_desc.specular_reflection}")

        # 3. Print Finish & Embellishments
        if fin_desc.name != "matte":
            parts.append(f"embellished with {fin_desc.specular_profile}")
            parts.append(f"{fin_desc.relief_depth}, {fin_desc.visual_highlights}")

        # 4. Color Palette & Branding
        if color_palette:
            parts.append(f"color palette featuring sophisticated {color_palette}")
        if brand_name:
            parts.append(f"branded with minimalist typography '{brand_name}'")

        # 5. User creative directive
        if user_prompt and user_prompt.strip():
            parts.append(user_prompt.strip())

        # 6. Camera, Lens & Studio Lighting Rig
        parts.append(f"{cam_desc.framing}, shot on {cam_desc.focal_depth}")
        parts.append(f"illuminated by {lit_desc.lighting_setup}, {lit_desc.shadow_profile}")

        # 7. Environment & Grounding
        parts.append(f"placed on {bg_desc.surface_description}, with {bg_desc.contact_shadows}")

        # 8. Photographic Realism Standards
        parts.append(
            "hyper-realistic product photography, 8k resolution, ray-traced subsurface scattering, "
            "physically based rendering (PBR), flawless industrial packaging production quality"
        )

        positive_str = ", ".join(parts)

        # Construct negative prompt components
        neg_parts = [self.base_negative, cat_desc.negative_cues]
        if custom_negative and custom_negative.strip():
            neg_parts.append(custom_negative.strip())

        negative_str = ", ".join(neg_parts)

        unrecognized: List[str] = []
        for arg_name, raw in (
            ("category", cat_raw),
            ("substrate", sub_raw),
            ("finish", fin_raw),
            ("lighting", lit_raw),
            ("camera_angle", cam_raw),
            ("background", bg_raw),
        ):
            if raw is not None:
                unrecognized.append(f"{arg_name}: {raw!r}")

        return PromptPackage(
            positive_prompt=positive_str,
            negative_prompt=negative_str,
            category=cat_desc.name,
            substrate=sub_desc.name,
            finish=fin_desc.name,
            lighting=lit_desc.name,
            camera_angle=cam_desc.name,
            background=bg_desc.name,
            metadata={
                "category_display": cat_desc.display_name,
                "substrate_display": sub_desc.display_name,
                "finish_display": fin_desc.display_name,
                "lighting_display": lit_desc.display_name,
                "camera_display": cam_desc.display_name,
                "background_display": bg_desc.display_name,
                "normalized_inputs": {
                    "category": cat_key,
                    "substrate": sub_key,
                    "finish": fin_key,
                    "lighting": lit_key,
                    "camera_angle": cam_key,
                    "background": bg_key,
                },
                "unrecognized_inputs": unrecognized,
            },
        )


def build_packaging_prompt(
    category: str = "carton",
    substrate: str = "coated SBS",
    finish: str = "matte",
    lighting: str = "studio softbox",
    camera_angle: str = "three-quarter",
    background: str = "seamless studio",
    user_prompt: Optional[str] = None,
    brand_name: Optional[str] = None,
    color_palette: Optional[str] = None,
    custom_negative: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Convenience function returning (positive_prompt, negative_prompt) tuple.
    """
    engine = PromptEngine()
    pkg = engine.build_prompt(
        category=category,
        substrate=substrate,
        finish=finish,
        lighting=lighting,
        camera_angle=camera_angle,
        background=background,
        user_prompt=user_prompt,
        brand_name=brand_name,
        color_palette=color_palette,
        custom_negative=custom_negative,
    )
    return pkg.positive_prompt, pkg.negative_prompt
