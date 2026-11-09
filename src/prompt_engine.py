"""
PackVision AI - Packaging Prompt Engineering Engine
====================================================
Comprehensive packaging taxonomy and prompt generation system. Translates structural
packaging intent, substrates, industrial print finishes, lighting, camera angles,
and luxury studio environments into photorealistic SDXL conditioning prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


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
# PACKAGING TAXONOMY DEFINITION
# =============================================================================

CATEGORIES: Dict[str, CategoryDescriptor] = {
    "carton": CategoryDescriptor(
        name="carton",
        display_name="Folding Carton",
        structural_cues="folding boxboard folding carton, crisp die-cut scored folding edges, tuck-end top closure, precision folded paperboard panels, architectural packaging structure",
        negative_cues="deformed cardboard flaps, wrinkled crushed paper, bent paperboard, asymmetrical carton crease",
    ),
    "box": CategoryDescriptor(
        name="box",
        display_name="Rigid Setup Box",
        structural_cues="luxury two-piece rigid setup box with tight-fitting lid and base, crisp 90-degree beveled corners, thick solid greyboard construction, seamless wrap",
        negative_cues="caved-in lid, misaligned box corners, warped box sides, ripped paper edges",
    ),
    "pouch": CategoryDescriptor(
        name="pouch",
        display_name="Stand-Up Flexible Pouch",
        structural_cues="premium stand-up flexible pouch (doypack), bottom oval gusset standing upright, heat-sealed perimeter seams, top tear notch and press-to-close ziplock seal, natural subtle foil foil-laminate crinkles",
        negative_cues="deflated plastic bag, punctured pouch, messy melted seals, irregular ballooning",
    ),
    "bottle": CategoryDescriptor(
        name="bottle",
        display_name="Cosmetic / Beverage Bottle",
        structural_cues="cylindrical luxury bottle with tapered shoulder and neck finish, premium dispensing pump or precision dropper pipette closure, optical glass wall thickness, seamless symmetry",
        negative_cues="warped crooked bottle, melting glass, bent dropper neck, lopsided cylinder",
    ),
    "jar": CategoryDescriptor(
        name="jar",
        display_name="Cosmetic / Food Jar",
        structural_cues="wide-mouth heavy-base cosmetic jar, double-wall screw cap closure with metallic rim, thick bottom glass lens base, perfectly symmetrical cylindrical vessel",
        negative_cues="crooked jar lid, cross-threaded cap, chipped glass rim, asymmetrical base",
    ),
    "can": CategoryDescriptor(
        name="can",
        display_name="Aluminum Can",
        structural_cues="seamless extruded aluminum cylindrical can, rolled double-seam top rim, recessed chime bottom, stay-on tab opening, precision metallic cylinder",
        negative_cues="dented aluminum, crushed can body, misaligned lid seam, warped metal",
    ),
    "tube": CategoryDescriptor(
        name="tube",
        display_name="Squeeze Tube",
        structural_cues="flexible cosmetic squeeze tube with ultrasonic crimped bottom seal, flip-top dispensing cap base standing upright, smooth cylindrical taper",
        negative_cues="flattened crushed tube, burst crimp seal, melted nozzle, irregular deformities",
    ),
    "sleeve": CategoryDescriptor(
        name="sleeve",
        display_name="Packaging Sleeve Band",
        structural_cues="slide-over paperboard belly band sleeve securely wrapped around inner container tray, open perimeter ends, crisp scored folding edges",
        negative_cues="loose torn sleeve, sliding askew paper, crooked wrapper, torn cardboard seam",
    ),
}

SUBSTRATES: Dict[str, SubstrateDescriptor] = {
    "kraft": SubstrateDescriptor(
        name="kraft",
        display_name="Natural Kraft Paperboard",
        shader_description="unbleached raw organic brown kraft paperboard, tactile fibrous paper grain, micro wood pulp inclusions, warm earth tone organic paper texture",
        texture_cues="visible organic paper fibers, subtle porous matte tooth, natural speckled recycled flecks",
        specular_reflection="100% diffuse non-specular matte light absorption, zero synthetic gloss",
    ),
    "corrugated": SubstrateDescriptor(
        name="corrugated",
        display_name="Corrugated Cardboard",
        shader_description="sturdy micro-flute e-flute corrugated cardboard, visible rhythmic horizontal linerboard ridges, structural industrial kraft packaging",
        texture_cues="subtle flute wave indentations, organic fibrous linerboard grain, raw packaging paper finish",
        specular_reflection="diffuse matte cardboard light scattering, zero shine",
    ),
    "matte carton": SubstrateDescriptor(
        name="matte carton",
        display_name="Soft-Touch Matte Carton",
        shader_description="premium solid bleached sulfate (SBS) paperboard with velvety soft-touch matte lamination, ultra-fine microscopic peach-skin surface",
        texture_cues="smooth velvet-like touch, ultra-clean pure paperboard surface, non-porous luxury feel",
        specular_reflection="ultra-broad diffused specular sheen, zero glare, velvety light roll-off",
    ),
    "coated SBS": SubstrateDescriptor(
        name="coated SBS",
        display_name="Coated SBS Board (Semi-Gloss)",
        shader_description="high-caliper solid bleached sulfate (SBS) clay-coated packaging board, smooth semi-gloss mineral coating, clean commercial printing surface",
        texture_cues="micro-smooth clay coating, crisp reflection fidelity, premium white paperboard core",
        specular_reflection="balanced semi-specular highlights, clean studio light kickers, smooth satin sheen",
    ),
    "flexible pouch": SubstrateDescriptor(
        name="flexible pouch",
        display_name="Multi-Layer Barrier Laminate",
        shader_description="multi-ply high-barrier aluminum foil and polymer laminate film, brushed matte metallic sheen with subtle specular tension highlights",
        texture_cues="smooth micro-embossed film surface, micro-crinkles along gusset folds, heat-seal foil texture",
        specular_reflection="anisotropic brushed metallic specular sheen, gentle light glints along pouch curvature",
    ),
    "frosted glass": SubstrateDescriptor(
        name="frosted glass",
        display_name="Frosted Translucent Glass",
        shader_description="acid-etched frosted soda-lime cosmetic glass, milky translucent subsurface light scattering (SSS), soft internal liquid shadow diffusion",
        texture_cues="micro-fine satin etched glass surface, optical glass wall thickness depth",
        specular_reflection="soft diffuse ambient sheen with gentle highlight diffusion across bottle shoulder",
    ),
    "brushed aluminum": SubstrateDescriptor(
        name="brushed aluminum",
        display_name="Anisotropic Brushed Aluminum",
        shader_description="spun-brushed aerospace-grade aluminum alloy, directional circular radial brushing striations, high-end metallic packaging",
        texture_cues="fine microscopic metallic grooves, seamless cold-rolled metal surface",
        specular_reflection="strong directional anisotropic specular highlights, sharp metallic light band reflections",
    ),
    "molded pulp": SubstrateDescriptor(
        name="molded pulp",
        display_name="Eco-Friendly Molded Pulp",
        shader_description="biodegradable thermoformed molded cellulose fiber pulp, organic porous egg-carton style texture, soft sculpted radii bevels",
        texture_cues="pressed natural fiber mesh pattern, earthy organic speckles, tactile porous density",
        specular_reflection="completely non-reflective dry matte, deep diffuse shadow absorption",
    ),
}

FINISHES: Dict[str, FinishDescriptor] = {
    "gold foil": FinishDescriptor(
        name="gold foil",
        display_name="Hot Stamped Gold Foil",
        specular_profile="hot-stamped 24-karat mirror-finish metallic gold foil, dazzling specular reflections, razor-sharp vector die boundary",
        relief_depth="micro-pressed relief edge with subtle die-strike depth",
        visual_highlights="brilliant warm golden specular glints and studio softbox light reflection strips",
    ),
    "silver foil": FinishDescriptor(
        name="silver foil",
        display_name="Cold Stamped Chrome Silver Foil",
        specular_profile="precision cold-stamped liquid chrome silver metallic foil, high-contrast mirror reflectivity",
        relief_depth="flush metallic layer with micro-stamped crisp contour lines",
        visual_highlights="pure cool platinum specular flare highlights and crisp studio reflection bands",
    ),
    "spot UV": FinishDescriptor(
        name="spot UV",
        display_name="High-Gloss Spot UV Varnish",
        specular_profile="selective clear high-build spot UV gloss varnish layer, ultra-glossy glass-like lacquer contrasting against matte base substrate",
        relief_depth="raised tactile dimensional varnish coating (50-micron relief edge)",
        visual_highlights="mirror-like specular light sheen that catches studio key light while base remains matte",
    ),
    "embossed": FinishDescriptor(
        name="embossed",
        display_name="Multi-Level Blind Embossing",
        specular_profile="tactile multi-level raised blind embossing, crisp sculptural paper sculpt die-pressed from behind",
        relief_depth="distinct 3D raised sculptural relief with tactile beveled edges",
        visual_highlights="cast micro-shadows along raised relief contours and subtle highlight ridges",
    ),
    "debossed": FinishDescriptor(
        name="debossed",
        display_name="Sunken Letterpress Debossing",
        specular_profile="deep mechanical letterpress debossed indentation pressed into thick substrate",
        relief_depth="sunken concave beveled depression with sharp indented walls",
        visual_highlights="deep inner cast shadow along depression rim, subtle micro-contrast inside recessed lettering",
    ),
    "holographic": FinishDescriptor(
        name="holographic",
        display_name="Rainbow Holographic Foil",
        specular_profile="prismatic diffraction grating holographic foil, radiant chromatic rainbow color shift (magenta, cyan, gold)",
        relief_depth="micro-etched diffraction pattern layer",
        visual_highlights="iridescent spectral prism reflections flaring under studio spotlighting",
    ),
    "matte": FinishDescriptor(
        name="matte",
        display_name="Protective Matte Finish",
        specular_profile="uniform non-reflective soft-touch protective matte seal, anti-glare finish",
        relief_depth="smooth flush print layer integrated with substrate",
        visual_highlights="gentle even illumination, zero distracting glare, pure color fidelity",
    ),
    "uncoated": FinishDescriptor(
        name="uncoated",
        display_name="Raw Uncoated Ink Absorption",
        specular_profile="raw natural printing ink absorbed directly into porous paperboard fibers",
        relief_depth="zero thickness, embedded organic ink sink",
        visual_highlights="earthy muted tones with authentic tactile letterpress absorption",
    ),
}

LIGHTING_PRESETS: Dict[str, LightingDescriptor] = {
    "studio softbox": LightingDescriptor(
        name="studio softbox",
        display_name="Three-Point Studio Softbox",
        lighting_setup="professional 3-point softbox studio lighting, large diffuse overhead key light, gentle shadow fill, balanced highlights",
        shadow_profile="soft gradual contact shadow gradient beneath base, clean studio vignette roll-off",
    ),
    "dramatic rim": LightingDescriptor(
        name="dramatic rim",
        display_name="Dramatic Rim Lighting (Chiaroscuro)",
        lighting_setup="high-contrast theatrical rim lighting, intense specular kicker lights grazing the package silhouettes, deep moody chiaroscuro contrast",
        shadow_profile="deep rich sculpted drop shadows, pronounced atmospheric depth",
    ),
    "high-key commercial": LightingDescriptor(
        name="high-key commercial",
        display_name="High-Key Commercial Packshot",
        lighting_setup="bright high-key advertising commercial lighting, even wraparound illumination, pristine clean look, luminous studio ambient fill",
        shadow_profile="whisper-soft featherweight floor shadow, pristine white illumination",
    ),
    "natural daylight": LightingDescriptor(
        name="natural daylight",
        display_name="Warm Window Daylight",
        lighting_setup="soft organic window daylight streaming from 45-degree angle, golden hour warmth, gentle environmental bounce light",
        shadow_profile="elongated soft natural sun shadows with subtle ambient penumbra",
    ),
}

CAMERA_ANGLES: Dict[str, CameraDescriptor] = {
    "three-quarter": CameraDescriptor(
        name="three-quarter",
        display_name="Three-Quarter Hero Angle (3/4 View)",
        framing="professional three-quarter perspective hero angle displaying front panel, side panel, and top closure simultaneously",
        focal_depth="85mm portrait packshot lens, f/5.6 aperture, razor-sharp optical focus throughout packaging body, subtle creamy background blur",
    ),
    "front": CameraDescriptor(
        name="front",
        display_name="Straight-On Commercial Front View",
        framing="straight-on eye-level commercial catalog packshot, perfectly centered and balanced orthographic perspective, true-to-life proportions",
        focal_depth="90mm tilt-shift studio lens, f/8 aperture, zero perspective distortion, 100% edge-to-edge optical sharpness",
    ),
    "top-down": CameraDescriptor(
        name="top-down",
        display_name="Flat-Lay Top-Down View",
        framing="overhead flat-lay tabletop perspective, 90-degree downward angle, editorial composition",
        focal_depth="50mm macro lens, f/8 aperture, crisp planar focus across entire surface",
    ),
}

BACKGROUNDS: Dict[str, BackgroundDescriptor] = {
    "seamless studio": BackgroundDescriptor(
        name="seamless studio",
        display_name="Seamless Studio Cyclorama",
        surface_description="infinite neutral warm-gray seamless cyclorama studio backdrop, minimalist high-end packshot aesthetic",
        contact_shadows="precise realistic contact ambient occlusion shadow grounding the base securely on the ground plane",
    ),
    "warm pedestal": BackgroundDescriptor(
        name="warm pedestal",
        display_name="Architectural Travertine Pedestal",
        surface_description="monolithic polished travertine limestone pedestal plinth, warm natural stone textures, high-end gallery atmosphere",
        contact_shadows="crisp contact shadow cast across textured stone pedestal top",
    ),
    "minimalist marble": BackgroundDescriptor(
        name="minimalist marble",
        display_name="Polished Carrara Marble",
        surface_description="luxurious white Carrara marble surface with faint subtle gray veining, gentle mirror reflection beneath the packaging",
        contact_shadows="soft dual contact shadow with faint glossy specular reflection on polished marble",
    ),
    "dark luxury": BackgroundDescriptor(
        name="dark luxury",
        display_name="Dark Luxury Slate & Charcoal",
        surface_description="moody dark luxury matte charcoal slate tabletop, deep ebony studio vignette backdrop, premium boutique cosmetics vibe",
        contact_shadows="deep rich velvet ambient occlusion shadow fading seamlessly into slate",
    ),
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


# =============================================================================
# PROMPT ENGINE CLASS & BUILDER FUNCTION
# =============================================================================

class PromptEngine:
    """
    Expert Prompt Engine for PackVision AI.
    
    Translates high-level packaging selections into fine-tuned, physically grounded
    conditioning prompts for Stable Diffusion XL ControlNet pipelines.
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
            category: Packaging form category (e.g. carton, pouch, bottle).
            substrate: Material substrate (e.g. kraft, coated SBS, frosted glass).
            finish: Print finish or embellishment (e.g. gold foil, spot UV, embossed).
            lighting: Studio lighting setup.
            camera_angle: Camera perspective angle.
            background: Backdrop and pedestal environment.
            user_prompt: Optional custom creative user directive.
            brand_name: Optional brand or product title to highlight.
            color_palette: Optional color scheme (e.g. 'sage green and terracotta').
            custom_negative: Additional negative terms to append.
            
        Returns:
            PromptPackage containing enriched positive and negative prompts.
        """
        # Resolve descriptors with graceful fallbacks
        cat_desc = CATEGORIES.get(category.lower(), CATEGORIES["carton"])
        sub_desc = SUBSTRATES.get(substrate.lower(), SUBSTRATES["coated SBS"])
        fin_desc = FINISHES.get(finish.lower(), FINISHES["matte"])
        lit_desc = LIGHTING_PRESETS.get(lighting.lower(), LIGHTING_PRESETS["studio softbox"])
        cam_desc = CAMERA_ANGLES.get(camera_angle.lower(), CAMERA_ANGLES["three-quarter"])
        bg_desc = BACKGROUNDS.get(background.lower(), BACKGROUNDS["seamless studio"])

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
