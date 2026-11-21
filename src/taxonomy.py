"""
PackVision AI - Canonical Packaging Taxonomy
=============================================
Single source of truth for packaging vocabulary shared by every call site:

  - src/prompt_engine.py            (prompt fragments fed to SDXL conditioning)
  - backend/app/api/v1/taxonomy.py  (REST taxonomy response)
  - ui/app_gradio.py                (Gradio dropdown labels)
  - data/dataset_generator.py       (dataset material/finish keys)
  - backend generation services     (package_type / material / finish ids)

Each category is an ordered mapping: ``canonical_key -> entry`` where entry is a
dict with at least::

    label            human-readable display name
    prompt_fragment  rich optical description used in generation prompts
    synonyms         alternative ids / spellings / labels accepted by normalize()

plus category-specific descriptive fields (negative_cues, shader_description,
specular_profile, ...) consumed by the prompt engine and the backend response.

Public helpers:
    normalize(value, category)       -> canonical key or None
    all_keys(category)               -> list of canonical keys
    entry(category, key)             -> entry dict or None
    ensure_valid(category, value, default) -> (key, was_normalized: bool)
    taxonomy_response()              -> backend-compatible response dict
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "PACKAGE_TYPES",
    "MATERIALS",
    "FINISHES",
    "LIGHTING",
    "CAMERA_ANGLES",
    "BACKGROUNDS",
    "CATEGORY_ALIASES",
    "TAXONOMY_CATEGORIES",
    "METRIC",
    "normalize",
    "all_keys",
    "entry",
    "ensure_valid",
    "taxonomy_response",
    "resolve_category",
]

# =============================================================================
# PACKAGE TYPES (structural forms)
# =============================================================================

PACKAGE_TYPES: Dict[str, Dict[str, Any]] = {
    "folding_carton": {
        "label": "Folding Carton",
        "prompt_fragment": (
            "folding boxboard folding carton, crisp die-cut scored folding edges, "
            "tuck-end top closure, precision folded paperboard panels, "
            "architectural packaging structure"
        ),
        "negative_cues": (
            "deformed cardboard flaps, wrinkled crushed paper, bent paperboard, "
            "asymmetrical carton crease"
        ),
        "description": "Folding boxboard cartons with tuck-end or crash-lock closures for retail products.",
        "icon": "package",
        "synonyms": [
            "carton", "cartons", "folding carton", "folding cartons", "folding box",
            "folding boxboard", "paperboard carton", "tuck end carton", "tuck-end carton",
            "tuck_end_carton",
        ],
    },
    "rigid_box": {
        "label": "Rigid Box",
        "prompt_fragment": (
            "luxury two-piece rigid setup box with tight-fitting lid and base, crisp "
            "90-degree beveled corners, thick solid greyboard construction, seamless wrap"
        ),
        "negative_cues": (
            "caved-in lid, misaligned box corners, warped box sides, ripped paper edges"
        ),
        "description": "Two-piece rigid setup boxes with thick greyboard construction for luxury goods.",
        "icon": "package",
        "synonyms": [
            "box", "boxes", "rigid box", "rigid boxes", "rigid setup box", "setup box",
            "gift box", "luxury box", "two piece box", "box carton", "box / carton",
            "box_carton", "rigid setup box display",
        ],
    },
    "mailer_box": {
        "label": "Mailer Box",
        "prompt_fragment": (
            "self-locking e-commerce mailer box with fold-in side walls and interlocking "
            "front tuck flap, corrugated or paperboard walls, flat ergonomic shipping profile"
        ),
        "negative_cues": (
            "collapsed mailer walls, misaligned tuck flap, torn corrugated edges, gaping box seams"
        ),
        "description": "Self-locking e-commerce mailer boxes with secure tuck flaps for shipping.",
        "icon": "package",
        "synonyms": [
            "mailer", "mailers", "mailer box", "mailer boxes", "shipping box",
            "shipping mailer", "ecommerce mailer", "e commerce mailer", "subscription box",
        ],
    },
    "stand_up_pouch": {
        "label": "Stand-Up Pouch",
        "prompt_fragment": (
            "premium stand-up flexible pouch (doypack), bottom oval gusset standing upright, "
            "heat-sealed perimeter seams, top tear notch and press-to-close ziplock seal, "
            "natural subtle foil-laminate crinkles"
        ),
        "negative_cues": (
            "deflated plastic bag, punctured pouch, messy melted seals, irregular ballooning"
        ),
        "description": "Flexible doy-pack stand-up pouches with zip seals for food, powders, and coffee.",
        "icon": "shopping-bag",
        "synonyms": [
            "pouch", "pouches", "stand up pouch", "stand-up pouch", "stand_up_pouch",
            "doypack", "doy pack", "doy_pack", "stand up pouches", "flexible pouch",
            "zip pouch", "zipper pouch", "coffee pouch", "stand-up flexible pouch",
        ],
    },
    "flat_pouch": {
        "label": "Flat Pouch",
        "prompt_fragment": (
            "flat three-seal sachet pouch, pillow-style sealed perimeter, crimped flat seals "
            "on all four edges, gentle product settle wrinkles"
        ),
        "negative_cues": (
            "inflated balloon pouch, burst seal crimps, warped pillow shape, jagged seal edges"
        ),
        "description": "Flat pillow and sachet pouches for single-serve portions and samples.",
        "icon": "shopping-bag",
        "synonyms": [
            "flat pouch", "flat_pouch", "flat bag", "sachet", "sachets", "pillow pouch",
            "pillow pack", "stick pack", "sample sachet",
        ],
    },
    "bottle": {
        "label": "Bottle",
        "prompt_fragment": (
            "cylindrical luxury bottle with tapered shoulder and neck finish, premium "
            "dispensing pump or precision dropper pipette closure, optical glass wall "
            "thickness, seamless symmetry"
        ),
        "negative_cues": (
            "warped crooked bottle, melting glass, bent dropper neck, lopsided cylinder"
        ),
        "description": "Glass or plastic bottles for beverages, spirits, oils, and skincare serums.",
        "icon": "bottle",
        "synonyms": [
            "bottle", "bottles", "dropper bottle", "pump bottle", "beverage bottle",
            "cosmetic bottle", "serum bottle", "cosmetic / beverage bottle",
            "cosmetic_beverage_bottle",
        ],
    },
    "jar": {
        "label": "Jar",
        "prompt_fragment": (
            "wide-mouth heavy-base cosmetic jar, double-wall screw cap closure with metallic "
            "rim, thick bottom glass lens base, perfectly symmetrical cylindrical vessel"
        ),
        "negative_cues": (
            "crooked jar lid, cross-threaded cap, chipped glass rim, asymmetrical base"
        ),
        "description": "Cosmetic cream jars and culinary glass preserve containers.",
        "icon": "circle",
        "synonyms": [
            "jar", "jars", "cream jar", "cosmetic jar", "food jar", "preserve jar",
            "candle jar", "cosmetic / food jar", "cosmetic_food_jar",
        ],
    },
    "can": {
        "label": "Can",
        "prompt_fragment": (
            "seamless extruded aluminum cylindrical can, rolled double-seam top rim, "
            "recessed chime bottom, stay-on tab opening, precision metallic cylinder"
        ),
        "negative_cues": (
            "dented aluminum, crushed can body, misaligned lid seam, warped metal"
        ),
        "description": "Aluminum beverage cans, aerosol cans, and tin canisters.",
        "icon": "disc",
        "synonyms": [
            "can", "cans", "tin can", "beverage can", "aluminum can", "aerosol can",
            "soda can",
        ],
    },
    "tube": {
        "label": "Squeeze Tube",
        "prompt_fragment": (
            "flexible cosmetic squeeze tube with ultrasonic crimped bottom seal, flip-top "
            "dispensing cap base standing upright, smooth cylindrical taper"
        ),
        "negative_cues": (
            "flattened crushed tube, burst crimp seal, melted nozzle, irregular deformities"
        ),
        "description": "Flexible squeeze tubes for lotions, creams, toothpastes, and cosmetics.",
        "icon": "cylinder",
        "synonyms": [
            "tube", "tubes", "squeeze tube", "cosmetic tube", "toothpaste tube",
            "laminate tube",
        ],
    },
    "sleeve": {
        "label": "Packaging Sleeve",
        "prompt_fragment": (
            "slide-over paperboard belly band sleeve securely wrapped around inner container "
            "tray, open perimeter ends, crisp scored folding edges"
        ),
        "negative_cues": (
            "loose torn sleeve, sliding askew paper, crooked wrapper, torn cardboard seam"
        ),
        "description": "Slide-over paperboard belly bands wrapped around trays, cups, or inner containers.",
        "icon": "package",
        "synonyms": [
            "sleeve", "sleeves", "belly band", "sleeve band", "paper sleeve", "wrap sleeve",
            "packaging sleeve band",
        ],
    },
}


# =============================================================================
# MATERIALS (substrates)
# =============================================================================

MATERIALS: Dict[str, Dict[str, Any]] = {
    "kraft_cardboard": {
        "label": "Kraft Cardboard",
        "shader_description": (
            "unbleached raw organic brown kraft paperboard, tactile fibrous paper grain, "
            "micro wood pulp inclusions, warm earth tone organic paper texture"
        ),
        "texture_cues": (
            "visible organic paper fibers, subtle porous matte tooth, natural speckled "
            "recycled flecks"
        ),
        "specular_reflection": "100% diffuse non-specular matte light absorption, zero synthetic gloss",
        "default_finish": "matte",
        "compatible_finishes": ["matte", "uncoated", "embossed", "debossed"],
        "description": "Natural unbleached recycled fibrous kraft paperboard for organic aesthetics.",
        "synonyms": [
            "kraft", "kraft cardboard", "kraft paper", "kraft paperboard", "kraft_paper",
            "kraft board", "kraft_board", "natural kraft", "natural kraft paperboard",
            "brown kraft", "recycled kraft", "natural kraft paper",
        ],
    },
    "corrugated_cardboard": {
        "label": "Corrugated Cardboard",
        "shader_description": (
            "sturdy micro-flute e-flute corrugated cardboard, visible rhythmic horizontal "
            "linerboard ridges, structural industrial kraft packaging"
        ),
        "texture_cues": (
            "subtle flute wave indentations, organic fibrous linerboard grain, raw packaging "
            "paper finish"
        ),
        "specular_reflection": "diffuse matte cardboard light scattering, zero shine",
        "default_finish": "uncoated",
        "compatible_finishes": ["uncoated", "matte", "embossed"],
        "description": "Fluted corrugated shipping board with rhythmic linerboard ridges.",
        "synonyms": [
            "corrugated", "corrugated cardboard", "corrugated board", "corrugated_box",
            "corrugated box", "corrugated boxboard", "e flute", "e-flute",
        ],
    },
    "coated_paperboard": {
        "label": "Coated Paperboard",
        "shader_description": (
            "high-caliper solid bleached sulfate (SBS) clay-coated packaging board, smooth "
            "semi-gloss mineral coating, clean commercial printing surface"
        ),
        "texture_cues": (
            "micro-smooth clay coating, crisp reflection fidelity, premium white paperboard core"
        ),
        "specular_reflection": (
            "balanced semi-specular highlights, clean studio light kickers, smooth satin sheen"
        ),
        "default_finish": "gloss",
        "compatible_finishes": ["gloss", "matte", "satin", "spot_uv", "gold_foil", "silver_foil", "embossed", "debossed", "soft_touch"],
        "description": "Sustainable clay-coated folding boxboard with sharp print reproduction.",
        "synonyms": [
            "coated sbs", "coated_sbs", "coated paperboard", "coated paper board", "sbs",
            "sbs board", "coated board", "cardboard", "paperboard", "paper board",
            "boxboard", "folding boxboard", "cardboard / paperboard", "cardboard_paperboard",
            "coated sbs board semi-gloss", "coated_sbs_board_semi_gloss",
        ],
    },
    "matte_laminated": {
        "label": "Matte Laminated Board",
        "shader_description": (
            "premium solid bleached sulfate (SBS) paperboard with velvety soft-touch matte "
            "lamination, ultra-fine microscopic peach-skin surface"
        ),
        "texture_cues": (
            "smooth velvet-like touch, ultra-clean pure paperboard surface, non-porous luxury feel"
        ),
        "specular_reflection": "ultra-broad diffused specular sheen, zero glare, velvety light roll-off",
        "default_finish": "soft_touch",
        "compatible_finishes": ["soft_touch", "matte", "satin", "spot_uv"],
        "description": "Paperboard wrapped in velvety soft-touch matte lamination.",
        "synonyms": [
            "matte carton", "matte_carton", "matte laminated", "matte lamination",
            "soft touch carton", "soft touch matte carton", "soft_touch_matte_carton",
            "laminated matte board",
        ],
    },
    "uncoated_paper": {
        "label": "Uncoated Paper",
        "shader_description": (
            "raw uncoated offset paperboard with open fibrous surface, natural cream-white "
            "tone, honest recycled character"
        ),
        "texture_cues": (
            "visible paper tooth, tactile print-absorbing fibers, subtle shade variation"
        ),
        "specular_reflection": "completely diffuse flat light absorption, zero specular hotspots",
        "default_finish": "uncoated",
        "compatible_finishes": ["uncoated", "matte", "embossed", "debossed"],
        "description": "Raw uncoated paper with open fibers that absorb ink naturally.",
        "synonyms": [
            "uncoated paper", "uncoated paperboard", "uncoated board", "offset paper",
            "recycled paper", "plain paper",
        ],
    },
    "flexible_mylar": {
        "label": "Flexible Mylar",
        "shader_description": (
            "multi-ply high-barrier aluminum foil and polymer laminate film, brushed matte "
            "metallic sheen with subtle specular tension highlights"
        ),
        "texture_cues": (
            "smooth micro-embossed film surface, micro-crinkles along gusset folds, heat-seal "
            "foil texture"
        ),
        "specular_reflection": (
            "anisotropic brushed metallic specular sheen, gentle light glints along pouch curvature"
        ),
        "default_finish": "matte",
        "compatible_finishes": ["matte", "gloss", "satin", "holographic_film", "silver_foil"],
        "description": "Multi-layer barrier PET/mylar laminate film for flexible pouches.",
        "synonyms": [
            "flexible mylar", "mylar", "mylar film", "flexible pouch", "flexible_pouch",
            "foil laminate", "barrier laminate", "laminated film", "plastic", "polymer",
            "pet plastic", "pet polymer", "hdpe", "polymer / plastic", "polymer_plastic",
            "plastic film", "multi-layer barrier laminate", "multi_layer_barrier_laminate",
        ],
    },
    "frosted_glass": {
        "label": "Frosted Glass",
        "shader_description": (
            "acid-etched frosted soda-lime cosmetic glass, milky translucent subsurface light "
            "scattering (SSS), soft internal liquid shadow diffusion"
        ),
        "texture_cues": (
            "micro-fine satin etched glass surface, optical glass wall thickness depth"
        ),
        "specular_reflection": (
            "soft diffuse ambient sheen with gentle highlight diffusion across bottle shoulder"
        ),
        "default_finish": "satin",
        "compatible_finishes": ["satin", "gloss", "matte", "embossed"],
        "description": "Acid-etched translucent glass with diffused subsurface light scattering.",
        "synonyms": [
            "frosted glass", "frosted_glass", "frosted", "acid etched glass", "etched glass",
            "matte glass", "translucent glass", "satin glass", "frosted translucent glass",
            "frosted_translucent_glass",
        ],
    },
    "clear_glass": {
        "label": "Clear Glass",
        "shader_description": (
            "optically clear crystal soda-lime glass, high transparency with subtle green-cyan "
            "edge tint, refraction and caustic light transmission through the vessel"
        ),
        "texture_cues": (
            "flawless polished glass surface, crisp meniscus fill line, clean wall thickness depth"
        ),
        "specular_reflection": (
            "sharp specular window reflections with bright caustic highlights and refractive "
            "background distortion"
        ),
        "default_finish": "gloss",
        "compatible_finishes": ["gloss", "satin", "matte", "embossed"],
        "description": "Premium optically clear glass with realistic refractions and caustics.",
        "synonyms": [
            "glass", "clear glass", "clear_glass", "transparent glass", "crystal glass",
            "soda lime glass", "optical glass",
        ],
    },
    "amber_glass": {
        "label": "Amber Glass",
        "shader_description": (
            "deep amber-tinted apothecary glass, warm honey-brown translucent body glowing "
            "under backlight, vintage pharmacy bottle character"
        ),
        "texture_cues": (
            "smooth glass wall with warm internal color gradient, precise molded symmetry"
        ),
        "specular_reflection": (
            "warm golden internal glow, soft specular edge highlights, rich translucent depth"
        ),
        "default_finish": "gloss",
        "compatible_finishes": ["gloss", "satin", "matte"],
        "description": "Warm amber-tinted apothecary glass for spirits, beer, and skincare.",
        "synonyms": [
            "amber glass", "amber_glass", "amber", "brown glass", "honey glass",
            "amber bottle glass", "apothecary glass",
        ],
    },
    "brushed_aluminum": {
        "label": "Brushed Aluminum",
        "shader_description": (
            "spun-brushed aerospace-grade aluminum alloy, directional circular radial brushing "
            "striations, high-end metallic packaging"
        ),
        "texture_cues": "fine microscopic metallic grooves, seamless cold-rolled metal surface",
        "specular_reflection": (
            "strong directional anisotropic specular highlights, sharp metallic light band "
            "reflections"
        ),
        "default_finish": "satin",
        "compatible_finishes": ["satin", "gloss", "matte", "embossed", "debossed"],
        "description": "Brushed or polished aluminum with authentic metallic sheen.",
        "synonyms": [
            "brushed aluminum", "aluminum", "aluminium", "brushed metal", "brushed steel",
            "metal", "anodized aluminum", "silver metal", "steel",
            "aluminum / metal", "aluminum_metal", "anisotropic brushed aluminum",
            "anisotropic_brushed_aluminum",
        ],
    },
    "tinplate_metal": {
        "label": "Tinplate Metal",
        "shader_description": (
            "lacquered tinplate steel sheet body, bright electrolytic tin coating under "
            "printed litho finish, classic can-making construction"
        ),
        "texture_cues": "hard smooth enamel surface over metal, precise rolled seam tooling",
        "specular_reflection": (
            "bright mirror-like metallic reflections with tight specular highlights and clean "
            "environment bounce"
        ),
        "default_finish": "gloss",
        "compatible_finishes": ["gloss", "satin", "matte", "embossed", "debossed"],
        "description": "Lacquered tinplate steel for tins, canisters, and closures.",
        "synonyms": [
            "tinplate", "tin plate", "tinplate metal", "tinplate_metal", "tin", "tin metal",
            "tin canister", "metal tin", "litho tin",
        ],
    },
    "molded_pulp": {
        "label": "Molded Pulp",
        "shader_description": (
            "biodegradable thermoformed molded cellulose fiber pulp, organic porous "
            "egg-carton style texture, soft sculpted radii bevels"
        ),
        "texture_cues": (
            "pressed natural fiber mesh pattern, earthy organic speckles, tactile porous density"
        ),
        "specular_reflection": "completely non-reflective dry matte, deep diffuse shadow absorption",
        "default_finish": "uncoated",
        "compatible_finishes": ["uncoated", "matte", "embossed"],
        "description": "Biodegradable thermoformed cellulose pulp with organic porous texture.",
        "synonyms": [
            "molded pulp", "molded_pulp", "moulded pulp", "pulp", "molded fiber",
            "moulded fiber", "egg carton pulp", "cellulose pulp", "bagasse",
            "eco friendly molded pulp", "eco_friendly_molded_pulp",
        ],
    },
}


# =============================================================================
# FINISHES (industrial embellishments)
# =============================================================================

FINISHES: Dict[str, Dict[str, Any]] = {
    "matte": {
        "label": "Matte",
        "specular_profile": (
            "uniform non-reflective soft-touch protective matte seal, anti-glare finish"
        ),
        "relief_depth": "smooth flush print layer integrated with substrate",
        "visual_highlights": "gentle even illumination, zero distracting glare, pure color fidelity",
        "description": "Non-glare protective matte finish with even ambient diffusion.",
        "synonyms": [
            "matte", "matt", "matte finish", "flat finish", "matte soft touch",
            "matte_soft_touch", "matte soft-touch",
        ],
    },
    "gloss": {
        "label": "High Gloss",
        "specular_profile": (
            "wet-look high-gloss varnish layer, mirror-like specular reflections, deep rich "
            "color saturation"
        ),
        "relief_depth": "flush flood-coated gloss film over the print layer",
        "visual_highlights": "brilliant white specular highlight strips reflecting studio softboxes",
        "description": "Mirror-like high-gloss varnish with rich specular reflections and color depth.",
        "synonyms": [
            "gloss", "glossy", "gloss finish", "glossy finish", "high gloss",
            "high gloss uv varnish", "high_gloss_uv_varnish", "uv varnish", "flood gloss",
            "wet look", "metallic", "metallic sheen", "metallic_sheen",
        ],
    },
    "satin": {
        "label": "Satin",
        "specular_profile": (
            "low-sheen satin etched finish, gentle translucent light scattering between matte "
            "and gloss"
        ),
        "relief_depth": "smooth micro-etched surface layer",
        "visual_highlights": "soft silky highlight roll-off without glare hotspots",
        "description": "Low-sheen satin surface between matte and gloss with soft highlight roll-off.",
        "synonyms": [
            "satin", "satin finish", "satin sheen", "semi gloss", "semi-gloss", "eggshell",
            "frosted", "frosted translucent", "frosted_translucent", "low sheen", "silk finish",
        ],
    },
    "spot_uv": {
        "label": "Spot UV",
        "specular_profile": (
            "selective clear high-build spot UV gloss varnish layer, ultra-glossy glass-like "
            "lacquer contrasting against matte base substrate"
        ),
        "relief_depth": "raised tactile dimensional varnish coating (50-micron relief edge)",
        "visual_highlights": (
            "mirror-like specular light sheen that catches studio key light while base remains "
            "matte"
        ),
        "description": "Selective high-build spot UV gloss lacquer contrasting against a matte base.",
        "synonyms": [
            "spot uv", "spot_uv", "spot uv varnish", "spot gloss", "uv spot",
            "high-gloss spot uv varnish", "high_gloss_spot_uv_varnish", "selective gloss",
            "spot gloss varnish",
        ],
    },
    "gold_foil": {
        "label": "Gold Foil",
        "specular_profile": (
            "hot-stamped 24-karat mirror-finish metallic gold foil, dazzling specular "
            "reflections, razor-sharp vector die boundary"
        ),
        "relief_depth": "micro-pressed relief edge with subtle die-strike depth",
        "visual_highlights": (
            "brilliant warm golden specular glints and studio softbox light reflection strips"
        ),
        "description": "Reflective hot-stamped gold foil accents for luxury branding.",
        "synonyms": [
            "gold foil", "gold_foil", "foil", "foil stamping", "foil_stamping", "hot foil",
            "hot stamping", "hot stamped foil", "gold foil stamping", "hot-stamped gold foil",
            "gold / silver foil stamping", "gold_silver_foil_stamping", "metallic gold foil",
        ],
    },
    "silver_foil": {
        "label": "Silver Foil",
        "specular_profile": (
            "precision cold-stamped liquid chrome silver metallic foil, high-contrast mirror "
            "reflectivity"
        ),
        "relief_depth": "flush metallic layer with micro-stamped crisp contour lines",
        "visual_highlights": "pure cool platinum specular flare highlights and crisp studio reflection bands",
        "description": "Cold-stamped chrome silver metallic foil with mirror reflectivity.",
        "synonyms": [
            "silver foil", "silver_foil", "chrome foil", "chrome silver foil", "silver foil stamping",
            "cold stamped chrome silver foil", "metallic silver",
        ],
    },
    "holographic_film": {
        "label": "Holographic Film",
        "specular_profile": (
            "prismatic diffraction grating holographic foil, radiant chromatic rainbow color "
            "shift (magenta, cyan, gold)"
        ),
        "relief_depth": "micro-etched diffraction pattern layer",
        "visual_highlights": "iridescent spectral prism reflections flaring under studio spotlighting",
        "description": "Prismatic holographic film with iridescent rainbow color shift.",
        "synonyms": [
            "holographic", "holographic_film", "holo", "holographic foil", "holographic film",
            "rainbow foil", "iridescent", "iridescent foil", "prismatic foil",
            "diffraction foil",
        ],
    },
    "embossed": {
        "label": "Embossed",
        "specular_profile": (
            "tactile multi-level raised blind embossing, crisp sculptural paper sculpt "
            "die-pressed from behind"
        ),
        "relief_depth": "distinct 3D raised sculptural relief with tactile beveled edges",
        "visual_highlights": "cast micro-shadows along raised relief contours and subtle highlight ridges",
        "description": "Raised 3D tactile contours creating authentic surface shadows.",
        "synonyms": [
            "embossed", "emboss", "embossing", "blind emboss", "blind embossing",
            "raised emboss", "embossed relief", "raised relief",
            "embossed / debossed relief", "embossed_debossed_relief",
            "multi-level blind embossing", "multi_level_blind_embossing",
        ],
    },
    "debossed": {
        "label": "Debossed",
        "specular_profile": (
            "deep mechanical letterpress debossed indentation pressed into thick substrate"
        ),
        "relief_depth": "sunken concave beveled depression with sharp indented walls",
        "visual_highlights": (
            "deep inner cast shadow along depression rim, subtle micro-contrast inside recessed "
            "lettering"
        ),
        "description": "Sunken letterpress indentations pressed into the substrate with crisp walls.",
        "synonyms": [
            "debossed", "deboss", "debossing", "letterpress", "letterpress debossing",
            "sunken deboss", "sunken letterpress debossing",
        ],
    },
    "soft_touch": {
        "label": "Soft Touch",
        "specular_profile": (
            "velvety soft-touch laminate coating, peach-skin tactile surface with ultra-matte "
            "appearance"
        ),
        "relief_depth": "micro-thin soft-touch film laminate",
        "visual_highlights": "even diffuse light roll-off with zero glare and inviting tactile warmth",
        "description": "Velvety soft-touch laminate with a peach-skin tactile surface.",
        "synonyms": [
            "soft touch", "soft_touch", "soft-touch", "soft touch laminate", "velvet",
            "velvety", "velvet laminate", "velvet touch", "peach skin", "tactile laminate",
        ],
    },
    "uncoated": {
        "label": "Uncoated",
        "specular_profile": "raw natural printing ink absorbed directly into porous paperboard fibers",
        "relief_depth": "zero thickness, embedded organic ink sink",
        "visual_highlights": "earthy muted tones with authentic tactile letterpress absorption",
        "description": "Raw uncoated surface letting ink absorb into visible paper fibers.",
        "synonyms": [
            "uncoated", "uncoated finish", "raw uncoated", "raw paper finish",
            "raw uncoated ink absorption", "raw_uncoated_ink_absorption", "ink absorption",
        ],
    },
}


# =============================================================================
# LIGHTING (studio rigs)
# =============================================================================

LIGHTING: Dict[str, Dict[str, Any]] = {
    "studio_softbox": {
        "label": "Studio Softbox",
        "lighting_setup": (
            "professional 3-point softbox studio lighting, large diffuse overhead key light, "
            "gentle shadow fill, balanced highlights"
        ),
        "shadow_profile": "soft gradual contact shadow gradient beneath base, clean studio vignette roll-off",
        "description": "Balanced commercial product lighting with subtle catchlights.",
        "synonyms": [
            "studio softbox", "studio_softbox", "softbox", "softbox lighting",
            "three point studio softbox", "three_point_studio_softbox",
            "dual studio softbox", "dual_studio_softbox", "three point lighting",
            "studio lighting", "three-point studio softbox",
        ],
    },
    "dramatic_rim": {
        "label": "Dramatic Rim",
        "lighting_setup": (
            "high-contrast theatrical rim lighting, intense specular kicker lights grazing the "
            "package silhouettes, deep moody chiaroscuro contrast"
        ),
        "shadow_profile": "deep rich sculpted drop shadows, pronounced atmospheric depth",
        "description": "Bold edge highlights separating package silhouette from backdrop.",
        "synonyms": [
            "dramatic rim", "dramatic_rim", "rim light", "rim lighting",
            "dramatic rim lighting", "dramatic rim lighting chiaroscuro",
            "dramatic_rim_lighting_chiaroscuro", "dramatic luxury rim light",
            "dramatic_luxury_rim_light", "chiaroscuro", "dramatic lighting", "luxury rim light",
        ],
    },
    "high_key_commercial": {
        "label": "High-Key Commercial",
        "lighting_setup": (
            "bright high-key advertising commercial lighting, even wraparound illumination, "
            "pristine clean look, luminous studio ambient fill"
        ),
        "shadow_profile": "whisper-soft featherweight floor shadow, pristine white illumination",
        "description": "Bright even wraparound advertising illumination with pristine clean look.",
        "synonyms": [
            "high key commercial", "high_key_commercial", "high key", "high-key commercial",
            "high key lighting", "commercial high key", "commercial high key clean",
            "commercial_high_key_clean", "high key commercial packshot",
            "high_key_commercial_packshot", "bright commercial lighting", "packshot lighting",
        ],
    },
    "natural_daylight": {
        "label": "Natural Daylight",
        "lighting_setup": (
            "soft organic window daylight streaming from 45-degree angle, golden hour warmth, "
            "gentle environmental bounce light"
        ),
        "shadow_profile": "elongated soft natural sun shadows with subtle ambient penumbra",
        "description": "Warm directional daylight with soft natural shadows and golden-hour tones.",
        "synonyms": [
            "natural daylight", "natural_daylight", "daylight", "window light",
            "natural window daylight", "natural_window_daylight", "natural golden",
            "natural_golden", "golden hour", "golden hour warmth", "golden_hour_warmth",
            "warm window daylight", "warm_window_daylight", "natural light", "sunlight",
        ],
    },
    "dark_luxury": {
        "label": "Dark Luxury",
        "lighting_setup": (
            "moody low-key luxury lighting, single focused spotlight with deep falloff, subtle "
            "warm accent rim"
        ),
        "shadow_profile": "deep velvet shadows pooling around the base, dramatic intimate vignette",
        "description": "Moody low-key lighting with deep sculpted shadows for premium boutique scenes.",
        "synonyms": [
            "dark luxury", "dark_luxury", "dark luxury lighting", "low key", "low key lighting",
            "moody lighting", "moody luxury", "luxury spotlight", "dark moody",
        ],
    },
}


# =============================================================================
# CAMERA ANGLES (viewpoints)
# =============================================================================

CAMERA_ANGLES: Dict[str, Dict[str, Any]] = {
    "front_view": {
        "label": "Front View",
        "framing": (
            "straight-on eye-level commercial catalog packshot, perfectly centered and "
            "balanced orthographic perspective, true-to-life proportions"
        ),
        "focal_depth": (
            "90mm tilt-shift studio lens, f/8 aperture, zero perspective distortion, 100% "
            "edge-to-edge optical sharpness"
        ),
        "description": "Direct eye-level frontal perspective emphasizing main label and form.",
        "synonyms": [
            "front", "front view", "front_view", "front facing", "front-facing hero",
            "front_facing_hero", "straight on", "straight-on", "eye level", "front eye level",
            "front_eye_level", "straight on commercial front view",
            "straight_on_commercial_front_view", "frontal", "catalog front",
        ],
    },
    "three_quarter_view": {
        "label": "Three-Quarter View",
        "framing": (
            "professional three-quarter perspective hero angle displaying front panel, side "
            "panel, and top closure simultaneously"
        ),
        "focal_depth": (
            "85mm portrait packshot lens, f/5.6 aperture, razor-sharp optical focus throughout "
            "packaging body, subtle creamy background blur"
        ),
        "description": "Hero perspective revealing front label, side depth, and volume.",
        "synonyms": [
            "three quarter", "three_quarter", "three-quarter", "three quarter view",
            "three_quarter_view", "three-quarter view", "3/4", "3/4 view", "3_4", "3_4_view",
            "3 4 view", "three quarters", "three quarter isometric", "three_quarter_isometric",
            "three quarter angle", "three quarter hero angle", "isometric", "isometric 3d",
            "isometric_3d", "hero angle", "three_quarter_hero_angle_3_4_view",
        ],
    },
    "top_down": {
        "label": "Top-Down",
        "framing": (
            "overhead flat-lay tabletop perspective, 90-degree downward angle, editorial "
            "composition"
        ),
        "focal_depth": "50mm macro lens, f/8 aperture, crisp planar focus across entire surface",
        "description": "90-degree overhead perspective ideal for product arrangements.",
        "synonyms": [
            "top down", "top_down", "top-down", "top down flat lay", "top_down_flat_lay",
            "flat lay", "flat lay top down", "flat_lay_top_down_view", "overhead",
            "overhead view", "birds eye view", "bird s eye view",
        ],
    },
    "hero_low_angle": {
        "label": "Hero Low Angle",
        "framing": (
            "dynamic low-angle hero perspective looking slightly upward at the package, "
            "monumental product presence, subtle converging verticals"
        ),
        "focal_depth": (
            "35mm wide-angle lens, f/4 aperture, crisp focus on the hero face with cinematic "
            "depth falloff"
        ),
        "description": "Dynamic low-angle hero shot lending monumental presence to the package.",
        "synonyms": [
            "hero low angle", "hero_low_angle", "low angle", "low angle dynamic",
            "low_angle_dynamic", "dynamic low angle", "low angle hero", "hero shot",
        ],
    },
}


# =============================================================================
# BACKGROUNDS (scenes)
# =============================================================================

BACKGROUNDS: Dict[str, Dict[str, Any]] = {
    "seamless_studio": {
        "label": "Seamless Studio",
        "surface_description": (
            "infinite neutral warm-gray seamless cyclorama studio backdrop, minimalist "
            "high-end packshot aesthetic"
        ),
        "contact_shadows": (
            "precise realistic contact ambient occlusion shadow grounding the base securely on "
            "the ground plane"
        ),
        "description": "Neutral studio cyclorama with soft ground contact shadow.",
        "synonyms": [
            "seamless studio", "seamless_studio", "studio clean", "studio_clean",
            "seamless cyclorama", "seamless white neutral studio",
            "seamless_white_neutral_studio", "seamless background", "cyclorama",
            "infinity cove", "clean studio", "seamless white", "neutral studio",
            "clean studio infinity", "clean_studio_infinity", "minimalist gradient",
            "minimalist_gradient", "seamless studio cyclorama",
        ],
    },
    "warm_pedestal": {
        "label": "Warm Pedestal",
        "surface_description": (
            "monolithic polished stone pedestal plinth in warm travertine or white Carrara "
            "marble with faint gray veining, warm natural textures, high-end gallery atmosphere"
        ),
        "contact_shadows": (
            "crisp contact shadow cast across the textured stone pedestal top with faint "
            "glossy reflection"
        ),
        "description": "Architectural stone pedestal plinth with warm gallery atmosphere.",
        "synonyms": [
            "warm pedestal", "warm_pedestal", "pedestal", "stone pedestal", "travertine pedestal",
            "architectural travertine pedestal", "architectural_travertine_pedestal",
            "warm beige pedestal", "warm_beige_pedestal", "marble", "minimalist marble",
            "minimalist_marble", "carrara marble", "polished carrara marble",
            "polished_carrara_marble", "marble pedestal", "gallery pedestal", "pedestal plinth",
        ],
    },
    "dark_luxury_scene": {
        "label": "Dark Luxury Scene",
        "surface_description": (
            "moody dark luxury matte charcoal slate tabletop, deep ebony studio vignette "
            "backdrop, premium boutique cosmetics vibe"
        ),
        "contact_shadows": "deep rich velvet ambient occlusion shadow fading seamlessly into slate",
        "description": "Moody charcoal slate tabletop and deep vignette backdrop for premium scenes.",
        "synonyms": [
            "dark luxury scene", "dark_luxury_scene", "dark luxury", "dark_luxury",
            "dark luxury stone", "minimalist dark luxury stone",
            "minimalist_dark_luxury_stone", "charcoal slate", "slate table", "luxury noir",
            "dark luxury slate charcoal", "dark_luxury_slate_charcoal", "black slate",
        ],
    },
    "natural_environment": {
        "label": "Natural Environment",
        "surface_description": (
            "organic lifestyle environment with softly blurred natural greenery, warm wooden "
            "table surface, gentle environmental context storytelling"
        ),
        "contact_shadows": "soft dappled natural contact shadows blending into the environment",
        "description": "Organic outdoor or lifestyle environment with soft sunlight.",
        "synonyms": [
            "natural environment", "natural_environment", "outdoor natural", "outdoor_natural",
            "outdoor natural ambient", "outdoor_natural_ambient", "retail shelf", "retail_shelf",
            "retail store display", "outdoor", "nature", "lifestyle scene", "wooden table",
            "natural setting", "environment",
        ],
    },
    "transparent_isolated": {
        "label": "Transparent / Isolated",
        "surface_description": (
            "pure transparent isolated background with no visible backdrop, alpha-channel "
            "product cutout rendering"
        ),
        "contact_shadows": "no ground shadow, floating product isolation ready for compositing",
        "description": "Transparent alpha-isolated background with no ground plane, ready for compositing.",
        "synonyms": [
            "transparent isolated", "transparent_isolated", "transparent", "isolated",
            "alpha", "alpha channel", "no background", "cutout", "png cutout",
            "transparent background", "isolated product",
        ],
    },
}


# =============================================================================
# COMPOSITE PROMPT FRAGMENTS
# =============================================================================
# Every entry exposes a uniform "prompt_fragment" rich optical description,
# composed from its category-specific descriptive fields.

for _ent in MATERIALS.values():
    _ent["prompt_fragment"] = ", ".join(
        [_ent["shader_description"], _ent["texture_cues"], _ent["specular_reflection"]]
    )
for _ent in FINISHES.values():
    _ent["prompt_fragment"] = ", ".join(
        [_ent["specular_profile"], _ent["relief_depth"], _ent["visual_highlights"]]
    )
for _ent in LIGHTING.values():
    _ent["prompt_fragment"] = ", ".join([_ent["lighting_setup"], _ent["shadow_profile"]])
for _ent in CAMERA_ANGLES.values():
    _ent["prompt_fragment"] = ", ".join([_ent["framing"], _ent["focal_depth"]])
for _ent in BACKGROUNDS.values():
    _ent["prompt_fragment"] = ", ".join([_ent["surface_description"], _ent["contact_shadows"]])
del _ent


# =============================================================================
# CATEGORY REGISTRY & ALIASES
# =============================================================================

#: canonical category name -> ordered entry table
TAXONOMY_CATEGORIES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "package_types": PACKAGE_TYPES,
    "materials": MATERIALS,
    "finishes": FINISHES,
    "lighting": LIGHTING,
    "camera_angles": CAMERA_ANGLES,
    "backgrounds": BACKGROUNDS,
}

#: accepted aliases for category arguments (already canonical-tokenized)
CATEGORY_ALIASES: Dict[str, str] = {
    "package_types": "package_types",
    "package_type": "package_types",
    "category": "package_types",
    "categories": "package_types",
    "materials": "materials",
    "material": "materials",
    "substrate": "materials",
    "substrates": "materials",
    "finishes": "finishes",
    "finish": "finishes",
    "lighting": "lighting",
    "lighting_presets": "lighting",
    "lighting_preset": "lighting",
    "camera_angles": "camera_angles",
    "camera_angle": "camera_angles",
    "camera": "camera_angles",
    "viewpoints": "camera_angles",
    "viewpoint": "camera_angles",
    "backgrounds": "backgrounds",
    "background": "backgrounds",
    "backdrops": "backgrounds",
    "backdrop": "backgrounds",
    "scene": "backgrounds",
    "scenes": "backgrounds",
}


_SEPARATOR_RE = re.compile(r"[^a-z0-9]+")


def _canon_token(value: str) -> str:
    """Lowercase, strip, and collapse any non-alphanumeric run into a single underscore."""
    return _SEPARATOR_RE.sub("_", value.strip().lower()).strip("_")


def _build_resolver(table: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Build normalized-token -> canonical-key lookup for one category."""
    resolver: Dict[str, str] = {}
    for key, ent in table.items():
        candidates = [key, ent.get("label", ""), *ent.get("synonyms", [])]
        for cand in candidates:
            if not isinstance(cand, str) or not cand:
                continue
            token = _canon_token(cand)
            if token and token not in resolver:
                resolver[token] = key
    return resolver


_RESOLVERS: Dict[str, Dict[str, str]] = {
    canon: _build_resolver(table)
    for canon, table in TAXONOMY_CATEGORIES.items()
}


def resolve_category(category: str) -> Optional[str]:
    """Map a category argument (any alias/spelling) to its canonical category name."""
    if not isinstance(category, str):
        return None
    return CATEGORY_ALIASES.get(_canon_token(category))


def normalize(value: str, category: str) -> Optional[str]:
    """
    Resolve a raw vocabulary value to its canonical key within a category.

    Matching is case-insensitive and separator-agnostic (``_``, ``-``, ``/`` and
    whitespace all collapse to ``_``). Keys, display labels, and synonyms are all
    accepted.

    Returns:
        The canonical key, or None when the value (or category) is unrecognized.
    """
    if not isinstance(value, str):
        return None
    canon = resolve_category(category)
    if canon is None:
        return None
    token = _canon_token(value)
    if not token:
        return None
    return _RESOLVERS[canon].get(token)


def all_keys(category: str) -> List[str]:
    """Return the ordered list of canonical keys for a category ([] if unknown)."""
    canon = resolve_category(category)
    if canon is None:
        return []
    return list(TAXONOMY_CATEGORIES[canon].keys())


def entry(category: str, key: str) -> Optional[Dict[str, Any]]:
    """Return the full entry dict for a key (normalized defensively), or None."""
    canon = resolve_category(category)
    if canon is None:
        return None
    resolved = normalize(key, canon)
    if resolved is None:
        return None
    return TAXONOMY_CATEGORIES[canon][resolved]


def ensure_valid(category: str, value: str, default: str) -> Tuple[str, bool]:
    """
    Normalize ``value`` within ``category``; fall back to ``default`` when unknown.

    Returns:
        (key, was_normalized): ``was_normalized`` is True when the input resolved
        to a canonical key, False when the default was substituted instead.
    """
    resolved = normalize(value, category) if value is not None else None
    if resolved is not None:
        return resolved, True
    fallback = normalize(default, category)
    return (fallback if fallback is not None else default), False


# =============================================================================
# BACKEND RESPONSE BUILDER (matches TaxonomyResponse schema contract)
# =============================================================================

# Legacy ids/labels/descriptions served by backend/app/api/v1/taxonomy.py before
# the canonical taxonomy existed. They are preserved verbatim (additively) in
# taxonomy_response() so existing frontend/service consumers keep working.
# shape: (id, label, description, icon, canonical_key)
_LEGACY_PACKAGE_TYPES = [
    ("bottle", "Bottle", "Glass or plastic bottles for beverages, spirits, oils, and skincare serums.", "bottle", "bottle"),
    ("box", "Box / Carton", "Rigid, folding, and corrugated gift boxes, retail cartons, and luxury packaging.", "package", "rigid_box"),
    ("can", "Can", "Aluminum beverage cans, aerosol cans, and tin canisters.", "disc", "can"),
    ("pouch", "Stand-Up Pouch", "Flexible doy-pack stand-up pouches with zip seals for food, powders, and coffee.", "shopping-bag", "stand_up_pouch"),
    ("jar", "Jar", "Cosmetic cream jars and culinary glass preserve containers.", "circle", "jar"),
    ("tube", "Squeeze Tube", "Flexible squeeze tubes for lotions, creams, toothpastes, and cosmetics.", "cylinder", "tube"),
]

# shape: (id, label, default_finish, compatible_finishes, description, canonical_key)
_LEGACY_MATERIALS = [
    ("cardboard", "Cardboard / Paperboard", "matte", ["matte", "glossy", "embossed", "foil_stamping"],
     "Sustainable folding boxboard with sharp print reproduction.", "coated_paperboard"),
    ("glass", "Glass", "glossy", ["glossy", "matte", "frosted", "embossed"],
     "Premium clear or tinted glass with realistic refractions and caustics.", "clear_glass"),
    ("aluminum", "Aluminum / Metal", "metallic", ["metallic", "matte", "glossy"],
     "Brushed or polished aluminum with authentic metallic sheen.", "brushed_aluminum"),
    ("kraft_paper", "Kraft Paper", "matte", ["matte", "embossed"],
     "Natural unbleached recycled fibrous paper for organic aesthetics.", "kraft_cardboard"),
    ("plastic", "Polymer / Plastic", "glossy", ["glossy", "matte", "frosted"],
     "PET and HDPE polymers for lightweight consumer goods packaging.", "flexible_mylar"),
]

# shape: (id, label, description, canonical_key)
_LEGACY_FINISHES = [
    ("glossy", "High Gloss UV Varnish", "Mirror-like specular reflections and rich color depth.", "gloss"),
    ("metallic", "Metallic Sheen", "Subtle metallic flakes producing light-reactive surface sheen.", "gloss"),
    ("foil_stamping", "Gold / Silver Foil Stamping", "Reflective hot-stamped foil accents for luxury branding.", "gold_foil"),
    ("frosted", "Frosted Translucent", "Etched semi-opaque surface with diffused light scattering.", "satin"),
]

# shape: (id, label, description, canonical_key)
_LEGACY_VIEWPOINTS = [
    ("front", "Front Eye-Level", "Direct eye-level frontal perspective emphasizing main label and form.", "front_view"),
    ("three_quarter", "Three-Quarter (3/4)", "Hero perspective revealing front label, side depth, and volume.", "three_quarter_view"),
    ("isometric", "Isometric 3D", "Balanced axonometric projection showing multiple panels simultaneously.", "three_quarter_view"),
]

# shape: (id, label, description, canonical_key)
_LEGACY_BACKGROUNDS = [
    ("studio_clean", "Clean Studio Infinity", "Neutral studio cyclorama with soft ground contact shadow.", "seamless_studio"),
    ("retail_shelf", "Retail Store Display", "Commercial supermarket or boutique shelf with warm ambient lighting.", "natural_environment"),
    ("minimalist_gradient", "Minimalist Gradient", "Modern aesthetic dual-tone monochromatic backdrop.", "seamless_studio"),
    ("outdoor_natural", "Outdoor Natural Ambient", "Soft sunlight and organic nature background elements.", "natural_environment"),
]

# shape: (id, label, description, canonical_key)
_LEGACY_LIGHTING = [
    ("natural_golden", "Golden Hour Warmth", "Late afternoon directional warm sunlight and soft long shadows.", "natural_daylight"),
]


def _item_response(key: str, ent: Dict[str, Any], canonical: str) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "canonical_category": canonical,
        "prompt_fragment": ent["prompt_fragment"],
    }
    return {
        "id": key,
        "label": ent["label"],
        "description": ent.get("description"),
        "icon": ent.get("icon"),
        "metadata": metadata,
    }


def taxonomy_response() -> Dict[str, List[Dict[str, Any]]]:
    """
    Build the backend taxonomy payload.

    Shape matches the existing ``TaxonomyResponse`` contract exactly
    (package_types / materials / finishes / viewpoints / backgrounds /
    lighting_presets). Canonical entries come first; pre-canonical ids used by
    earlier consumers (e.g. ``box``, ``glass``, ``front``, ``studio_clean``) are
    preserved additively as legacy items whose metadata points at the canonical
    key. Material entries carry ``default_finish`` and ``compatible_finishes``
    referencing ids that all exist in the ``finishes`` list.
    """
    package_types = [_item_response(k, e, "package_types") for k, e in PACKAGE_TYPES.items()]
    package_types += [
        {
            "id": lid,
            "label": label,
            "description": desc,
            "icon": icon,
            "metadata": {"legacy": True, "canonical_category": "package_types", "canonical_key": ck},
        }
        for lid, label, desc, icon, ck in _LEGACY_PACKAGE_TYPES
        if lid not in PACKAGE_TYPES
    ]

    finishes = [_item_response(k, e, "finishes") for k, e in FINISHES.items()]
    finishes += [
        {
            "id": lid,
            "label": label,
            "description": desc,
            "icon": None,
            "metadata": {"legacy": True, "canonical_category": "finishes", "canonical_key": ck},
        }
        for lid, label, desc, ck in _LEGACY_FINISHES
        if lid not in FINISHES
    ]

    viewpoints = [_item_response(k, e, "camera_angles") for k, e in CAMERA_ANGLES.items()]
    viewpoints += [
        {
            "id": lid,
            "label": label,
            "description": desc,
            "icon": None,
            "metadata": {"legacy": True, "canonical_category": "camera_angles", "canonical_key": ck},
        }
        for lid, label, desc, ck in _LEGACY_VIEWPOINTS
        if lid not in CAMERA_ANGLES
    ]

    backgrounds = [_item_response(k, e, "backgrounds") for k, e in BACKGROUNDS.items()]
    backgrounds += [
        {
            "id": lid,
            "label": label,
            "description": desc,
            "icon": None,
            "metadata": {"legacy": True, "canonical_category": "backgrounds", "canonical_key": ck},
        }
        for lid, label, desc, ck in _LEGACY_BACKGROUNDS
        if lid not in BACKGROUNDS
    ]

    lighting_presets = [_item_response(k, e, "lighting") for k, e in LIGHTING.items()]
    lighting_presets += [
        {
            "id": lid,
            "label": label,
            "description": desc,
            "icon": None,
            "metadata": {"legacy": True, "canonical_category": "lighting", "canonical_key": ck},
        }
        for lid, label, desc, ck in _LEGACY_LIGHTING
        if lid not in LIGHTING
    ]

    materials: List[Dict[str, Any]] = []
    for key, ent in MATERIALS.items():
        materials.append(
            {
                "id": key,
                "label": ent["label"],
                "default_finish": ent["default_finish"],
                "compatible_finishes": list(ent["compatible_finishes"]),
                "description": ent.get("description"),
            }
        )
    materials += [
        {
            "id": lid,
            "label": label,
            "default_finish": default_finish,
            "compatible_finishes": list(compat),
            "description": desc,
        }
        for lid, label, default_finish, compat, desc, _ck in _LEGACY_MATERIALS
        if lid not in MATERIALS
    ]

    return {
        "package_types": package_types,
        "materials": materials,
        "finishes": finishes,
        "viewpoints": viewpoints,
        "backgrounds": backgrounds,
        "lighting_presets": lighting_presets,
    }
