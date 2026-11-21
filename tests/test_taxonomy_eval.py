"""
Tests for the canonical packaging taxonomy (src/taxonomy.py), the prompt engine
integration (src/prompt_engine.py), and honest evaluation metric provenance
(src/evaluation.py).

No test in this module touches the network: CLIP is forced offline and the VLM
path is exercised through a monkeypatched fake client.
"""

import sys
import types
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.evaluation as evaluation_module
from src import taxonomy
from src.evaluation import (
    METRIC_PROVENANCE_KEYS,
    PackagingEvaluator,
    _VLM_CACHE,
    _VLM_CACHE_SIZE,
    _heuristic_clip_score,
    _parse_vlm_score,
    evaluate_render as module_evaluate_render,
)
from src.prompt_engine import (
    BACKGROUNDS,
    CAMERA_ANGLES,
    CATEGORIES,
    FINISHES,
    LIGHTING_PRESETS,
    SUBSTRATES,
    PromptEngine,
    build_packaging_prompt,
)

CANONICAL_COUNTS = {
    "package_types": 10,
    "materials": 12,
    "finishes": 11,
    "lighting": 5,
    "camera_angles": 4,
    "backgrounds": 5,
}


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def _clean_evaluation_state(monkeypatch):
    """Force a clean, order-independent ``src.evaluation`` state for every test.

    Other test modules in ``tests/`` (e.g. ``test_cloud_engine.py``) exercise the
    real ``src.vlm_client`` and may leave response/negative caches, cooldowns, or
    environment variables behind. This fixture resets everything
    ``src.evaluation`` consults so each test starts pristine regardless of what
    ran before it.
    """
    # Neutralize any leaked opt-in so the offline guard is authoritative unless
    # a specific test explicitly opts in via the allow_network fixture.
    monkeypatch.delenv("PACKVISION_ALLOW_MODEL_DOWNLOADS", raising=False)
    evaluation_module._reset_caches()
    yield
    evaluation_module._reset_caches()


@pytest.fixture
def gray_image() -> Image.Image:
    return Image.new("RGB", (64, 64), color=(120, 120, 120))


@pytest.fixture
def offline_evaluator(monkeypatch) -> PackagingEvaluator:
    """Evaluator with real CLIP forced unavailable (no local weights, no network)."""
    monkeypatch.setattr(PackagingEvaluator, "_init_clip", lambda self: False)
    return PackagingEvaluator()


@pytest.fixture
def allow_network(monkeypatch):
    """Explicitly opt in to model loading so guard logic does not short-circuit.

    Only needed for tests that exercise the CLIP/VLM *plumbing* with fakes; the
    default guard stays active in every other test.
    """
    monkeypatch.setenv("PACKVISION_ALLOW_MODEL_DOWNLOADS", "true")


@pytest.fixture
def fake_vlm(monkeypatch, allow_network):
    """Patch the defensively-imported VLM hooks with a controllable fake."""
    calls = {"count": 0, "prompts": []}

    def fake_available() -> bool:
        return True

    def fake_ask(prompt, image=None, json_mode=False, max_tokens=512):
        calls["count"] += 1
        calls["prompts"].append(prompt)
        return '{"score": 8}'

    monkeypatch.setattr(evaluation_module, "_vlm_available", fake_available)
    monkeypatch.setattr(evaluation_module, "_ask_vlm", fake_ask)
    return calls


# =============================================================================
# TAXONOMY: STRUCTURE
# =============================================================================

@pytest.mark.parametrize("category,expected_count", sorted(CANONICAL_COUNTS.items()))
def test_canonical_category_sizes(category, expected_count):
    keys = taxonomy.all_keys(category)
    assert len(keys) == expected_count
    assert len(keys) == len(set(keys)), "duplicate canonical keys"


def test_unknown_category_returns_empty():
    assert taxonomy.all_keys("bogus_category") == []
    assert taxonomy.normalize("kraft", "bogus_category") is None
    assert taxonomy.entry("materials", "kraft") is not None
    assert taxonomy.entry("bogus_category", "kraft") is None


@pytest.mark.parametrize("category", sorted(CANONICAL_COUNTS))
def test_every_canonical_key_round_trips(category):
    for key in taxonomy.all_keys(category):
        assert taxonomy.normalize(key, category) == key, key
        ent = taxonomy.entry(category, key)
        assert ent is not None
        assert ent["label"], key
        assert ent["prompt_fragment"], key
        assert isinstance(ent["synonyms"], list) and ent["synonyms"], key


@pytest.mark.parametrize("category", sorted(CANONICAL_COUNTS))
def test_every_declared_synonym_resolves_back(category):
    """Every declared synonym must normalize to its own canonical key."""
    for key in taxonomy.all_keys(category):
        ent = taxonomy.entry(category, key)
        for syn in ent["synonyms"]:
            assert taxonomy.normalize(syn, category) == key, (category, key, syn)
        # Display label must resolve as well
        assert taxonomy.normalize(ent["label"], category) == key, (category, key)


def test_normalize_is_separator_and_case_insensitive():
    assert taxonomy.normalize("  GOLD   FOIL ", "finishes") == "gold_foil"
    assert taxonomy.normalize("gold-foil", "finishes") == "gold_foil"
    assert taxonomy.normalize("GOLD_FOIL", "finishes") == "gold_foil"
    assert taxonomy.normalize("Kraft Cardboard", "materials") == "kraft_cardboard"


@pytest.mark.parametrize(
    "raw",
    [None, "", "   ", 42, "quantum platypus", "totally unknown material"],
)
def test_unknown_values_return_none(raw):
    assert taxonomy.normalize(raw, "materials") is None


def test_ensure_valid_flags():
    key, was_normalized = taxonomy.ensure_valid("materials", "Kraft", "coated_paperboard")
    assert key == "kraft_cardboard"
    assert was_normalized is True

    key, was_normalized = taxonomy.ensure_valid("materials", "kraft_cardboard", "coated_paperboard")
    assert key == "kraft_cardboard"
    assert was_normalized is True

    key, was_normalized = taxonomy.ensure_valid("materials", "unobtanium board", "coated_paperboard")
    assert key == "coated_paperboard"
    assert was_normalized is False

    key, was_normalized = taxonomy.ensure_valid("materials", None, "coated_paperboard")
    assert key == "coated_paperboard"
    assert was_normalized is False


# =============================================================================
# TAXONOMY: SYNONYM COVERAGE OF EXISTING CALL SITES
# =============================================================================

@pytest.mark.parametrize(
    "raw,category,expected",
    [
        # --- ui/app_gradio.py dropdown labels ---
        ("Carton", "package_types", "folding_carton"),
        ("Box", "package_types", "rigid_box"),
        ("Pouch", "package_types", "stand_up_pouch"),
        ("Kraft", "materials", "kraft_cardboard"),
        ("Corrugated", "materials", "corrugated_cardboard"),
        ("Matte Carton", "materials", "matte_laminated"),
        ("Flexible Mylar", "materials", "flexible_mylar"),
        ("Frosted Glass", "materials", "frosted_glass"),
        ("Brushed Aluminum", "materials", "brushed_aluminum"),
        ("Gold Foil", "finishes", "gold_foil"),
        ("Silver Foil", "finishes", "silver_foil"),
        ("Spot UV", "finishes", "spot_uv"),
        ("Embossed", "finishes", "embossed"),
        ("Debossed", "finishes", "debossed"),
        ("Holographic", "finishes", "holographic_film"),
        ("Matte", "finishes", "matte"),
        ("Three-Point Studio Softbox", "lighting", "studio_softbox"),
        ("Dramatic Luxury Rim Light", "lighting", "dramatic_rim"),
        ("Commercial High-Key Clean", "lighting", "high_key_commercial"),
        ("Natural Window Daylight", "lighting", "natural_daylight"),
        ("Front-Facing Hero", "camera_angles", "front_view"),
        ("Three-Quarter Isometric", "camera_angles", "three_quarter_view"),
        ("Low-Angle Dynamic", "camera_angles", "hero_low_angle"),
        ("Top-Down Flat Lay", "camera_angles", "top_down"),
        ("Seamless White/Neutral Studio", "backgrounds", "seamless_studio"),
        ("Warm Beige Pedestal", "backgrounds", "warm_pedestal"),
        ("Minimalist Dark Luxury Stone", "backgrounds", "dark_luxury_scene"),
        ("Transparent/Isolated", "backgrounds", "transparent_isolated"),
        # --- backend/app/api/v1/taxonomy.py legacy ids ---
        ("box", "package_types", "rigid_box"),
        ("pouch", "package_types", "stand_up_pouch"),
        ("bottle", "package_types", "bottle"),
        ("can", "package_types", "can"),
        ("jar", "package_types", "jar"),
        ("tube", "package_types", "tube"),
        ("cardboard", "materials", "coated_paperboard"),
        ("glass", "materials", "clear_glass"),
        ("aluminum", "materials", "brushed_aluminum"),
        ("kraft_paper", "materials", "kraft_cardboard"),
        ("plastic", "materials", "flexible_mylar"),
        ("glossy", "finishes", "gloss"),
        ("metallic", "finishes", "gloss"),
        ("foil_stamping", "finishes", "gold_foil"),
        ("frosted", "finishes", "satin"),
        ("front", "camera_angles", "front_view"),
        ("three_quarter", "camera_angles", "three_quarter_view"),
        ("isometric", "camera_angles", "three_quarter_view"),
        ("top_down", "camera_angles", "top_down"),
        ("studio_clean", "backgrounds", "seamless_studio"),
        ("retail_shelf", "backgrounds", "natural_environment"),
        ("minimalist_gradient", "backgrounds", "seamless_studio"),
        ("outdoor_natural", "backgrounds", "natural_environment"),
        ("natural_golden", "lighting", "natural_daylight"),
        # --- src/prompt_engine.py legacy dictionary keys ---
        ("carton", "package_types", "folding_carton"),
        ("kraft", "materials", "kraft_cardboard"),
        ("corrugated", "materials", "corrugated_cardboard"),
        ("matte carton", "materials", "matte_laminated"),
        ("coated SBS", "materials", "coated_paperboard"),
        ("flexible pouch", "materials", "flexible_mylar"),
        ("frosted glass", "materials", "frosted_glass"),
        ("brushed aluminum", "materials", "brushed_aluminum"),
        ("molded pulp", "materials", "molded_pulp"),
        ("gold foil", "finishes", "gold_foil"),
        ("silver foil", "finishes", "silver_foil"),
        ("spot UV", "finishes", "spot_uv"),
        ("holographic", "finishes", "holographic_film"),
        ("uncoated", "finishes", "uncoated"),
        ("studio softbox", "lighting", "studio_softbox"),
        ("dramatic rim", "lighting", "dramatic_rim"),
        ("high-key commercial", "lighting", "high_key_commercial"),
        ("natural daylight", "lighting", "natural_daylight"),
        ("three-quarter", "camera_angles", "three_quarter_view"),
        ("top-down", "camera_angles", "top_down"),
        ("seamless studio", "backgrounds", "seamless_studio"),
        ("warm pedestal", "backgrounds", "warm_pedestal"),
        ("minimalist marble", "backgrounds", "warm_pedestal"),
        ("dark luxury", "backgrounds", "dark_luxury_scene"),
        # --- data/dataset_generator.py snake_case keys ---
        ("kraft_cardboard", "materials", "kraft_cardboard"),
        ("corrugated_box", "materials", "corrugated_cardboard"),
        ("matte_carton", "materials", "matte_laminated"),
        ("coated_sbs", "materials", "coated_paperboard"),
        ("flexible_pouch", "materials", "flexible_mylar"),
        ("frosted_glass", "materials", "frosted_glass"),
        ("brushed_aluminum", "materials", "brushed_aluminum"),
        ("gold_foil", "finishes", "gold_foil"),
        ("silver_foil", "finishes", "silver_foil"),
        ("spot_uv", "finishes", "spot_uv"),
        ("studio_softbox", "lighting", "studio_softbox"),
        ("dramatic_rim", "lighting", "dramatic_rim"),
        ("high_key_commercial", "lighting", "high_key_commercial"),
        ("natural_daylight", "lighting", "natural_daylight"),
        ("seamless_studio", "backgrounds", "seamless_studio"),
        ("warm_pedestal", "backgrounds", "warm_pedestal"),
        ("dark_luxury", "backgrounds", "dark_luxury_scene"),
        ("three_quarter", "camera_angles", "three_quarter_view"),
        # --- natural variants required by the spec ---
        ("3/4 view", "camera_angles", "three_quarter_view"),
        ("3/4", "camera_angles", "three_quarter_view"),
        ("front view", "camera_angles", "front_view"),
    ],
)
def test_synonym_variants_from_all_call_sites(raw, category, expected):
    assert taxonomy.normalize(raw, category) == expected


# =============================================================================
# TAXONOMY: BACKEND RESPONSE CONTRACT
# =============================================================================

def test_taxonomy_response_matches_backend_contract():
    data = taxonomy.taxonomy_response()
    for section in (
        "package_types",
        "materials",
        "finishes",
        "viewpoints",
        "backgrounds",
        "lighting_presets",
    ):
        assert section in data
        assert isinstance(data[section], list) and data[section]

    pt_ids = {i["id"] for i in data["package_types"]}
    # Legacy ids preserved (additively) ...
    assert {"bottle", "box", "can", "pouch", "jar", "tube"} <= pt_ids
    # ... alongside canonical keys.
    assert {
        "folding_carton", "rigid_box", "mailer_box", "stand_up_pouch", "flat_pouch",
        "bottle", "jar", "can", "tube", "sleeve",
    } <= pt_ids

    mat_ids = {m["id"] for m in data["materials"]}
    assert {"cardboard", "glass", "aluminum", "kraft_paper", "plastic"} <= mat_ids
    assert {
        "kraft_cardboard", "corrugated_cardboard", "coated_paperboard", "matte_laminated",
        "uncoated_paper", "flexible_mylar", "frosted_glass", "clear_glass", "amber_glass",
        "brushed_aluminum", "tinplate_metal", "molded_pulp",
    } <= mat_ids

    finish_ids = {f["id"] for f in data["finishes"]}
    assert {"matte"} <= finish_ids
    assert {
        "matte", "gloss", "satin", "spot_uv", "gold_foil", "silver_foil",
        "holographic_film", "embossed", "debossed", "soft_touch", "uncoated",
    } <= finish_ids

    view_ids = {v["id"] for v in data["viewpoints"]}
    assert {"front", "three_quarter", "isometric", "top_down"} <= view_ids
    assert {"front_view", "three_quarter_view", "top_down", "hero_low_angle"} <= view_ids

    bg_ids = {b["id"] for b in data["backgrounds"]}
    assert {"studio_clean", "retail_shelf", "minimalist_gradient", "outdoor_natural"} <= bg_ids
    assert {
        "seamless_studio", "warm_pedestal", "dark_luxury_scene",
        "natural_environment", "transparent_isolated",
    } <= bg_ids

    light_ids = {l["id"] for l in data["lighting_presets"]}
    assert {"studio_softbox", "dramatic_rim", "natural_golden"} <= light_ids
    assert {
        "studio_softbox", "dramatic_rim", "high_key_commercial",
        "natural_daylight", "dark_luxury",
    } <= light_ids


def test_taxonomy_response_materials_are_internally_consistent():
    data = taxonomy.taxonomy_response()
    finish_ids = {f["id"] for f in data["finishes"]}
    for material in data["materials"]:
        assert material["default_finish"] in finish_ids, material["id"]
        for finish in material["compatible_finishes"]:
            assert finish in finish_ids, (material["id"], finish)


def test_taxonomy_response_legacy_items_marked_with_canonical_key():
    data = taxonomy.taxonomy_response()
    legacy_ids = {}
    for section in ("package_types", "finishes", "viewpoints", "backgrounds", "lighting_presets"):
        for item in data[section]:
            if item["metadata"].get("legacy"):
                legacy_ids[item["id"]] = item["metadata"]["canonical_key"]
    assert legacy_ids["box"] == "rigid_box"
    assert legacy_ids["studio_clean"] == "seamless_studio"
    assert legacy_ids["front"] == "front_view"
    assert legacy_ids["foil_stamping"] == "gold_foil"
    assert legacy_ids["natural_golden"] == "natural_daylight"


def test_taxonomy_response_validates_against_backend_schema():
    pytest.importorskip("pydantic")
    from backend.app.schemas.taxonomy import TaxonomyResponse

    data = taxonomy.taxonomy_response()
    # Must validate without dropping/renaming any required field.
    response = TaxonomyResponse(**data)
    assert len(response.package_types) == len(data["package_types"])
    assert len(response.materials) == len(data["materials"])
    assert {pt.id for pt in response.package_types} >= {"bottle", "box"}
    assert {m.id for m in response.materials} >= {"glass"}


# =============================================================================
# PROMPT ENGINE INTEGRATION
# =============================================================================

def test_alias_dictionaries_are_derived_from_taxonomy():
    assert set(CATEGORIES) == set(taxonomy.all_keys("package_types"))
    assert set(SUBSTRATES) == set(taxonomy.all_keys("materials"))
    assert set(FINISHES) == set(taxonomy.all_keys("finishes"))
    assert set(LIGHTING_PRESETS) == set(taxonomy.all_keys("lighting"))
    assert set(CAMERA_ANGLES) == set(taxonomy.all_keys("camera_angles"))
    assert set(BACKGROUNDS) == set(taxonomy.all_keys("backgrounds"))
    # Descriptor attributes stay populated for existing consumers.
    assert CATEGORIES["folding_carton"].structural_cues
    assert SUBSTRATES["kraft_cardboard"].shader_description
    assert FINISHES["gold_foil"].specular_profile
    assert LIGHTING_PRESETS["studio_softbox"].lighting_setup
    assert CAMERA_ANGLES["three_quarter_view"].framing
    assert BACKGROUNDS["seamless_studio"].surface_description


def test_prompt_engine_returns_canonical_fragments():
    engine = PromptEngine()
    pkg = engine.build_prompt(
        category="carton",
        substrate="kraft",
        finish="gold foil",
        lighting="studio softbox",
        camera_angle="three-quarter",
        background="seamless studio",
    )
    assert pkg.category == "folding_carton"
    assert pkg.substrate == "kraft_cardboard"
    assert pkg.finish == "gold_foil"
    assert pkg.lighting == "studio_softbox"
    assert pkg.camera_angle == "three_quarter_view"
    assert pkg.background == "seamless_studio"
    # Rich fragments from the canonical taxonomy land in the positive prompt
    assert "unbleached raw organic brown kraft paperboard" in pkg.positive_prompt
    assert "hot-stamped 24-karat" in pkg.positive_prompt
    assert pkg.metadata["normalized_inputs"]["substrate"] == "kraft_cardboard"
    assert pkg.metadata["normalized_inputs"]["finish"] == "gold_foil"
    assert pkg.metadata["unrecognized_inputs"] == []


def test_prompt_engine_resolves_gradio_labels_not_coated_sbs_fallback():
    """The audited bug: Gradio labels used to silently fall back to defaults."""
    engine = PromptEngine()
    pkg = engine.build_prompt(
        category="Carton",
        substrate="Frosted Glass",
        finish="Gold Foil",
        lighting="Natural Window Daylight",
        camera_angle="Three-Quarter Isometric",
        background="Transparent/Isolated",
    )
    assert pkg.substrate == "frosted_glass"
    assert pkg.finish == "gold_foil"
    assert pkg.lighting == "natural_daylight"
    assert pkg.camera_angle == "three_quarter_view"
    assert pkg.background == "transparent_isolated"
    assert "acid-etched frosted soda-lime" in pkg.positive_prompt
    assert "coated paperboard" not in pkg.positive_prompt


def test_prompt_engine_resolves_snake_case_and_legacy_backend_ids():
    engine = PromptEngine()
    pkg = engine.build_prompt(
        category="box",
        substrate="coated_sbs",
        finish="foil_stamping",
        lighting="dramatic_rim",
        camera_angle="three_quarter",
        background="studio_clean",
    )
    assert pkg.category == "rigid_box"
    assert pkg.substrate == "coated_paperboard"
    assert pkg.finish == "gold_foil"
    assert pkg.lighting == "dramatic_rim"
    assert pkg.camera_angle == "three_quarter_view"
    assert pkg.background == "seamless_studio"


def test_prompt_engine_reports_unrecognized_inputs():
    engine = PromptEngine()
    pkg = engine.build_prompt(substrate="unobtanium board", finish="platinum leaf")
    # Graceful fallback preserved ...
    assert pkg.substrate == "coated_paperboard"
    assert pkg.finish == "matte"
    # ... but now reported instead of silent.
    assert pkg.metadata["normalized_inputs"]["substrate"] == "coated_paperboard"
    reported = " | ".join(pkg.metadata["unrecognized_inputs"])
    assert "substrate" in reported and "unobtanium board" in reported
    assert "finish" in reported and "platinum leaf" in reported
    assert pkg.metadata["normalized_inputs"]["category"] == "folding_carton"


def test_build_packaging_prompt_convenience_tuple():
    pos, neg = build_packaging_prompt(
        category="pouch",
        substrate="flexible mylar",
        finish="matte",
        brand_name="ACME",
        color_palette="sage green",
    )
    assert isinstance(pos, str) and isinstance(neg, str) and pos and neg
    assert "stand-up flexible pouch" in pos
    assert "multi-ply high-barrier" in pos
    assert "'ACME'" in pos
    # matte is intentionally omitted from the embellishment clause
    assert "embellished with" not in pos


def test_prompt_engine_listing_helpers_match_canonical_keys():
    engine = PromptEngine()
    assert engine.list_categories() == taxonomy.all_keys("package_types")
    assert engine.list_substrates() == taxonomy.all_keys("materials")
    assert engine.list_finishes() == taxonomy.all_keys("finishes")
    assert engine.list_lighting() == taxonomy.all_keys("lighting")
    assert engine.list_camera_angles() == taxonomy.all_keys("camera_angles")
    assert engine.list_backgrounds() == taxonomy.all_keys("backgrounds")


# =============================================================================
# EVALUATION: PROVENANCE SCHEME
# =============================================================================

def test_metric_provenance_keys_documented():
    assert set(METRIC_PROVENANCE_KEYS) == {"library", "vlm", "heuristic", "unavailable"}
    for label, doc in METRIC_PROVENANCE_KEYS.items():
        assert isinstance(doc, str) and doc, label


def test_heuristic_clip_score_is_bounded_and_documented(gray_image):
    score = _heuristic_clip_score(gray_image, "kraft pouch packaging with gold foil")
    assert 0.22 <= score <= 0.38
    doc = (_heuristic_clip_score.__doc__ or "").lower()
    assert "not a real clip score" in doc


def test_clip_score_heuristic_source_labeled(offline_evaluator, gray_image):
    detail = offline_evaluator.compute_clip_score_detailed(gray_image, "kraft pouch packaging")
    assert detail["source"] == "heuristic"
    assert 0.0 <= detail["score"] <= 1.0
    assert offline_evaluator.last_metric_sources["clip_score"] == "heuristic"
    # float-returning public API unchanged
    assert offline_evaluator.compute_clip_score(gray_image, "kraft pouch packaging") == pytest.approx(detail["score"])


def test_clip_score_vlm_path_with_fake_client(offline_evaluator, fake_vlm, gray_image):
    detail = offline_evaluator.compute_clip_score_detailed(gray_image, "gold foil box")
    assert detail["source"] == "vlm"
    assert detail["score"] == pytest.approx(0.8)
    assert fake_vlm["count"] == 1
    assert "gold foil box" in fake_vlm["prompts"][0]
    assert offline_evaluator.last_metric_sources["clip_score"] == "vlm"


def test_vlm_score_cached_by_image_and_text(offline_evaluator, fake_vlm, gray_image):
    first = offline_evaluator.compute_clip_score(gray_image, "cache probe text")
    second = offline_evaluator.compute_clip_score(gray_image, "cache probe text")
    assert first == second == pytest.approx(0.8)
    assert fake_vlm["count"] == 1  # second call served from the LRU cache
    # Different text must miss the cache
    offline_evaluator.compute_clip_score(gray_image, "different text")
    assert fake_vlm["count"] == 2
    assert len(_VLM_CACHE) <= _VLM_CACHE_SIZE


def test_vlm_unparseable_reply_falls_back_to_heuristic(offline_evaluator, monkeypatch, allow_network, gray_image):
    monkeypatch.setattr(evaluation_module, "_vlm_available", lambda: True)
    monkeypatch.setattr(evaluation_module, "_ask_vlm", lambda prompt, image=None: "I cannot rate that, sorry.")
    detail = offline_evaluator.compute_clip_score_detailed(gray_image, "some prompt")
    assert detail["source"] == "heuristic"
    assert offline_evaluator.last_metric_sources["clip_score"] == "heuristic"


def test_vlm_unavailable_flag_falls_back_to_heuristic(offline_evaluator, monkeypatch, allow_network, gray_image):
    monkeypatch.setattr(evaluation_module, "_vlm_available", lambda: False)
    monkeypatch.setattr(
        evaluation_module, "_ask_vlm",
        lambda prompt, image=None: pytest.fail("ask_vlm must not be called when unavailable"),
    )
    detail = offline_evaluator.compute_clip_score_detailed(gray_image, "some prompt")
    assert detail["source"] == "heuristic"


def test_vlm_exception_falls_back_to_heuristic(offline_evaluator, monkeypatch, allow_network, gray_image):
    def boom(prompt, image=None):
        raise RuntimeError("provider down")

    monkeypatch.setattr(evaluation_module, "_vlm_available", lambda: True)
    monkeypatch.setattr(evaluation_module, "_ask_vlm", boom)
    detail = offline_evaluator.compute_clip_score_detailed(gray_image, "some prompt")
    assert detail["source"] == "heuristic"


def test_parse_vlm_score_robustness():
    assert _parse_vlm_score('{"score": 8}') == pytest.approx(0.8)
    assert _parse_vlm_score('{"score": 10}') == pytest.approx(1.0)
    assert _parse_vlm_score('{"score": 0}') == pytest.approx(0.0)
    assert _parse_vlm_score('{"rating": 7}') == pytest.approx(0.7)
    assert _parse_vlm_score('noise before {"score": 4} noise after') == pytest.approx(0.4)
    assert _parse_vlm_score("I would say 3 out of 10") == pytest.approx(0.3)
    assert _parse_vlm_score('{"score": 42}') == pytest.approx(1.0)  # clamped
    assert _parse_vlm_score('{"score": -5}') == pytest.approx(0.0)  # clamped
    assert _parse_vlm_score("no numbers here") is None
    assert _parse_vlm_score("") is None
    assert _parse_vlm_score(None) is None


def test_fid_and_lpips_are_labeled_heuristic_proxies(offline_evaluator, gray_image):
    doc_lpips = (PackagingEvaluator.compute_lpips.__doc__ or "").lower()
    doc_fid = (PackagingEvaluator.compute_fid.__doc__ or "").lower()
    assert "proxy" in doc_lpips and "not true lpips" in doc_lpips
    assert "proxy" in doc_fid and "not true fid" in doc_fid

    lpips = offline_evaluator.compute_lpips(gray_image, gray_image)
    assert 0.0 <= lpips <= 1.0
    assert offline_evaluator.last_metric_sources["lpips_proxy"] == "heuristic"

    fid = offline_evaluator.compute_fid([gray_image], [gray_image])
    assert isinstance(fid, float)
    assert offline_evaluator.last_metric_sources["fid_proxy"] == "heuristic"


def test_evaluate_sample_carries_provenance(offline_evaluator, gray_image):
    result = offline_evaluator.evaluate_sample("s1", gray_image, gray_image, "prompt text")
    assert result.clip_source == "heuristic"
    assert result.metric_sources == {
        "edge_iou": "library",
        "edge_ssim": "library",
        "clip_score": "heuristic",
        "lpips_proxy": "heuristic",
    }
    as_dict = result.to_dict()
    assert as_dict["clip_source"] == "heuristic"
    assert as_dict["metric_sources"]["clip_score"] == "heuristic"


# =============================================================================
# EVALUATION: evaluate_render CONVENIENCE
# =============================================================================

def test_evaluate_render_returns_score_source(offline_evaluator, gray_image):
    result = offline_evaluator.evaluate_render(gray_image, "kraft cardboard mailer box")
    assert result["score_source"] == "heuristic"
    assert result["metric_sources"]["clip_score"] == "heuristic"
    assert "clip_score" in result and 0.0 <= result["clip_score"] <= 1.0
    assert result["sample_id"] == "render_001"
    # No sketch -> structural metrics omitted
    assert "edge_iou" not in result


def test_evaluate_render_with_sketch_includes_structural_metrics(offline_evaluator, gray_image):
    result = offline_evaluator.evaluate_render(gray_image, "luxury serum bottle", sketch=gray_image)
    assert {"edge_iou", "edge_ssim", "lpips_proxy"} <= set(result)
    assert result["edge_iou"] == pytest.approx(1.0)  # identical images
    assert result["metric_sources"]["edge_iou"] == "library"
    assert result["metric_sources"]["lpips_proxy"] == "heuristic"
    # Best available provenance wins: library edge metrics exist alongside the
    # heuristic CLIP proxy.
    assert result["score_source"] == "library"


def test_evaluate_render_vlm_source_summary(offline_evaluator, fake_vlm, gray_image):
    result = offline_evaluator.evaluate_render(gray_image, "gold foil rigid box")
    assert result["clip_source"] == "vlm"
    assert result["score_source"] == "vlm"


def test_module_level_evaluate_render(monkeypatch, gray_image):
    monkeypatch.setattr(PackagingEvaluator, "_init_clip", lambda self: False)
    result = module_evaluate_render(gray_image, "matte tube with soft touch finish")
    assert result["score_source"] == "heuristic"
    # The shared evaluator is reused across calls (same input -> same score).
    shared = module_evaluate_render(gray_image, "matte tube with soft touch finish")
    assert shared["clip_score"] == result["clip_score"]


# =============================================================================
# EVALUATION: CLIP GATE (offline under pytest; opt-in model downloads)
# =============================================================================

def test_clip_loads_local_weights_even_in_mock_mode(monkeypatch, allow_network):
    """USE_MOCK_INFERENCE must not disable real CLIP when weights are cached.

    Requires the explicit PACKVISION_ALLOW_MODEL_DOWNLOADS opt-in so the default
    pytest run never touches HF loading at all.
    """
    monkeypatch.setenv("USE_MOCK_INFERENCE", "true")

    class FakeCLIPModel:
        @classmethod
        def from_pretrained(cls, name, local_files_only=False, **kwargs):
            assert local_files_only is True, "must never attempt a network download"
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

    class FakeCLIPProcessor:
        @classmethod
        def from_pretrained(cls, name, local_files_only=False, **kwargs):
            assert local_files_only is True
            return cls()

    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
    transformers_mod = types.ModuleType("transformers")
    transformers_mod.CLIPModel = FakeCLIPModel
    transformers_mod.CLIPProcessor = FakeCLIPProcessor
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

    evaluator = PackagingEvaluator()
    assert evaluator._init_clip() is True
    assert evaluator.clip_model is not None


def test_offline_guard_blocks_clip_and_vlm_under_pytest(monkeypatch, tmp_path, gray_image):
    """Default pytest run: no CLIP loading and no VLM calls - heuristics only."""
    monkeypatch.delenv("PACKVISION_ALLOW_MODEL_DOWNLOADS", raising=False)
    monkeypatch.setenv("USE_MOCK_INFERENCE", "true")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))  # no cached weights
    assert evaluation_module._offline_guard_active() is True
    assert evaluation_module._model_downloads_allowed() is False

    # Even with perfectly loadable fake transformers present, the guard wins.
    class ShouldNotLoad:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise AssertionError("CLIP must not be initialized under pytest without opt-in")

    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
    transformers_mod = types.ModuleType("transformers")
    transformers_mod.CLIPModel = ShouldNotLoad
    transformers_mod.CLIPProcessor = ShouldNotLoad
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

    evaluator = PackagingEvaluator()
    assert evaluator._init_clip() is False

    # And the VLM path must be inert too: if ask_vlm is reached, fail loudly.
    monkeypatch.setattr(evaluation_module, "_vlm_available", lambda: True)
    monkeypatch.setattr(
        evaluation_module, "_ask_vlm",
        lambda *a, **k: pytest.fail("VLM must not be called under pytest without opt-in"),
    )
    detail = evaluator.compute_clip_score_detailed(gray_image, "some prompt")
    assert detail["source"] == "heuristic"


def test_opt_in_env_var_deactivates_offline_guard(monkeypatch):
    monkeypatch.setenv("PACKVISION_ALLOW_MODEL_DOWNLOADS", "true")
    assert evaluation_module._model_downloads_allowed() is True
    assert evaluation_module._offline_guard_active() is False
    monkeypatch.setenv("PACKVISION_ALLOW_MODEL_DOWNLOADS", "false")
    assert evaluation_module._offline_guard_active() is True


def test_clip_init_fails_gracefully_without_weights(monkeypatch, allow_network):
    """With no transformers installed (or no cached weights), init returns False.

    Fake torch is injected so the real (multi-second) torch import never runs -
    this test must fail fast, not on a network or import timeout.
    """
    monkeypatch.setenv("USE_MOCK_INFERENCE", "true")
    monkeypatch.setitem(
        sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    )
    monkeypatch.setitem(sys.modules, "transformers", None)  # forces ImportError

    evaluator = PackagingEvaluator()
    assert evaluator._init_clip() is False
    assert evaluator.clip_model is None
    detail = evaluator.compute_clip_score_detailed(Image.new("RGB", (8, 8), (0, 0, 0)), "text")
    assert detail["source"] in ("vlm", "heuristic")


def test_clip_precheck_short_circuits_without_cached_weights(monkeypatch, tmp_path):
    """No cached weights + no opt-in -> instant False, no torch/transformers import."""
    monkeypatch.delenv("PACKVISION_ALLOW_MODEL_DOWNLOADS", raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))  # empty cache dir

    assert evaluation_module._clip_weights_locally_cached("openai/clip-vit-base-patch32") is False

    # Poison the heavy imports: if _init_clip tried them, the test would blow up.
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)

    evaluator = PackagingEvaluator()
    assert evaluator._init_clip() is False  # instant short-circuit, no imports


def test_clip_precheck_detects_cached_weights(monkeypatch, tmp_path):
    model_dir = tmp_path / "models--openai--clip-vit-base-patch32"
    model_dir.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    assert evaluation_module._clip_weights_locally_cached("openai/clip-vit-base-patch32") is True
    assert evaluation_module._clip_weights_locally_cached("org/other-model") is False


def test_reset_caches_hook_restores_clean_state(offline_evaluator, fake_vlm, gray_image):
    """_reset_caches() drops the VLM LRU and the shared evaluator singleton."""
    # Populate both caches.
    offline_evaluator.compute_clip_score(gray_image, "reset probe")
    evaluation_module._get_shared_evaluator()

    assert len(_VLM_CACHE) > 0
    assert evaluation_module._SHARED_EVALUATOR is not None

    evaluation_module._reset_caches()

    assert len(_VLM_CACHE) == 0
    assert evaluation_module._SHARED_EVALUATOR is None
