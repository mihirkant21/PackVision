"""Tests for the local packaging configuration classifier (src/config_classifier.py).

Verifies the trained checkpoint loads, that ``predict_config`` returns canonical
``src.taxonomy`` keys for the benchmark sketches, accepts PIL Images directly,
and is fast enough for interactive use on CPU (< 10s for load + predictions).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_classifier import (  # noqa: E402
    CATEGORY_KEYS,
    CHECKPOINT_PATH,
    FINISH_KEYS,
    MATERIAL_KEYS,
    build_model,
    load_checkpoint,
    predict_config,
)
from src.taxonomy import all_keys, normalize  # noqa: E402

BENCHMARK_SKETCHES = [
    ROOT / "data/samples/carton_sketch.png",
    ROOT / "data/samples/bottle_sketch.png",
    ROOT / "data/samples/pouch_sketch.png",
]


# =============================================================================
# CHECKPOINT
# =============================================================================

def test_checkpoint_exists_and_loads():
    assert CHECKPOINT_PATH.exists(), (
        f"missing trained checkpoint {CHECKPOINT_PATH} -- run "
        f"scripts/train_config_classifier.py first"
    )
    bundle = load_checkpoint(force_reload=True)
    ck = bundle["checkpoint"]
    assert "model_state_dict" in ck
    assert "val_metrics" in ck
    assert isinstance(ck["val_metrics"], dict)

    # Label maps must span the full canonical taxonomy vocabulary.
    assert sorted(ck["category_map"].keys()) == sorted(all_keys("package_types"))
    assert sorted(ck["material_map"].keys()) == sorted(all_keys("materials"))
    assert sorted(ck["finish_map"].keys()) == sorted(all_keys("finishes"))
    # Sanity vs module-level constants (used to size the heads).
    assert sorted(CATEGORY_KEYS) == sorted(all_keys("package_types"))
    assert len(CATEGORY_KEYS) == 10 and len(MATERIAL_KEYS) == 12 and len(FINISH_KEYS) == 11


def test_model_state_dict_loads_strictly():
    bundle = load_checkpoint()
    ck = bundle["checkpoint"]
    model = build_model(ck.get("model_config"))
    model.load_state_dict(ck["model_state_dict"], strict=True)
    n_params = sum(p.numel() for p in model.parameters())
    assert 1_000_000 <= n_params <= 3_000_000, n_params


# =============================================================================
# PREDICTIONS
# =============================================================================

def _assert_canonical(result: dict) -> None:
    """Every predicted attribute must already be a canonical taxonomy key."""
    for attr, canon_cat in (
        ("category", "package_types"),
        ("material", "materials"),
        ("finish", "finishes"),
    ):
        key = result[attr]
        assert isinstance(key, str) and key, f"{attr} missing from result: {result}"
        # normalize() of a canonical key is an identity mapping; a legacy key
        # (e.g. 'carton' or 'corrugated_box') would fail this check.
        assert normalize(key, canon_cat) == key, (
            f"{attr}={key!r} is not a canonical {canon_cat} key"
        )
    assert result["input_mode"] in {"sketch", "photo"}
    for attr, canon_cat in (
        ("category", "package_types"),
        ("material", "materials"),
        ("finish", "finishes"),
    ):
        conf = result[f"{attr}_confidence"]
        assert 0.0 <= conf <= 1.0
        probs = result["probabilities"][attr]
        assert abs(sum(probs.values()) - 1.0) < 1e-3
        # Probability keys are the full canonical vocabulary for that attribute.
        assert set(probs.keys()) == set(all_keys(canon_cat))


def test_predict_config_on_benchmark_sketch_returns_canonical_keys():
    img = Image.open(ROOT / "data/samples/carton_sketch.png")
    result = predict_config(img)
    _assert_canonical(result)
    # A lineart upload must be routed through the sketch input mode.
    assert result["input_mode"] == "sketch"


def test_predict_config_accepts_pil_image_directly_and_non_square():
    img = Image.open(ROOT / "data/samples/bottle_sketch.png").convert("RGB")
    result = predict_config(img)
    _assert_canonical(result)

    # Non-square PIL image exercises the center-crop preprocessing path.
    tall = ImageOps.fit(img, (300, 500))
    result2 = predict_config(tall)
    _assert_canonical(result2)


def test_all_benchmark_sketches_predict_under_10s_cpu():
    """Checkpoint load + predictions for all 3 benchmark sketches in < 10s on CPU."""
    start = time.perf_counter()
    results = [predict_config(Image.open(p)) for p in BENCHMARK_SKETCHES]
    elapsed = time.perf_counter() - start
    for result in results:
        _assert_canonical(result)
    assert elapsed < 10.0, f"prediction took {elapsed:.2f}s (> 10s budget)"


# =============================================================================
# CONSISTENCY
# =============================================================================

def test_predict_config_is_deterministic():
    img = Image.open(BENCHMARK_SKETCHES[0])
    r1 = predict_config(img)
    r2 = predict_config(img)
    for attr in ("category", "material", "finish", "input_mode"):
        assert r1[attr] == r2[attr]


@pytest.mark.parametrize(
    "sketch, expected_mode",
    [(BENCHMARK_SKETCHES[0], "sketch"), (BENCHMARK_SKETCHES[1], "sketch"),
     (BENCHMARK_SKETCHES[2], "sketch")],
)
def test_benchmark_sketches_detected_as_sketches(sketch: Path, expected_mode: str):
    result = predict_config(Image.open(sketch))
    assert result["input_mode"] == expected_mode
    _assert_canonical(result)
