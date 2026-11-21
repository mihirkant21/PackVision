"""Unit tests for photo-vs-sketch input mode detection."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.photo_mode import detect_input_mode

ROOT = Path(__file__).resolve().parents[1]


def _make_sketch_image() -> Image.Image:
    img = Image.new("RGB", (512, 512), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.line([(150, 120), (360, 120), (360, 400), (150, 400), (150, 120)], fill=(20, 20, 20), width=4)
    d.rectangle([(200, 200), (310, 330)], outline=(60, 60, 60), width=2)
    return img


def _make_photo_image() -> Image.Image:
    rng = np.random.default_rng(7)
    base = rng.integers(90, 220, size=(512, 512, 3), dtype=np.uint8)
    # colourful product block on a warm backdrop with texture noise
    base[120:400, 150:360] = (178, 132, 84)
    base[200:330, 200:310] = (222, 180, 60)
    img = Image.fromarray(base.astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.text((210, 250), "TEA", fill=(20, 20, 20))
    return img


def test_sample_sketch_detected_as_sketch():
    img = Image.open(ROOT / "data/samples/carton_sketch.png")
    assert detect_input_mode(img) == "sketch"


def test_photoreal_render_detected_as_photo():
    # A real photoreal output (saved render from the FLUX chain, downloaded
    # from the DB if present) or a synthetic textured fallback — the point is
    # a photo-like image must NOT be classified as a sketch.
    artifact = ROOT / "storage/outputs/probe_hf_black-forest-labs_FLUX.1-Kontext-dev.png"
    if artifact.exists():
        img = Image.open(artifact)
    else:
        rng = np.random.default_rng(3)
        base = rng.integers(120, 230, size=(512, 512, 3), dtype=np.uint8)
        base[100:420, 130:400] = (196, 158, 116)  # kraft-toned box body
        img = Image.fromarray(base)
    assert detect_input_mode(img) == "photo"


def test_synthetic_line_drawing_is_sketch():
    assert detect_input_mode(_make_sketch_image()) == "sketch"


def test_synthetic_colourful_textured_image_is_photo():
    assert detect_input_mode(_make_photo_image()) == "photo"


def test_numpy_array_input_supported():
    arr = np.asarray(_make_photo_image())
    assert detect_input_mode(arr) == "photo"
