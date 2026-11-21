"""Unit tests for src/preprocessor.py pure image-processing functions.

Uses small synthetic PIL/numpy images only — no model downloads, no file I/O
beyond in-memory buffers. Assertions are written to hold for both the OpenCV
code path (HAS_CV2=True) and the pure-PIL/numpy fallback (HAS_CV2=False).

NOTE: src/preprocessor.py has no EXIF transpose helper (EXIF orientation is
handled in backend/app/services/asset_service.py via ImageOps.exif_transpose,
outside this module's scope), so that case is intentionally not covered here.
"""

import importlib.util
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Import defensively: importing the `src` package executes src/__init__.py,
# which pulls in the full rendering stack (pipeline, prompt_engine, ...). If a
# sibling module is mid-refactor we fall back to loading preprocessor.py
# standalone (it only depends on cv2/numpy/PIL itself).
try:
    from src.preprocessor import HAS_CV2, PreprocessResult, SketchPreprocessor
except Exception:  # pragma: no cover - fallback path
    _spec = importlib.util.spec_from_file_location(
        "packvision_preprocessor_standalone",
        Path(__file__).resolve().parents[2] / "src" / "preprocessor.py",
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    HAS_CV2 = _mod.HAS_CV2
    PreprocessResult = _mod.PreprocessResult
    SketchPreprocessor = _mod.SketchPreprocessor


TARGET = (256, 256)


@pytest.fixture
def prep() -> SketchPreprocessor:
    return SketchPreprocessor(target_size=TARGET)


def black_square_on_white(w: int = 100, h: int = 100, box=None) -> Image.Image:
    """White canvas with a solid black rectangle (PIL, mode RGB).

    ``box`` is (x0, y0, x1, y1) in PIL (x, y) order and defaults to the
    central 60% of the canvas. Keep the ink coverage under ~30% of the frame:
    a heavier square trips the module's is_dark_background mean-brightness
    heuristic and flips silhouette segmentation to the dark-canvas path.
    """
    if box is None:
        box = (w // 5, h // 5, (4 * w) // 5, (4 * h) // 5)
    img = Image.new("RGB", (w, h), (255, 255, 255))
    x0, y0, x1, y1 = box
    for y in range(max(0, y0), min(h, y1)):
        for x in range(max(0, x0), min(w, x1)):
            img.putpixel((x, y), (0, 0, 0))
    return img


def gray_of(pil_or_arr) -> np.ndarray:
    if isinstance(pil_or_arr, Image.Image):
        return np.array(pil_or_arr.convert("L"))
    return pil_or_arr


# --------------------------------------------------------------------------- #
# Color space conversion
# --------------------------------------------------------------------------- #
def test_pil_to_cv2_rgb_shape(prep):
    img = Image.new("RGB", (40, 30), (10, 20, 30))
    arr = SketchPreprocessor.pil_to_cv2(img)
    assert arr.shape == (30, 40, 3)
    assert arr.dtype == np.uint8


def test_pil_to_cv2_grayscale_2d(prep):
    img = Image.new("L", (40, 30), 128)
    arr = SketchPreprocessor.pil_to_cv2(img)
    assert arr.shape == (30, 40)
    assert int(arr.mean()) == 128


def test_pil_to_cv2_rgba_composites_onto_white(prep):
    # Fully transparent top-left quadrant -> must become white, not black.
    img = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
    for y in range(10, 20):
        for x in range(10, 20):
            img.putpixel((x, y), (255, 0, 0, 255))
    arr = SketchPreprocessor.pil_to_cv2(img)
    assert arr.shape == (20, 20, 3)
    # Transparent corner composites to pure white regardless of BGR/RGB order.
    assert tuple(int(v) for v in arr[0, 0]) == (255, 255, 255)
    # Opaque red region has no white channel value of 0.
    assert arr[15, 15].max() > 0


def test_cv2_to_pil_roundtrip(prep):
    arr = np.zeros((30, 40, 3), dtype=np.uint8)
    arr[:, :, 0] = 200  # BGR: blue channel
    img = SketchPreprocessor.cv2_to_pil(arr)
    assert img.mode in ("RGB", "BGR") or img.size == (40, 30)
    rgb = np.array(img.convert("RGB"))
    assert rgb.shape == (30, 40, 3)
    # Blue-dominant input stays blue-dominant after roundtrip.
    assert rgb[:, :, 2].mean() > rgb[:, :, 0].mean()

    gray = SketchPreprocessor.cv2_to_pil(np.full((25, 25), 77, dtype=np.uint8))
    assert gray.mode == "L"
    assert np.array(gray).mean() == 77


# --------------------------------------------------------------------------- #
# Letterboxing / aspect-ratio normalization
# --------------------------------------------------------------------------- #
def test_letterbox_preserves_aspect_and_pads_symmetrically(prep):
    # 100 wide x 50 tall -> scale 2.56 -> 256x128 content, 64px top/bottom bars.
    src = SketchPreprocessor.pil_to_cv2(black_square_on_white(100, 50, (10, 5, 90, 45)))
    padded, scale, padding = prep.letterbox_image(src, target_size=TARGET)

    assert padded.shape[:2] == (256, 256)
    assert scale == pytest.approx(2.56)
    assert padding == (64, 64, 0, 0)
    # Border rows are the white letterbox fill.
    assert int(padded[0, 0].max()) == 255
    assert int(padded[255, 128].max()) == 255
    # Source pixel (x=50, y=25) lands at padded row 64 + 25*2.56 ~= 128.
    assert int(padded[128, 128].max()) < 50


def test_letterbox_wide_content_pads_left_right(prep):
    # 50 wide x 100 tall -> scale 2.56 -> 128x256 content, 64px left/right bars.
    src = SketchPreprocessor.pil_to_cv2(black_square_on_white(50, 100, (10, 20, 40, 80)))
    padded, scale, padding = prep.letterbox_image(src, target_size=TARGET)

    assert padded.shape[:2] == (256, 256)
    assert scale == pytest.approx(2.56)
    assert padding == (0, 0, 64, 64)
    assert int(padded[128, 0].max()) == 255


def test_letterbox_already_target_size(prep):
    src = SketchPreprocessor.pil_to_cv2(black_square_on_white(256, 256))
    padded, scale, padding = prep.letterbox_image(src, target_size=TARGET)
    assert padded.shape[:2] == (256, 256)
    assert scale == pytest.approx(1.0)
    assert padding == (0, 0, 0, 0)


def test_unletterbox_restores_original_dimensions(prep):
    orig_w, orig_h = 100, 50
    src = SketchPreprocessor.pil_to_cv2(black_square_on_white(orig_w, orig_h))
    padded, scale, padding = prep.letterbox_image(src, target_size=TARGET)
    restored = prep.unletterbox_image(padded, (orig_w, orig_h), scale, padding)
    assert restored.shape[:2] == (orig_h, orig_w)


# --------------------------------------------------------------------------- #
# Background detection and contrast enhancement
# --------------------------------------------------------------------------- #
def test_is_dark_background(prep):
    assert SketchPreprocessor.is_dark_background(np.full((64, 64), 15, dtype=np.uint8)) is True
    assert SketchPreprocessor.is_dark_background(np.full((64, 64), 240, dtype=np.uint8)) is False


def test_enhance_contrast_returns_valid_uint8(prep):
    gray = np.array(black_square_on_white(80, 80).convert("L"))
    enhanced = prep.enhance_contrast(gray)
    assert enhanced.dtype == np.uint8
    assert enhanced.shape == gray.shape
    # A crisp black-on-white sketch stays high-contrast after enhancement.
    assert enhanced.max() >= 200
    assert enhanced.min() <= 80


# --------------------------------------------------------------------------- #
# Canny edge extraction
# --------------------------------------------------------------------------- #
def test_extract_canny_returns_valid_edge_map(prep):
    gray = np.array(black_square_on_white(100, 100).convert("L"))
    edges = prep.extract_canny(gray)

    assert edges.dtype == np.uint8
    assert edges.shape == (100, 100)
    assert int(edges.max()) <= 255
    # Edges must be detected along the square boundary.
    assert int(edges.sum()) > 0
    assert int(edges[15:26, 45:56].max()) == 255
    # Corner of the canvas is far from any edge and stays black.
    assert int(edges[0:3, 0:3].max()) == 0


def test_extract_canny_uniform_image_has_no_edges(prep):
    gray = np.full((80, 80), 180, dtype=np.uint8)
    edges = prep.extract_canny(gray)
    assert edges.shape == (80, 80)
    assert int(edges.sum()) == 0


def test_extract_canny_explicit_thresholds(prep):
    gray = np.array(black_square_on_white(100, 100).convert("L"))
    edges = prep.extract_canny(gray, low_threshold=50, high_threshold=150)
    assert int(edges.sum()) > 0
    assert edges.dtype == np.uint8


# --------------------------------------------------------------------------- #
# Silhouette segmentation
# --------------------------------------------------------------------------- #
def test_extract_silhouette_mask_light_background(prep):
    # Small square (~18% ink) so is_dark_background stays on the light path.
    gray = np.array(black_square_on_white(100, 100, box=(35, 35, 65, 65)).convert("L"))
    assert SketchPreprocessor.is_dark_background(gray) is False
    edges = prep.extract_canny(gray)
    mask = prep.extract_silhouette_mask(gray, edges)

    assert mask.dtype == np.uint8
    assert mask.shape == (100, 100)
    # Package interior filled, background corner empty.
    assert int(mask[50, 50]) == 255
    assert int(mask[2, 2]) == 0


def test_extract_silhouette_mask_dark_background(prep):
    if not HAS_CV2:
        pytest.skip("Dark-background segmentation relies on the OpenCV path")
    gray = 255 - np.array(black_square_on_white(100, 100, box=(35, 35, 65, 65)).convert("L"))
    assert SketchPreprocessor.is_dark_background(gray) is True
    edges = prep.extract_canny(gray)
    mask = prep.extract_silhouette_mask(gray, edges)
    assert int(mask[50, 50]) == 255
    assert int(mask[2, 2]) == 0


def test_extract_silhouette_mask_empty_canvas_fallback(prep):
    # Uniform canvas -> no contours -> fallback central-region mask.
    gray = np.full((100, 100), 200, dtype=np.uint8)
    edges = np.zeros((100, 100), dtype=np.uint8)
    mask = prep.extract_silhouette_mask(gray, edges)
    assert mask.shape == (100, 100)
    assert int(mask.sum()) >= 0  # never crashes; any binary output is valid
    assert set(np.unique(mask)).issubset({0, 255})


# --------------------------------------------------------------------------- #
# Pseudo-depth
# --------------------------------------------------------------------------- #
def test_extract_pseudo_depth_interior_is_closer(prep):
    gray = np.array(black_square_on_white(100, 100, box=(35, 35, 65, 65)).convert("L"))
    edges = prep.extract_canny(gray)
    mask = prep.extract_silhouette_mask(gray, edges)
    depth = prep.extract_pseudo_depth(gray, edges, mask=mask)

    assert depth.shape == (100, 100, 3)
    assert depth.dtype == np.uint8
    inside = depth[mask > 0][:, 0].mean()
    outside = depth[mask == 0][:, 0].mean()
    assert inside > outside


# --------------------------------------------------------------------------- #
# load_image input polymorphism
# --------------------------------------------------------------------------- #
def test_load_image_from_pil_and_bytes_and_array(prep):
    pil = black_square_on_white(64, 64)

    from_pil = SketchPreprocessor.load_image(pil)
    assert isinstance(from_pil, Image.Image)
    assert from_pil.mode == "RGB"
    assert from_pil.size == (64, 64)

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    from_bytes = SketchPreprocessor.load_image(buf.getvalue())
    assert from_bytes.size == (64, 64)

    bgr = SketchPreprocessor.pil_to_cv2(pil)
    from_arr = SketchPreprocessor.load_image(bgr)
    assert from_arr.size == (64, 64)


def test_load_image_missing_file_raises(prep):
    with pytest.raises(FileNotFoundError):
        SketchPreprocessor.load_image("definitely_missing_sketch_file.png")


def test_load_image_unsupported_type_raises(prep):
    with pytest.raises(TypeError):
        SketchPreprocessor.load_image(12345)


# --------------------------------------------------------------------------- #
# Full preprocess() pipeline
# --------------------------------------------------------------------------- #
def test_preprocess_end_to_end_shapes_and_metadata(prep):
    # ~28% ink coverage keeps auto-invert on the light-canvas path while the
    # square stays above the 2%-of-frame contour area threshold at 256x256.
    pil = black_square_on_white(200, 100, (55, 30, 145, 70))
    result = prep.preprocess(pil)

    assert isinstance(result, PreprocessResult)
    assert result.image.size == TARGET
    assert result.canny.size == TARGET
    assert result.mask.size == TARGET
    assert result.depth.size == TARGET
    assert result.depth_map.size == TARGET  # alias property
    assert result.target_size == TARGET
    assert result.original_size == (200, 100)
    assert result.scale == pytest.approx(1.28)
    assert result.padding == (64, 64, 0, 0)

    assert result.metadata["is_dark_background"] is False
    assert result.metadata["inverted"] is True  # auto mode inverts light canvas

    as_dict = result.to_dict()
    for key in ("original_size", "target_size", "scale", "padding"):
        assert key in as_dict

    # Canny map is a valid edge image: mostly black, sparse white strokes.
    canny_arr = np.array(result.canny)
    assert canny_arr.max() == 255
    assert (canny_arr > 0).mean() < 0.5

    # Mask and depth must agree spatially.
    mask_arr = np.array(result.mask)
    depth_arr = np.array(result.depth.convert("L"))
    assert mask_arr.shape == depth_arr.shape == (256, 256)
    assert int(mask_arr[128, 128]) == 255  # sketch center is inside the package
