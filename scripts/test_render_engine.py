"""
Test script for the PackVision AI procedural render engine.

Imports the production FacetAware3DSegmenter / FacetAware3DShader /
ProceduralPackagingRenderer from src.pipeline (no duplicated logic) and:
1. Segments the sample carton sketch, printing facet classification counts.
2. Times each render stage (preprocess, contact shadow, substrate, lighting,
   finish, full composite).
3. Saves the final render as a PNG under storage/outputs/.

Run with the project venv:
    .venv/Scripts/python.exe scripts/test_render_engine.py
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from src.pipeline import (  # noqa: E402
    FacetAware3DShader,
    FacetAware3DSegmenter,
    ProceduralPackagingRenderer,
)
from src.preprocessor import SketchPreprocessor  # noqa: E402
from src.prompt_engine import PromptEngine  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the PackVision procedural render engine.")
    parser.add_argument(
        "--sketch",
        default=str(ROOT / "data" / "samples" / "carton_sketch.png"),
        help="Input sketch image (default: data/samples/carton_sketch.png)",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "storage" / "outputs" / "test_render_carton.png"),
        help="Output PNG path (default: storage/outputs/test_render_carton.png)",
    )
    parser.add_argument("--category", default="carton", help="Packaging category (default: carton)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # 1. Preprocess -----------------------------------------------------------
    t0 = time.perf_counter()
    prep = SketchPreprocessor(target_size=(1024, 1024))
    res = prep.preprocess(args.sketch)
    t_preprocess = time.perf_counter() - t0

    mask_np = np.array(res.mask)
    canny_np = np.array(res.canny)
    h, w = mask_np.shape[:2]

    # 2. Segment facets -------------------------------------------------------
    segmenter = FacetAware3DSegmenter(target_size=(1024, 1024))
    t0 = time.perf_counter()
    seg = segmenter.segment(mask_np, canny_np, args.category)
    t_segment = time.perf_counter() - t0

    counts: dict = {}
    for f in seg.facets:
        counts[f.facet_type] = counts.get(f.facet_type, 0) + 1
    print(f"Segmented {len(seg.facets)} facets "
          f"(cylindrical={seg.is_cylindrical}, bbox={seg.package_bbox}):")
    for f in seg.facets:
        print(f" - Facet {f.id}: type={f.facet_type:<12} area={f.area:<7} "
              f"mult={f.light_multiplier:.2f} norm={f.normal}")
    print(f"Facet classification counts: {counts}")

    # 3. Time each shader stage ----------------------------------------------
    shader = FacetAware3DShader(target_size=(1024, 1024))

    t0 = time.perf_counter()
    y = np.linspace(248, 230, h)[:, None]
    x = np.linspace(-1, 1, w)[None, :]
    vignette = (1.0 - 0.08 * (x ** 2))
    bg = (y * vignette)[:, :, None].repeat(3, axis=-1).astype(np.uint8)
    t_background = time.perf_counter() - t0

    t0 = time.perf_counter()
    comp_bg = shader.render_contact_shadow(bg, mask_np, seg)
    t_shadow = time.perf_counter() - t0

    t0 = time.perf_counter()
    sub_img = shader.render_substrate("kraft paperboard", mask_np, seg)
    t_substrate = time.perf_counter() - t0

    t0 = time.perf_counter()
    lit_img = shader.apply_3d_lighting(sub_img, seg, "studio softbox")
    t_lighting = time.perf_counter() - t0

    t0 = time.perf_counter()
    finished_img = shader.apply_finish(lit_img, seg, "gold foil")
    t_finish = time.perf_counter() - t0

    t0 = time.perf_counter()
    mask_3ch = np.stack([mask_np > 128] * 3, axis=-1)
    final_bgr = np.where(mask_3ch, finished_img, comp_bg)
    grain = np.random.normal(0, 2.0, (h, w, 3)).astype(np.float32)
    final_bgr = np.clip(final_bgr.astype(np.float32) + grain, 0.0, 255.0).astype(np.uint8)
    final_bgr = cv2.bilateralFilter(final_bgr, d=3, sigmaColor=12, sigmaSpace=12)
    t_composite = time.perf_counter() - t0

    print("\nStage timings:")
    print(f" - preprocess:       {t_preprocess * 1000:8.1f} ms")
    print(f" - segmentation:     {t_segment * 1000:8.1f} ms")
    print(f" - background:       {t_background * 1000:8.1f} ms")
    print(f" - contact shadow:   {t_shadow * 1000:8.1f} ms")
    print(f" - substrate:        {t_substrate * 1000:8.1f} ms")
    print(f" - 3D lighting:      {t_lighting * 1000:8.1f} ms")
    print(f" - finish:           {t_finish * 1000:8.1f} ms")
    print(f" - composite/grain:  {t_composite * 1000:8.1f} ms")

    # 4. Full production render via ProceduralPackagingRenderer ---------------
    prompt_pkg = PromptEngine().build_prompt(
        category=args.category,
        substrate="kraft paperboard",
        finish="gold foil",
        lighting="studio softbox",
    )
    renderer = ProceduralPackagingRenderer(target_size=(1024, 1024))
    t0 = time.perf_counter()
    final_image = renderer.render(res, prompt_pkg, seed=args.seed)
    t_full = time.perf_counter() - t0
    print(f" - full render:      {t_full * 1000:8.1f} ms  "
          f"(ProceduralPackagingRenderer, size={final_image.size[0]}x{final_image.size[1]})")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_image.save(out_path, quality=95)
    print(f"3D carton render successfully saved to {out_path}")


if __name__ == "__main__":
    main()
