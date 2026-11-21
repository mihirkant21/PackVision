"""
Test script for the production FacetAware3DSegmenter from src.pipeline.

Runs segmentation on all three sample sketches (carton, bottle, pouch) and
prints per-facet geometry stats and classification counts.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.pipeline import FacetAware3DSegmenter  # noqa: E402
from src.preprocessor import SketchPreprocessor  # noqa: E402

SAMPLES = [
    ("carton", ROOT / "data" / "samples" / "carton_sketch.png"),
    ("bottle", ROOT / "data" / "samples" / "bottle_sketch.png"),
    ("pouch", ROOT / "data" / "samples" / "pouch_sketch.png"),
]


def main() -> None:
    segmenter = FacetAware3DSegmenter(target_size=(1024, 1024))
    prep = SketchPreprocessor(target_size=(1024, 1024))

    for category, sketch_path in SAMPLES:
        if not sketch_path.exists():
            print(f"[skip] {sketch_path} not found")
            continue

        print(f"\n=== {category}: {sketch_path.name} ===")
        res = prep.preprocess(str(sketch_path))
        mask = np.array(res.mask)
        canny = np.array(res.canny)

        seg = segmenter.segment(mask, canny, category)
        print(f"bbox={seg.package_bbox} is_cylindrical={seg.is_cylindrical} "
              f"contact_pts={len(seg.bottom_contact_pts)}")

        counts: dict = {}
        for f in seg.facets:
            counts[f.facet_type] = counts.get(f.facet_type, 0) + 1
            print(f" Facet {f.id}: type={f.facet_type:<12} area={f.area:<7} "
                  f"rel=({f.rel_x:.2f},{f.rel_y:.2f}) aspect={f.aspect_ratio:.2f} "
                  f"mult={f.light_multiplier:.2f} norm={f.normal} nested={f.is_nested}")
        print(f" Classification counts: {counts}")


if __name__ == "__main__":
    main()
