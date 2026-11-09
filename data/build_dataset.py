"""
PackVision AI - Dataset Builder & HuggingFace Metadata Creator
==============================================================
Builds paired (sketch, photoreal_render) datasets for ControlNet and LoRA fine-tuning.
Provides:
1. Automated inverse sketch extraction from raw packaging renders/photos.
2. Synthetic paired sample generation.
3. HuggingFace compatible metadata.jsonl creator.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessor import SketchPreprocessor
from src.prompt_engine import PromptEngine, build_packaging_prompt


def extract_inverse_sketch(
    image: Image.Image,
    style: str = "pencil",  # 'pencil', 'canny', or 'hybrid'
) -> Image.Image:
    """
    Extract an authentic packaging sketch from a photorealistic render or photo.
    
    Uses Color Dodge blend with inverted Gaussian blur for pencil sketches,
    or adaptive Canny thresholding for structural die-lines.
    """
    img_bgr = SketchPreprocessor.pil_to_cv2(image)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if style == "pencil":
        # Realistic pencil sketch using Color Dodge inversion
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), sigmaX=0, sigmaY=0)
        # Dodge blend: (image / (255 - blur)) * 255
        dodge = cv2.divide(gray, 255 - blur, scale=256)
        # Contrast adjustment to crisp up pencil line strokes
        sketch = cv2.multiply(dodge, dodge, scale=1.0 / 255.0)
        return Image.fromarray(sketch, mode="L")

    elif style == "canny":
        # Die-cut vector style line art (black lines on white canvas)
        edges = cv2.Canny(gray, 60, 160)
        # Invert so lines are black on white paper
        inv_edges = cv2.bitwise_not(edges)
        return Image.fromarray(inv_edges, mode="L")

    else:  # 'hybrid'
        # Combines pencil shading with structural Canny creases
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (15, 15), 0)
        dodge = cv2.divide(gray, 255 - blur, scale=256)
        edges = cv2.Canny(gray, 70, 180)
        hybrid = cv2.bitwise_and(dodge, cv2.bitwise_not(edges))
        return Image.fromarray(hybrid, mode="L")


def generate_synthetic_paired_dataset(
    output_dir: Path,
    num_samples: int = 6,
) -> List[Dict[str, str]]:
    """
    Generate synthetic raw renders, extract paired sketches, and build dataset entries.
    """
    from src.pipeline import PackVisionRenderer

    renderer = PackVisionRenderer(use_mock_inference=True)

    raw_renders_dir = output_dir / "raw_renders"
    sketches_dir = output_dir / "extracted_sketches"
    raw_renders_dir.mkdir(parents=True, exist_ok=True)
    sketches_dir.mkdir(parents=True, exist_ok=True)

    sample_configs = [
        ("carton", "kraft", "gold foil", "Tea Folding Carton", "sage green botanical pattern"),
        ("pouch", "flexible pouch", "spot UV", "Stand-Up Coffee Pouch", "matte black with bronze"),
        ("bottle", "frosted glass", "silver foil", "Cosmetic Dropper Bottle", "minimalist white and gold"),
        ("box", "matte carton", "embossed", "Luxury Rigid Perfume Box", "midnight navy blue"),
        ("can", "brushed aluminum", "matte", "Craft Soda Can", "vibrant citrus orange"),
        ("tube", "coated SBS", "debossed", "Hand Cream Squeeze Tube", "pastel blush pink"),
    ]

    dataset_entries: List[Dict[str, str]] = []

    print(f"[DatasetBuilder] Synthesizing {min(num_samples, len(sample_configs))} paired samples...")

    for idx, (cat, sub, fin, brand, colors) in enumerate(sample_configs[:num_samples]):
        sample_id = f"pkg_{idx + 1:03d}_{cat}"
        print(f"  -> Generating sample {idx + 1}/{num_samples}: {cat} ({sub}, {fin})...")

        # 1. Generate clean geometric baseline sketch canvas
        sketch_canvas = np.full((1024, 1024), 255, dtype=np.uint8)
        cx, cy = 512, 512

        if cat == "carton":
            # 3D folding carton lines
            cv2.rectangle(sketch_canvas, (320, 260), (704, 760), 0, 3)
            cv2.line(sketch_canvas, (320, 380), (704, 380), 0, 2)  # Tuck fold
            cv2.line(sketch_canvas, (450, 260), (450, 760), 0, 2)  # Front panel crease
            cv2.line(sketch_canvas, (320, 260), (370, 210), 0, 2)  # Top flap angle
            cv2.line(sketch_canvas, (704, 260), (754, 210), 0, 2)
            cv2.line(sketch_canvas, (370, 210), (754, 210), 0, 2)
        elif cat == "pouch":
            # Doypack contour
            cv2.rectangle(sketch_canvas, (330, 280), (694, 780), 0, 3)
            cv2.ellipse(sketch_canvas, (512, 780), (182, 35), 0, 0, 180, 0, 3)  # Gusset bottom
            cv2.line(sketch_canvas, (330, 340), (694, 340), 0, 2)  # Zip line
            cv2.circle(sketch_canvas, (350, 320), 4, 0, -1)  # Tear notch
            cv2.circle(sketch_canvas, (674, 320), 4, 0, -1)
        elif cat == "bottle":
            # Cylinder bottle with dropper cap
            cv2.rectangle(sketch_canvas, (380, 420), (644, 820), 0, 3)  # Body
            cv2.line(sketch_canvas, (380, 420), (440, 340), 0, 3)  # Left shoulder
            cv2.line(sketch_canvas, (644, 420), (584, 340), 0, 3)  # Right shoulder
            cv2.rectangle(sketch_canvas, (440, 260), (584, 340), 0, 3)  # Neck & Cap
            cv2.ellipse(sketch_canvas, (512, 230), (35, 30), 0, 0, 360, 0, 3)  # Pipette bulb
        else:
            # Generic packaging rectangular prism
            cv2.rectangle(sketch_canvas, (340, 300), (684, 760), 0, 3)
            cv2.line(sketch_canvas, (340, 300), (420, 220), 0, 2)
            cv2.line(sketch_canvas, (684, 300), (764, 220), 0, 2)
            cv2.line(sketch_canvas, (420, 220), (764, 220), 0, 2)

        input_sketch = Image.fromarray(sketch_canvas, mode="L")

        # 2. Render photoreal image via pipeline
        render_res = renderer.render(
            sketch=input_sketch,
            category=cat,
            substrate=sub,
            finish=fin,
            brand_name=brand,
            color_palette=colors,
            seed=42 + idx,
        )

        render_path = raw_renders_dir / f"{sample_id}_render.png"
        sketch_path = sketches_dir / f"{sample_id}_sketch.png"

        render_res.image.save(render_path, quality=95)

        # 3. Extract inverse sketch pairing
        extracted_sketch = extract_inverse_sketch(render_res.image, style="pencil")
        extracted_sketch.save(sketch_path, quality=95)

        entry = {
            "sample_id": sample_id,
            "image": str(render_path.relative_to(output_dir.parent)),
            "conditioning_image": str(sketch_path.relative_to(output_dir.parent)),
            "text": render_res.positive_prompt,
            "category": cat,
            "substrate": sub,
            "finish": fin,
            "brand": brand,
        }
        dataset_entries.append(entry)

    return dataset_entries


def build_metadata_jsonl(
    entries: List[Dict[str, str]],
    output_path: Path,
) -> None:
    """Write HuggingFace formatted metadata.jsonl file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    print(f"[DatasetBuilder] Wrote {len(entries)} entries to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PackVision AI Dataset Builder")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw_renders",
        help="Directory with raw renders/photos",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/extracted_sketches",
        help="Directory for extracted conditioning sketches",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="data/metadata.jsonl",
        help="Output HuggingFace metadata.jsonl file",
    )
    parser.add_argument(
        "--generate_synthetic_samples",
        type=int,
        default=6,
        help="Number of synthetic paired samples to generate if raw directory is empty",
    )
    parser.add_argument(
        "--sketch_style",
        type=str,
        default="pencil",
        choices=["pencil", "canny", "hybrid"],
        help="Algorithm for sketch extraction",
    )

    args = parser.parse_args()

    input_p = Path(args.input_dir)
    output_p = Path(args.output_dir)
    meta_p = Path(args.metadata_file)

    input_p.mkdir(parents=True, exist_ok=True)
    output_p.mkdir(parents=True, exist_ok=True)

    raw_images = list(input_p.glob("*.png")) + list(input_p.glob("*.jpg")) + list(input_p.glob("*.jpeg"))

    entries: List[Dict[str, str]] = []

    if not raw_images:
        print("[DatasetBuilder] No raw images found. Generating synthetic dataset...")
        entries = generate_synthetic_paired_dataset(
            output_dir=input_p.parent,
            num_samples=args.generate_synthetic_samples,
        )
    else:
        print(f"[DatasetBuilder] Processing {len(raw_images)} raw images from {input_p}...")
        for img_p in raw_images:
            sample_id = img_p.stem
            render_img = Image.open(img_p).convert("RGB")
            sketch_img = extract_inverse_sketch(render_img, style=args.sketch_style)
            sketch_filename = f"{sample_id}_sketch.png"
            sketch_out = output_p / sketch_filename
            sketch_img.save(sketch_out)

            pos_prompt, _ = build_packaging_prompt()

            entry = {
                "sample_id": sample_id,
                "image": str(img_p),
                "conditioning_image": str(sketch_out),
                "text": pos_prompt,
                "category": "carton",
                "substrate": "coated SBS",
                "finish": "matte",
            }
            entries.append(entry)

    build_metadata_jsonl(entries, meta_p)
    print("[DatasetBuilder] Dataset processing complete!")


if __name__ == "__main__":
    main()
