"""
PackVision AI - Large-Scale Custom Packaging Dataset Generator
==============================================================
Generates a comprehensive, paired synthetic packaging dataset:
- Ground-truth photorealistic studio packaging renders
- Paired synthetic designer sketches (clean lineart, rough pencil, inverted edge maps)
- Structural pseudo-depth maps
- Structured vision-language captions and optical metadata
- Train / Validation splits with manifest JSONL
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prompt_engine import (
    PromptEngine,
    build_packaging_prompt,
    CATEGORIES,
    SUBSTRATES,
    FINISHES,
    LIGHTING_PRESETS,
    BACKGROUNDS,
)
from src.pipeline import ProceduralPackagingRenderer
from src.preprocessor import SketchPreprocessor


class PackagingDatasetGenerator:
    """Generates paired (sketch, render, depth, metadata) packaging datasets."""

    def __init__(self, output_root: str = "data/dataset"):
        self.output_root = Path(output_root)
        self.train_dir = self.output_root / "train"
        self.val_dir = self.output_root / "val"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)

        self.preprocessor = SketchPreprocessor(target_size=(512, 512))
        self.renderer = ProceduralPackagingRenderer()
        self.prompt_engine = PromptEngine()

        # Taxonomy attributes
        self.categories = ["carton", "box", "pouch", "bottle", "jar", "can", "tube"]
        self.materials = [
            "kraft_cardboard",
            "corrugated_box",
            "matte_carton",
            "coated_sbs",
            "flexible_pouch",
            "frosted_glass",
            "brushed_aluminum",
        ]
        self.finishes = [
            "gold_foil",
            "silver_foil",
            "spot_uv",
            "embossed",
            "debossed",
            "holographic",
            "matte",
        ]
        self.lighting_setups = ["studio_softbox", "dramatic_rim", "high_key_commercial", "natural_daylight"]
        self.backgrounds = ["seamless_studio", "warm_pedestal", "dark_luxury"]
        self.camera_angles = ["three_quarter", "front", "top_down"]

    def _draw_geometric_sketch(
        self,
        category: str,
        size: Tuple[int, int] = (512, 512),
        rng: random.Random = None,
    ) -> Image.Image:
        """Draws a clean, parameterized packaging sketch for the given category."""
        if rng is None:
            rng = random.Random()

        w, h = size
        img = Image.new("L", size, 255)
        draw = ImageDraw.Draw(img)

        # Apply slight random scale / aspect variation
        sx = rng.uniform(0.85, 1.15)
        sy = rng.uniform(0.85, 1.15)
        cx, cy = w // 2, int(h * 0.52)

        if category in ("carton", "box"):
            bw = int(w * 0.44 * sx)
            bh = int(h * 0.54 * sy)
            depth_x = int(bw * 0.32)
            depth_y = int(depth_x * 0.45)

            p1 = (cx - bw // 2, cy - bh // 2)
            p2 = (cx + bw // 2, cy - bh // 2)
            p3 = (cx + bw // 2, cy + bh // 2)
            p4 = (cx - bw // 2, cy + bh // 2)

            t1 = (p1[0] + depth_x, p1[1] - depth_y)
            t2 = (p2[0] + depth_x, p2[1] - depth_y)
            r3 = (p3[0] + depth_x, p3[1] - depth_y)

            # Front face
            draw.polygon([p1, p2, p3, p4], outline=0, width=3)
            # Top face
            draw.polygon([p1, t1, t2, p2], outline=0, width=3)
            # Right face
            draw.polygon([p2, t2, r3, p3], outline=0, width=3)
            # Creases / tuck flap
            flap_y = int(p1[1] + (p4[1] - p1[1]) * 0.22)
            draw.line([(p1[0], flap_y), (p2[0], flap_y)], fill=0, width=2)
            # Branding zone placeholder
            bx1, by1 = int(cx - bw * 0.28), int(cy - bh * 0.12)
            bx2, by2 = int(cx + bw * 0.28), int(cy + bh * 0.22)
            draw.rectangle([bx1, by1, bx2, by2], outline=60, width=1)

        elif category == "pouch":
            pw = int(w * 0.46 * sx)
            ph = int(h * 0.58 * sy)
            top_y = cy - ph // 2
            bot_y = cy + ph // 2
            left_x = cx - pw // 2
            right_x = cx + pw // 2

            # Stand-up pouch silhouette with bottom gusset & top seal
            top_seal_y = top_y + int(ph * 0.08)
            draw.line([(left_x + 10, top_y), (right_x - 10, top_y)], fill=0, width=3)
            draw.line([(left_x + 10, top_seal_y), (right_x - 10, top_seal_y)], fill=0, width=2)
            # Side contours
            draw.line([(left_x + 10, top_y), (left_x, bot_y - 25)], fill=0, width=3)
            draw.line([(right_x - 10, top_y), (right_x, bot_y - 25)], fill=0, width=3)
            # Bottom curved gusset
            gusset_pts = [
                (left_x, bot_y - 25),
                (cx - pw // 4, bot_y),
                (cx + pw // 4, bot_y),
                (right_x, bot_y - 25),
            ]
            draw.line(gusset_pts, fill=0, width=3)
            # Tear notches
            notch_y = top_seal_y + 12
            draw.line([(left_x + 2, notch_y), (left_x + 14, notch_y)], fill=0, width=2)
            draw.line([(right_x - 14, notch_y), (right_x - 2, notch_y)], fill=0, width=2)
            # Center label
            draw.ellipse([cx - pw // 4, cy - ph // 8, cx + pw // 4, cy + ph // 4], outline=70, width=1)

        elif category in ("bottle", "jar"):
            bw = int(w * 0.36 * sx)
            bh = int(h * 0.60 * sy) if category == "bottle" else int(h * 0.42 * sy)
            left_x = cx - bw // 2
            right_x = cx + bw // 2
            bot_y = cy + bh // 2
            shoulder_y = cy - int(bh * 0.22)
            neck_w = int(bw * 0.38)
            neck_h = int(bh * 0.24)
            cap_h = int(neck_h * 0.75)

            # Cap
            draw.rectangle([cx - neck_w // 2, shoulder_y - neck_h - cap_h, cx + neck_w // 2, shoulder_y - neck_h], outline=0, width=3)
            # Neck collar
            draw.rectangle([cx - neck_w // 2 + 3, shoulder_y - neck_h, cx + neck_w // 2 - 3, shoulder_y], outline=0, width=2)
            # Bottle body
            draw.line([(cx - neck_w // 2, shoulder_y), (left_x, shoulder_y + 20)], fill=0, width=3)
            draw.line([(cx + neck_w // 2, shoulder_y), (right_x, shoulder_y + 20)], fill=0, width=3)
            draw.line([(left_x, shoulder_y + 20), (left_x, bot_y)], fill=0, width=3)
            draw.line([(right_x, shoulder_y + 20), (right_x, bot_y)], fill=0, width=3)
            # Base curve
            draw.arc([left_x, bot_y - 12, right_x, bot_y + 12], 0, 180, fill=0, width=3)
            # Label outline
            lbl_top = shoulder_y + int(bh * 0.15)
            lbl_bot = bot_y - int(bh * 0.15)
            draw.rectangle([left_x + 12, lbl_top, right_x - 12, lbl_bot], outline=60, width=1)

        elif category in ("can", "tube"):
            cw = int(w * 0.38 * sx)
            ch = int(h * 0.62 * sy)
            left_x = cx - cw // 2
            right_x = cx + cw // 2
            top_y = cy - ch // 2
            bot_y = cy + ch // 2
            rim_h = 16

            # Top ellipse rim
            draw.ellipse([left_x, top_y, right_x, top_y + rim_h * 2], outline=0, width=3)
            # Sides
            draw.line([(left_x, top_y + rim_h), (left_x, bot_y)], fill=0, width=3)
            draw.line([(right_x, top_y + rim_h), (right_x, bot_y)], fill=0, width=3)
            # Bottom rim
            draw.arc([left_x, bot_y - rim_h, right_x, bot_y + rim_h], 0, 180, fill=0, width=3)
            # Label
            draw.rectangle([left_x + 8, top_y + int(ch * 0.2), right_x - 8, bot_y - int(ch * 0.18)], outline=70, width=1)

        # Add organic sketch imperfections (rough hand-drawn line jitter)
        sketch_np = np.array(img)
        noise = np.random.normal(0, rng.uniform(1.2, 3.5), sketch_np.shape)
        noisy = np.clip(sketch_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Apply subtle Gaussian blur and threshold to mimic graphite pencil texture
        blurred = cv2.GaussianBlur(noisy, (3, 3), 0.5)
        return Image.fromarray(blurred)

    def generate_sample(
        self,
        sample_id: str,
        category: str,
        material: str,
        finish: str,
        lighting: str,
        background: str,
        split: str = "train",
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Generates a complete paired sample: sketch, render, depth map, and metadata."""
        rng = random.Random(seed)
        np.random.seed(seed)

        target_dir = self.train_dir if split == "train" else self.val_dir

        # 1. Generate parameterized 512x512 sketch
        sketch_pil = self._draw_geometric_sketch(category, size=(512, 512), rng=rng)
        sketch_filename = f"{sample_id}_sketch.png"
        sketch_path = target_dir / sketch_filename
        sketch_pil.save(sketch_path, format="PNG")

        # 2. Build structured packaging prompt
        brand_names = ["Aura Botanical", "Nordic Pure", "Lumina Luxe", "Verde Craft", "Solstice Organics", "Kyoto Studio"]
        brand_name = rng.choice(brand_names)
        pos_prompt, neg_prompt = build_packaging_prompt(
            category=category,
            substrate=material,
            finish=finish,
            lighting=lighting,
            brand_name=brand_name,
            color_palette=rng.choice(["minimalist earth tones", "deep luxury noir", "matte pastel sage", "warm terracotta"]),
        )

        # 3. Preprocess sketch to get masks and Canny/depth representation
        preprocess_res = self.preprocessor.preprocess(sketch_pil)
        depth_filename = f"{sample_id}_depth.png"
        depth_path = target_dir / depth_filename
        preprocess_res.depth.save(depth_path, format="PNG")

        # 4. Render photorealistic packaging mockup via procedural PBR engine
        render_pil = self.renderer.render(
            preprocess_res=preprocess_res,
            prompt_pkg=self.prompt_engine.build_prompt(
                category=category,
                substrate=material,
                finish=finish,
                lighting=lighting,
                background=background,
                brand_name=brand_name,
            ),
        )
        # Resize to 512x512 matching dataset resolution
        render_pil = render_pil.resize((512, 512), Image.Resampling.LANCZOS)
        render_filename = f"{sample_id}_render.png"
        render_path = target_dir / render_filename
        render_pil.save(render_path, format="PNG")

        # 5. Compile structured metadata record
        record = {
            "sample_id": sample_id,
            "split": split,
            "category": category,
            "material": material,
            "finish": finish,
            "lighting": lighting,
            "background": background,
            "brand_name": brand_name,
            "sketch_file": str(sketch_path.relative_to(self.output_root)).replace("\\", "/"),
            "render_file": str(render_path.relative_to(self.output_root)).replace("\\", "/"),
            "depth_file": str(depth_path.relative_to(self.output_root)).replace("\\", "/"),
            "positive_prompt": pos_prompt,
            "negative_prompt": neg_prompt,
            "resolution": [512, 512],
            "seed": seed,
        }
        return record

    def build_dataset(
        self,
        total_samples: int = 150,
        val_ratio: float = 0.15,
        base_seed: int = 2026,
    ) -> List[Dict[str, Any]]:
        """Generates a full paired dataset and writes metadata.jsonl manifests."""
        print(f"Generating PackVision AI custom paired dataset ({total_samples} samples)...")
        records = []
        val_count = int(total_samples * val_ratio)
        train_count = total_samples - val_count

        # Assign splits deterministically
        splits = ["train"] * train_count + ["val"] * val_count
        rng_shuffle = random.Random(base_seed)
        rng_shuffle.shuffle(splits)

        start_time = time.time()
        for idx in range(total_samples):
            sample_id = f"pv_pkg_{idx + 1:04d}"
            split = splits[idx]
            seed = base_seed + idx * 37

            # Sample attributes with balanced distribution
            category = self.categories[idx % len(self.categories)]
            material = self.materials[(idx // len(self.categories)) % len(self.materials)]
            finish = self.finishes[(idx // (len(self.categories) * len(self.materials))) % len(self.finishes)]
            lighting = self.lighting_setups[idx % len(self.lighting_setups)]
            background = self.backgrounds[idx % len(self.backgrounds)]

            rec = self.generate_sample(
                sample_id=sample_id,
                category=category,
                material=material,
                finish=finish,
                lighting=lighting,
                background=background,
                split=split,
                seed=seed,
            )
            records.append(rec)

            if (idx + 1) % 25 == 0 or idx == total_samples - 1:
                elapsed = time.time() - start_time
                print(f"  Generated {idx + 1}/{total_samples} samples ({elapsed:.1f}s)")

        # Write overall metadata.jsonl and split manifests
        meta_file = self.output_root / "metadata.jsonl"
        train_file = self.output_root / "train_manifest.jsonl"
        val_file = self.output_root / "val_manifest.jsonl"

        with open(meta_file, "w", encoding="utf-8") as f_all, \
             open(train_file, "w", encoding="utf-8") as f_train, \
             open(val_file, "w", encoding="utf-8") as f_val:
            for r in records:
                line = json.dumps(r) + "\n"
                f_all.write(line)
                if r["split"] == "train":
                    f_train.write(line)
                else:
                    f_val.write(line)

        print(f"Dataset generation complete!")
        print(f"  Total: {len(records)} samples ({train_count} train, {val_count} val)")
        print(f"  Manifest: {meta_file}")
        return records


if __name__ == "__main__":
    generator = PackagingDatasetGenerator(output_root="data/dataset")
    generator.build_dataset(total_samples=160, val_ratio=0.15, base_seed=42)
