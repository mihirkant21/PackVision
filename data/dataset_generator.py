"""
PackVision AI - Large-Scale Custom Packaging Dataset Generator
==============================================================
Generates a comprehensive, paired synthetic packaging dataset:
- Ground-truth photorealistic studio packaging renders with multi-facet 3D PBR shading
- Paired synthetic designer sketches (clean lineart, tuck creases, gussets, crimps, pull-tabs)
- Structural 3D pseudo-depth maps
- Structured vision-language captions and optical metadata
- Train / Validation splits with manifest JSONL
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from src.preprocessor import SketchPreprocessor


# Prompt Engine Taxonomy Mappings
SUBSTRATE_TO_PROMPT = {
    "kraft_cardboard": "kraft",
    "corrugated_box": "corrugated",
    "matte_carton": "matte carton",
    "coated_sbs": "coated SBS",
    "flexible_pouch": "flexible pouch",
    "frosted_glass": "frosted glass",
    "brushed_aluminum": "brushed aluminum",
}

FINISH_TO_PROMPT = {
    "gold_foil": "gold foil",
    "silver_foil": "silver foil",
    "spot_uv": "spot UV",
    "embossed": "embossed",
    "debossed": "debossed",
    "holographic": "holographic",
    "matte": "matte",
}

LIGHTING_TO_PROMPT = {
    "studio_softbox": "studio softbox",
    "dramatic_rim": "dramatic rim",
    "high_key_commercial": "high-key commercial",
    "natural_daylight": "natural daylight",
}

BACKGROUND_TO_PROMPT = {
    "seamless_studio": "seamless studio",
    "warm_pedestal": "warm pedestal",
    "dark_luxury": "dark luxury",
}


class FacetPBRRenderer:
    """
    Studio-grade 3D facet-aware PBR renderer for ground-truth packaging generation.
    Renders packaging with distinct facet normal lighting, realistic physical materials,
    crisp specular bevels, contact ambient occlusion shadows, and luxury embellishments.
    """

    def __init__(self, size: Tuple[int, int] = (512, 512)) -> None:
        self.w, self.h = size

    def render_background(self, bg_type: str) -> np.ndarray:
        """Render studio cyclorama, stone pedestal, or dark luxury backdrop."""
        w, h = self.w, self.h
        canvas = np.zeros((h, w, 3), dtype=np.float32)

        if bg_type in ("dark_luxury", "dark luxury"):
            y = np.linspace(24, 38, h)[:, None]
            canvas[:, :] = y[:, :, None]
            noise = np.random.normal(0, 2.5, (h, w, 3))
            canvas = np.clip(canvas + noise, 18, 45)
        elif bg_type in ("warm_pedestal", "warm pedestal"):
            y_wall = np.linspace(235, 215, h)[:, None]
            canvas[:, :] = y_wall[:, :, None]
            canvas[:, :, 0] = np.clip(canvas[:, :, 0] - 8, 0, 255)
            canvas[:, :, 2] = np.clip(canvas[:, :, 2] + 4, 0, 255)
            slab_y = int(h * 0.76)
            canvas[slab_y:, :, :] = (195, 205, 215)
            cv2.line(canvas, (0, slab_y), (w, slab_y), (245, 248, 250), 2)
        else:  # seamless_studio
            y = np.linspace(248, 228, h)[:, None]
            x = np.linspace(-1, 1, w)[None, :]
            vignette = 1.0 - 0.08 * (x ** 2)
            canvas[:, :] = (y * vignette)[:, :, None]

        return canvas

    def generate_substrate_texture(self, material: str) -> np.ndarray:
        """Synthesize physical packaging material texture."""
        w, h = self.w, self.h
        sub = material.lower().replace(" ", "_")
        canvas = np.zeros((h, w, 3), dtype=np.float32)

        if "kraft" in sub:
            base = np.array([102, 154, 194], dtype=np.float32)  # BGR Kraft Brown
            canvas[:, :] = base
            noise = np.random.normal(0, 10, (h, w, 3))
            fibers = cv2.GaussianBlur(np.random.normal(0, 16, (h, w, 3)), (3, 7), 1.2)
            flecks = (np.random.random((h, w)) > 0.993).astype(np.float32) * -35.0
            canvas = canvas + noise + fibers + flecks[:, :, None]
        elif "corrugated" in sub:
            base = np.array([95, 142, 182], dtype=np.float32)
            canvas[:, :] = base
            y_coords = np.arange(h)[:, None]
            flute_ridges = np.sin(y_coords * 0.12) * 14.0
            canvas = canvas + flute_ridges[:, :, None] + np.random.normal(0, 8, (h, w, 3))
        elif "matte_carton" in sub or ("matte" in sub and "carton" in sub):
            base = np.array([68, 58, 48], dtype=np.float32)  # Refined dark navy/slate
            canvas[:, :] = base
            canvas += np.random.normal(0, 3.5, (h, w, 3))
        elif "flexible_pouch" in sub:
            base = np.array([180, 182, 186], dtype=np.float32)
            canvas[:, :] = base
            x_coords = np.arange(w)[None, :]
            brush_waves = np.sin(x_coords * 0.05) * 12.0
            canvas += brush_waves[:, :, None] + np.random.normal(0, 5, (h, w, 3))
        elif "frosted_glass" in sub:
            base = np.array([225, 230, 232], dtype=np.float32)
            canvas[:, :] = base
            glass_noise = cv2.GaussianBlur(np.random.normal(0, 10, (h, w, 3)), (15, 15), 5.0)
            canvas += glass_noise
        elif "brushed_aluminum" in sub:
            base = np.array([216, 218, 220], dtype=np.float32)
            canvas[:, :] = base
            streaks = np.random.normal(0, 15, (h, 1, 3))
            canvas += np.tile(streaks, (1, w, 1))
        else:  # coated_sbs
            base = np.array([242, 244, 246], dtype=np.float32)
            canvas[:, :] = base
            canvas += np.random.normal(0, 2.0, (h, w, 3))

        return np.clip(canvas, 0, 255)

    def apply_finish_to_panel(
        self,
        canvas: np.ndarray,
        panel_mask: np.ndarray,
        finish: str,
        lighting: str,
    ) -> np.ndarray:
        """Apply luxury hot stamping foil, spot UV, or embossing onto label panel."""
        fin = finish.lower().replace(" ", "_")
        h, w = canvas.shape[:2]

        if "gold_foil" in fin:
            gold_color = np.array([30, 200, 242], dtype=np.float32)
            canvas = np.where(panel_mask[:, :, None] > 0.5, gold_color * 1.15, canvas)
        elif "silver_foil" in fin:
            silver_color = np.array([245, 246, 252], dtype=np.float32)
            canvas = np.where(panel_mask[:, :, None] > 0.5, silver_color * 1.05, canvas)
        elif "spot_uv" in fin:
            uv_gloss = np.zeros((h, w), dtype=np.float32)
            cv2.circle(uv_gloss, (int(w * 0.45), int(h * 0.4)), int(w * 0.15), 1.0, -1)
            uv_gloss = cv2.GaussianBlur(uv_gloss, (45, 45), 15.0)
            canvas = np.where(panel_mask[:, :, None] > 0.5, canvas + uv_gloss[:, :, None] * 55.0, canvas)
        elif "embossed" in fin:
            edge = cv2.Canny((panel_mask * 255).astype(np.uint8), 50, 150)
            shift_hl = np.roll(edge, (-2, -2), axis=(0, 1)).astype(np.float32) / 255.0
            shift_sh = np.roll(edge, (2, 2), axis=(0, 1)).astype(np.float32) / 255.0
            canvas += (shift_hl[:, :, None] * 45.0) - (shift_sh[:, :, None] * 35.0)
        elif "debossed" in fin:
            edge = cv2.Canny((panel_mask * 255).astype(np.uint8), 50, 150)
            shift_sh = np.roll(edge, (-2, -2), axis=(0, 1)).astype(np.float32) / 255.0
            shift_hl = np.roll(edge, (2, 2), axis=(0, 1)).astype(np.float32) / 255.0
            canvas += (shift_hl[:, :, None] * 35.0) - (shift_sh[:, :, None] * 45.0)
        elif "holographic" in fin:
            x_mesh = np.linspace(0, 10 * np.pi, w)[None, :]
            rb_b = (np.sin(x_mesh) * 0.5 + 0.5) * 60.0
            rb_g = (np.sin(x_mesh + 2.0) * 0.5 + 0.5) * 60.0
            rb_r = (np.sin(x_mesh + 4.0) * 0.5 + 0.5) * 60.0
            rainbow = np.dstack([rb_b, rb_g, rb_r]).repeat(h, axis=0)
            canvas = np.where(panel_mask[:, :, None] > 0.5, canvas * 0.55 + rainbow * 0.85, canvas)
        else:  # matte
            canvas = np.where(panel_mask[:, :, None] > 0.5, canvas * 0.92, canvas)

        return np.clip(canvas, 0, 255)


class PackagingDatasetGenerator:
    """Generates paired (sketch, render, depth, metadata) packaging datasets with 3D facet shading."""

    def __init__(self, output_root: str = "data/dataset"):
        self.output_root = Path(output_root)
        self.train_dir = self.output_root / "train"
        self.val_dir = self.output_root / "val"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)

        self.size = (512, 512)
        self.facet_renderer = FacetPBRRenderer(size=self.size)
        self.prompt_engine = PromptEngine()

        # Taxonomy attributes - 7 categories, 7 substrates, 7 finishes
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

    def _get_geometry(self, category: str, rng: random.Random) -> Dict[str, Any]:
        """Calculates parameterized 3D geometry parameters for the category."""
        w, h = self.size
        cx, cy = w // 2, int(h * 0.52)
        sx = rng.uniform(0.92, 1.08)
        sy = rng.uniform(0.92, 1.08)

        geom = {"category": category, "cx": cx, "cy": cy, "sx": sx, "sy": sy}

        if category in ("carton", "box"):
            bw_left = int(w * 0.22 * sx)
            bw_right = int(w * 0.20 * sx)
            bh = int(h * 0.48 * sy)
            dy_left = int(bw_left * 0.40)
            dy_right = int(bw_right * 0.38)

            geom.update({
                "P_front_top": (cx, cy - bh // 2),
                "P_front_bot": (cx, cy + bh // 2),
                "P_left_top": (cx - bw_left, cy - bh // 2 - dy_left),
                "P_left_bot": (cx - bw_left, cy + bh // 2 - dy_left),
                "P_right_top": (cx + bw_right, cy - bh // 2 - dy_right),
                "P_right_bot": (cx + bw_right, cy + bh // 2 - dy_right),
                "P_back_top": (cx - bw_left + bw_right, cy - bh // 2 - dy_left - dy_right),
                "bh": bh,
                "bw_left": bw_left,
                "bw_right": bw_right,
            })
        elif category == "pouch":
            pw = int(w * 0.44 * sx)
            ph = int(h * 0.56 * sy)
            top_y = cy - ph // 2
            bot_y = cy + ph // 2
            left_x = cx - pw // 2
            right_x = cx + pw // 2
            seal_h = 26
            geom.update({
                "pw": pw, "ph": ph, "top_y": top_y, "bot_y": bot_y,
                "left_x": left_x, "right_x": right_x, "seal_h": seal_h,
            })
        elif category == "bottle":
            bw = int(w * 0.36 * sx)
            bh = int(h * 0.60 * sy)
            left_x = cx - bw // 2
            right_x = cx + bw // 2
            bot_y = cy + bh // 2
            shoulder_y = cy - int(bh * 0.20)
            neck_w = int(bw * 0.36)
            neck_h = int(bh * 0.22)
            cap_h = int(neck_h * 0.70)
            cap_y = shoulder_y - neck_h - cap_h
            geom.update({
                "bw": bw, "bh": bh, "left_x": left_x, "right_x": right_x,
                "bot_y": bot_y, "shoulder_y": shoulder_y, "neck_w": neck_w,
                "neck_h": neck_h, "cap_h": cap_h, "cap_y": cap_y,
            })
        elif category == "jar":
            jw = int(w * 0.52 * sx)
            jh = int(h * 0.44 * sy)
            left_x = cx - jw // 2
            right_x = cx + jw // 2
            bot_y = cy + jh // 2
            top_y = cy - jh // 2
            cap_h = int(jh * 0.28)
            cap_w = int(jw * 0.94)
            geom.update({
                "jw": jw, "jh": jh, "left_x": left_x, "right_x": right_x,
                "bot_y": bot_y, "top_y": top_y, "cap_h": cap_h, "cap_w": cap_w,
            })
        elif category == "can":
            cw = int(w * 0.38 * sx)
            ch = int(h * 0.62 * sy)
            left_x = cx - cw // 2
            right_x = cx + cw // 2
            top_y = cy - ch // 2
            bot_y = cy + ch // 2
            chime_w = int(cw * 0.88)
            geom.update({
                "cw": cw, "ch": ch, "left_x": left_x, "right_x": right_x,
                "top_y": top_y, "bot_y": bot_y, "chime_w": chime_w,
            })
        elif category == "tube":
            tw = int(w * 0.38 * sx)
            th = int(h * 0.60 * sy)
            top_y = cy - th // 2
            bot_y = cy + th // 2
            crimp_w = int(tw * 1.15)
            crimp_h = 24
            geom.update({
                "tw": tw, "th": th, "top_y": top_y, "bot_y": bot_y,
                "crimp_w": crimp_w, "crimp_h": crimp_h,
            })

        return geom

    def _draw_geometric_sketch(self, geom: Dict[str, Any], rng: random.Random) -> Image.Image:
        """Draws an authentic packaging designer sketch with graphite lineart."""
        w, h = self.size
        img = Image.new("L", (w, h), 255)
        draw = ImageDraw.Draw(img)
        cat = geom["category"]
        cx, cy = geom["cx"], geom["cy"]

        if cat in ("carton", "box"):
            P_front_top = geom["P_front_top"]
            P_front_bot = geom["P_front_bot"]
            P_left_top = geom["P_left_top"]
            P_left_bot = geom["P_left_bot"]
            P_right_top = geom["P_right_top"]
            P_right_bot = geom["P_right_bot"]
            P_back_top = geom["P_back_top"]
            bh = geom["bh"]

            # Left face
            draw.polygon([P_left_top, P_front_top, P_front_bot, P_left_bot], outline=0, width=3)
            # Right face
            draw.polygon([P_front_top, P_right_top, P_right_bot, P_front_bot], outline=0, width=3)
            # Top face (Top Flap / Lid)
            draw.polygon([P_left_top, P_back_top, P_right_top, P_front_top], outline=0, width=3)

            if cat == "carton":
                # Tuck flap crease across top face
                draw.line([P_left_top, P_right_top], fill=0, width=2)
                # Dust flap score lines
                draw.line([(P_left_top[0] + 12, P_left_top[1] - 4), (P_front_top[0] - 12, P_front_top[1] - 4)], fill=40, width=1)
                # Left face tuck crease line near top
                crease_y_offset = int(bh * 0.12)
                draw.line([(P_left_top[0], P_left_top[1] + crease_y_offset),
                           (P_front_top[0], P_front_top[1] + crease_y_offset)], fill=0, width=2)
                # Label panel on left face
                lbl_tl = (int(P_left_top[0] * 0.25 + P_front_top[0] * 0.75), int(P_left_top[1] + bh * 0.26))
                lbl_tr = (int(P_left_top[0] * 0.05 + P_front_top[0] * 0.95), int(P_front_top[1] + bh * 0.26))
                lbl_br = (lbl_tr[0], int(lbl_tr[1] + bh * 0.36))
                lbl_bl = (lbl_tl[0], int(lbl_tl[1] + bh * 0.36))
                draw.polygon([lbl_tl, lbl_tr, lbl_br, lbl_bl], outline=60, width=1)
            else:  # "box"
                # Rigid setup box lid seam
                lid_h = int(bh * 0.28)
                lid_l1 = (P_left_top[0], P_left_top[1] + lid_h)
                lid_l2 = (P_front_top[0], P_front_top[1] + lid_h)
                lid_r = (P_right_top[0], P_right_top[1] + lid_h)
                draw.line([lid_l1, lid_l2], fill=0, width=2)
                draw.line([lid_l2, lid_r], fill=0, width=2)
                # Bevel double edge indicating board caliper
                draw.line([(P_front_top[0] - 3, P_front_top[1] + 2), (P_front_bot[0] - 3, P_front_bot[1] - 2)], fill=70, width=1)
                # Embossed label panel frame on lid
                lbl_tl = (int(P_left_top[0] * 0.3 + P_front_top[0] * 0.7), int(P_left_top[1] + lid_h * 0.25))
                lbl_tr = (int(P_left_top[0] * 0.05 + P_front_top[0] * 0.95), int(P_front_top[1] + lid_h * 0.25))
                lbl_br = (lbl_tr[0], int(lbl_tr[1] + lid_h * 0.55))
                lbl_bl = (lbl_tl[0], int(lbl_tl[1] + lid_h * 0.55))
                draw.polygon([lbl_tl, lbl_tr, lbl_br, lbl_bl], outline=60, width=1)

        elif cat == "pouch":
            pw, ph = geom["pw"], geom["ph"]
            top_y, bot_y = geom["top_y"], geom["bot_y"]
            left_x, right_x = geom["left_x"], geom["right_x"]
            seal_h = geom["seal_h"]

            # Top heat seal
            draw.rectangle([left_x, top_y, right_x, top_y + seal_h], outline=0, width=3)
            # Crimp lines across heat seal
            for kx in range(left_x + 8, right_x - 6, 6):
                draw.line([(kx, top_y + 2), (kx, top_y + seal_h - 2)], fill=70, width=1)
            # Tear notches
            notch_y = top_y + seal_h + 10
            draw.polygon([(left_x, notch_y - 4), (left_x + 8, notch_y), (left_x, notch_y + 4)], fill=0)
            draw.polygon([(right_x, notch_y - 4), (right_x - 8, notch_y), (right_x, notch_y + 4)], fill=0)
            # Side gusset folds
            draw.line([(left_x, top_y + seal_h), (left_x + 16, bot_y - 28)], fill=50, width=2)
            draw.line([(right_x, top_y + seal_h), (right_x - 16, bot_y - 28)], fill=50, width=2)
            # Outer contours
            draw.line([(left_x, top_y + seal_h), (left_x, bot_y - 25)], fill=0, width=3)
            draw.line([(right_x, top_y + seal_h), (right_x, bot_y - 25)], fill=0, width=3)
            # Stand-up bottom curved gusset
            gusset_pts = [
                (left_x, bot_y - 25),
                (cx - pw // 4, bot_y),
                (cx + pw // 4, bot_y),
                (right_x, bot_y - 25),
            ]
            draw.line(gusset_pts, fill=0, width=3)
            draw.arc([left_x + 10, bot_y - 38, right_x - 10, bot_y - 8], 0, 180, fill=60, width=2)
            # Central branding zone
            draw.ellipse([cx - pw // 3, cy - ph // 6, cx + pw // 3, cy + ph // 4], outline=60, width=1)

        elif cat == "bottle":
            bw, bh = geom["bw"], geom["bh"]
            left_x, right_x = geom["left_x"], geom["right_x"]
            bot_y, shoulder_y = geom["bot_y"], geom["shoulder_y"]
            neck_w, neck_h = geom["neck_w"], geom["neck_h"]
            cap_h, cap_y = geom["cap_h"], geom["cap_y"]

            # Cap
            draw.ellipse([cx - neck_w // 2, cap_y - 4, cx + neck_w // 2, cap_y + 4], outline=0, width=2)
            draw.rectangle([cx - neck_w // 2, cap_y, cx + neck_w // 2, cap_y + cap_h], outline=0, width=3)
            for kx in range(cx - neck_w // 2 + 5, cx + neck_w // 2 - 4, 4):
                draw.line([(kx, cap_y + 2), (kx, cap_y + cap_h - 2)], fill=80, width=1)
            # Neck & Collar
            draw.rectangle([cx - neck_w // 2 + 4, cap_y + cap_h, cx + neck_w // 2 - 4, shoulder_y], outline=0, width=2)
            draw.line([(cx - neck_w // 2 + 4, shoulder_y - 6), (cx + neck_w // 2 - 4, shoulder_y - 6)], fill=60, width=1)
            # Shoulder
            draw.line([(cx - neck_w // 2, shoulder_y), (left_x, shoulder_y + 22)], fill=0, width=3)
            draw.line([(cx + neck_w // 2, shoulder_y), (right_x, shoulder_y + 22)], fill=0, width=3)
            # Body sides
            draw.line([(left_x, shoulder_y + 22), (left_x, bot_y)], fill=0, width=3)
            draw.line([(right_x, shoulder_y + 22), (right_x, bot_y)], fill=0, width=3)
            # Base rim
            draw.arc([left_x, bot_y - 14, right_x, bot_y + 14], 0, 180, fill=0, width=3)
            draw.arc([left_x + 8, bot_y - 20, right_x - 8, bot_y], 0, 180, fill=80, width=1)
            # Wraparound label
            lbl_top = shoulder_y + int(bh * 0.16)
            lbl_bot = bot_y - int(bh * 0.14)
            draw.arc([left_x + 6, lbl_top - 8, right_x - 6, lbl_top + 8], 0, 180, fill=60, width=2)
            draw.arc([left_x + 6, lbl_bot - 8, right_x - 6, lbl_bot + 8], 0, 180, fill=60, width=2)
            draw.line([(left_x + 6, lbl_top), (left_x + 6, lbl_bot)], fill=60, width=1)
            draw.line([(right_x - 6, lbl_top), (right_x - 6, lbl_bot)], fill=60, width=1)

        elif cat == "jar":
            jw, jh = geom["jw"], geom["jh"]
            left_x, right_x = geom["left_x"], geom["right_x"]
            bot_y, top_y = geom["bot_y"], geom["top_y"]
            cap_h, cap_w = geom["cap_h"], geom["cap_w"]
            cap_lx = cx - cap_w // 2
            cap_rx = cx + cap_w // 2

            # Cap
            draw.ellipse([cap_lx, top_y, cap_rx, top_y + 16], outline=0, width=3)
            draw.line([(cap_lx, top_y + 8), (cap_lx, top_y + cap_h)], fill=0, width=3)
            draw.line([(cap_rx, top_y + 8), (cap_rx, top_y + cap_h)], fill=0, width=3)
            draw.arc([cap_lx, top_y + cap_h - 10, cap_rx, top_y + cap_h + 10], 0, 180, fill=0, width=3)
            for kx in range(cap_lx + 8, cap_rx - 6, 6):
                draw.line([(kx, top_y + 12), (kx, top_y + cap_h)], fill=80, width=1)
            draw.line([(cap_lx + 6, top_y + cap_h + 4), (cap_rx - 6, top_y + cap_h + 4)], fill=60, width=1)
            # Jar body
            draw.line([(left_x, top_y + cap_h + 8), (left_x, bot_y)], fill=0, width=3)
            draw.line([(right_x, top_y + cap_h + 8), (right_x, bot_y)], fill=0, width=3)
            draw.arc([left_x, bot_y - 14, right_x, bot_y + 14], 0, 180, fill=0, width=3)
            draw.arc([left_x + 10, bot_y - 32, right_x - 10, bot_y - 8], 0, 180, fill=70, width=2)
            # Front label
            lbl_top = top_y + cap_h + int(jh * 0.16)
            lbl_bot = bot_y - int(jh * 0.22)
            draw.rectangle([left_x + 16, lbl_top, right_x - 16, lbl_bot], outline=60, width=1)

        elif cat == "can":
            cw, ch = geom["cw"], geom["ch"]
            left_x, right_x = geom["left_x"], geom["right_x"]
            top_y, bot_y = geom["top_y"], geom["bot_y"]
            chime_w = geom["chime_w"]
            chime_lx = cx - chime_w // 2
            chime_rx = cx + chime_w // 2

            # Top chime outer rim
            draw.ellipse([chime_lx, top_y, chime_rx, top_y + 18], outline=0, width=3)
            # Inner countersink rim
            inner_lx = cx - int(chime_w * 0.78) // 2
            inner_rx = cx + int(chime_w * 0.78) // 2
            draw.ellipse([inner_lx, top_y + 4, inner_rx, top_y + 14], outline=60, width=1)
            # Pull tab
            draw.ellipse([cx - 5, top_y + 8, cx + 5, top_y + 12], outline=0, width=2)
            draw.line([(cx, top_y + 12), (cx + 14, top_y + 11)], fill=0, width=2)
            draw.ellipse([cx + 10, top_y + 7, cx + 22, top_y + 15], outline=0, width=2)
            # Upper neck taper
            draw.line([(chime_lx, top_y + 10), (left_x, top_y + 32)], fill=0, width=3)
            draw.line([(chime_rx, top_y + 10), (right_x, top_y + 32)], fill=0, width=3)
            # Can body
            draw.line([(left_x, top_y + 32), (left_x, bot_y - 20)], fill=0, width=3)
            draw.line([(right_x, top_y + 32), (right_x, bot_y - 20)], fill=0, width=3)
            # Bottom neck taper
            draw.line([(left_x, bot_y - 20), (chime_lx + 4, bot_y)], fill=0, width=3)
            draw.line([(right_x, bot_y - 20), (chime_rx - 4, bot_y)], fill=0, width=3)
            draw.arc([chime_lx + 4, bot_y - 12, chime_rx - 4, bot_y + 12], 0, 180, fill=0, width=3)
            # Body label zone
            draw.arc([left_x, top_y + 36, right_x, top_y + 54], 0, 180, fill=70, width=1)
            draw.arc([left_x, bot_y - 34, right_x, bot_y - 16], 0, 180, fill=70, width=1)

        elif cat == "tube":
            tw, th = geom["tw"], geom["th"]
            top_y, bot_y = geom["top_y"], geom["bot_y"]
            crimp_w, crimp_h = geom["crimp_w"], geom["crimp_h"]
            crimp_lx = cx - crimp_w // 2
            crimp_rx = cx + crimp_w // 2
            cap_w = int(tw * 0.65)
            cap_lx = cx - cap_w // 2
            cap_rx = cx + cap_w // 2

            # Top crimped seam
            draw.rectangle([crimp_lx, top_y, crimp_rx, top_y + crimp_h], outline=0, width=3)
            for kx in range(crimp_lx + 6, crimp_rx - 4, 4):
                draw.line([(kx, top_y + 2), (kx, top_y + crimp_h - 2)], fill=70, width=1)
            # Tube body
            draw.line([(crimp_lx, top_y + crimp_h), (cx - tw // 2, bot_y - 45)], fill=0, width=3)
            draw.line([(crimp_rx, top_y + crimp_h), (cx + tw // 2, bot_y - 45)], fill=0, width=3)
            # Shoulder taper
            draw.line([(cx - tw // 2, bot_y - 45), (cap_lx, bot_y - 32)], fill=0, width=3)
            draw.line([(cx + tw // 2, bot_y - 45), (cap_rx, bot_y - 32)], fill=0, width=3)
            # Flip cap
            draw.rectangle([cap_lx, bot_y - 32, cap_rx, bot_y], outline=0, width=3)
            draw.line([(cap_lx, bot_y - 16), (cap_rx, bot_y - 16)], fill=0, width=2)
            draw.rectangle([cx - 8, bot_y - 20, cx + 8, bot_y - 12], outline=0, width=1)
            draw.arc([cap_lx, bot_y - 10, cap_rx, bot_y + 10], 0, 180, fill=0, width=2)
            # Center branding zone
            draw.rectangle([cx - int(tw * 0.32), top_y + crimp_h + 30, cx + int(tw * 0.32), bot_y - 65], outline=60, width=1)

        # Apply graphite pencil texture
        sketch_np = np.array(img)
        noise = np.random.normal(0, rng.uniform(1.2, 2.8), sketch_np.shape)
        noisy = np.clip(sketch_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(noisy, (3, 3), 0.5)
        return Image.fromarray(blurred)

    def _render_ground_truth(
        self,
        geom: Dict[str, Any],
        material: str,
        finish: str,
        lighting: str,
        background: str,
        rng: random.Random,
    ) -> Image.Image:
        """Renders photorealistic ground-truth image with 3D multi-facet PBR lighting."""
        w, h = self.size
        cat = geom["category"]
        cx, cy = geom["cx"], geom["cy"]

        # 1. Background
        bg_canvas = self.facet_renderer.render_background(background)

        # 2. Substrate procedural texture
        sub_canvas = self.facet_renderer.generate_substrate_texture(material)

        # Lighting multipliers
        is_rim = "dramatic" in lighting
        is_highkey = "high_key" in lighting
        is_daylight = "daylight" in lighting

        final_canvas = bg_canvas.copy()

        if cat in ("carton", "box"):
            P_front_top = geom["P_front_top"]
            P_front_bot = geom["P_front_bot"]
            P_left_top = geom["P_left_top"]
            P_left_bot = geom["P_left_bot"]
            P_right_top = geom["P_right_top"]
            P_right_bot = geom["P_right_bot"]
            P_back_top = geom["P_back_top"]
            bh = geom["bh"]

            # Ground contact shadow
            shadow_mask = np.zeros((h, w), dtype=np.float32)
            bot_poly = np.array([
                P_left_bot, P_front_bot, P_right_bot,
                (cx + 20, P_right_bot[1] + 28), (cx - 20, P_left_bot[1] + 28)
            ], dtype=np.int32)
            cv2.fillPoly(shadow_mask, [bot_poly], 1.0)
            soft_sh = cv2.GaussianBlur(shadow_mask, (45, 25), 12) * 0.55
            seam_sh = cv2.GaussianBlur(shadow_mask, (11, 7), 3) * 0.45
            final_canvas = final_canvas * (1.0 - np.clip(soft_sh + seam_sh, 0, 0.75)[:, :, None])

            # Facet Masks
            mask_top = np.zeros((h, w), dtype=np.float32)
            cv2.fillPoly(mask_top, [np.array([P_left_top, P_back_top, P_right_top, P_front_top], dtype=np.int32)], 1.0)

            mask_left = np.zeros((h, w), dtype=np.float32)
            cv2.fillPoly(mask_left, [np.array([P_left_top, P_front_top, P_front_bot, P_left_bot], dtype=np.int32)], 1.0)

            mask_right = np.zeros((h, w), dtype=np.float32)
            cv2.fillPoly(mask_right, [np.array([P_front_top, P_right_top, P_right_bot, P_front_bot], dtype=np.int32)], 1.0)

            # Facet Lighting
            lt_top = 1.35 if is_highkey else (1.28 if not is_rim else 0.85)
            lt_left = 1.12 if is_highkey else (1.05 if not is_rim else 0.70)
            lt_right = 0.75 if is_highkey else (0.62 if not is_rim else 0.45)

            final_canvas = np.where(mask_top[:, :, None] > 0.5, np.clip(sub_canvas * lt_top, 0, 255), final_canvas)
            final_canvas = np.where(mask_left[:, :, None] > 0.5, np.clip(sub_canvas * lt_left, 0, 255), final_canvas)
            final_canvas = np.where(mask_right[:, :, None] > 0.5, np.clip(sub_canvas * lt_right, 0, 255), final_canvas)

            # Bevel highlights & crease lines
            cv2.line(final_canvas, P_front_top, P_front_bot, (255, 255, 255), 2)
            cv2.line(final_canvas, P_left_top, P_front_top, (255, 255, 255), 2)
            cv2.line(final_canvas, P_front_top, P_right_top, (180, 180, 180), 1)

            if cat == "carton":
                crease_y = int(bh * 0.12)
                cv2.line(final_canvas, (P_left_top[0], P_left_top[1] + crease_y), (P_front_top[0], P_front_top[1] + crease_y), (60, 80, 95), 2)
                # Label panel on left face
                lbl_tl = (int(P_left_top[0] * 0.25 + P_front_top[0] * 0.75), int(P_left_top[1] + bh * 0.26))
                lbl_tr = (int(P_left_top[0] * 0.05 + P_front_top[0] * 0.95), int(P_front_top[1] + bh * 0.26))
                lbl_br = (lbl_tr[0], int(lbl_tr[1] + bh * 0.36))
                lbl_bl = (lbl_tl[0], int(lbl_tl[1] + bh * 0.36))
                mask_lbl = np.zeros((h, w), dtype=np.float32)
                cv2.fillPoly(mask_lbl, [np.array([lbl_tl, lbl_tr, lbl_br, lbl_bl], dtype=np.int32)], 1.0)
                final_canvas = self.facet_renderer.apply_finish_to_panel(final_canvas, mask_lbl, finish, lighting)
            else:  # box
                lid_h = int(bh * 0.28)
                cv2.line(final_canvas, (P_left_top[0], P_left_top[1] + lid_h), (P_front_top[0], P_front_top[1] + lid_h), (50, 60, 75), 2)
                cv2.line(final_canvas, (P_front_top[0], P_front_top[1] + lid_h), (P_right_top[0], P_right_top[1] + lid_h), (40, 50, 60), 2)
                lbl_tl = (int(P_left_top[0] * 0.3 + P_front_top[0] * 0.7), int(P_left_top[1] + lid_h * 0.25))
                lbl_tr = (int(P_left_top[0] * 0.05 + P_front_top[0] * 0.95), int(P_front_top[1] + lid_h * 0.25))
                lbl_br = (lbl_tr[0], int(lbl_tr[1] + lid_h * 0.55))
                lbl_bl = (lbl_tl[0], int(lbl_tl[1] + lid_h * 0.55))
                mask_lbl = np.zeros((h, w), dtype=np.float32)
                cv2.fillPoly(mask_lbl, [np.array([lbl_tl, lbl_tr, lbl_br, lbl_bl], dtype=np.int32)], 1.0)
                final_canvas = self.facet_renderer.apply_finish_to_panel(final_canvas, mask_lbl, finish, lighting)

        elif cat == "pouch":
            pw, ph = geom["pw"], geom["ph"]
            top_y, bot_y = geom["top_y"], geom["bot_y"]
            left_x, right_x = geom["left_x"], geom["right_x"]
            seal_h = geom["seal_h"]

            # Floor shadow
            sh_mask = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(sh_mask, (cx, bot_y + 8), (int(pw * 0.52), int(pw * 0.12)), 0, 0, 360, 1.0, -1)
            final_canvas *= (1.0 - cv2.GaussianBlur(sh_mask, (35, 19), 8)[:, :, None] * 0.70)

            # Volumetric 3D pillow shading across pouch body
            pouch_body_mask = np.zeros((h, w), dtype=np.float32)
            body_poly = np.array([
                (left_x, top_y + seal_h), (right_x, top_y + seal_h),
                (right_x, bot_y - 25), (cx + pw // 4, bot_y),
                (cx - pw // 4, bot_y), (left_x, bot_y - 25)
            ], dtype=np.int32)
            cv2.fillPoly(pouch_body_mask, [body_poly], 1.0)

            x_coords = np.arange(w)[None, :]
            u = np.clip((x_coords - cx) / (pw / 2.0), -1.0, 1.0)
            nz = np.sqrt(np.maximum(0.01, 1.0 - (u * 0.85) ** 2))
            diffuse = np.maximum(0.0, u * (-0.55) + nz * 0.65)
            light_map = 0.28 + diffuse * 0.80
            specular = np.exp(-((u + 0.30) ** 2) / 0.02) * 65.0
            rim = np.power(np.abs(u), 4) * 40.0

            shaded_pouch = sub_canvas * light_map[:, :, None] + specular[:, :, None] + rim[:, :, None]
            final_canvas = np.where(pouch_body_mask[:, :, None] > 0.5, np.clip(shaded_pouch, 0, 255), final_canvas)

            # Top heat seal
            seal_mask = np.zeros((h, w), dtype=np.float32)
            cv2.rectangle(seal_mask, (left_x, top_y), (right_x, top_y + seal_h), 1.0, -1)
            seal_shaded = sub_canvas * 0.95
            for kx in range(left_x + 8, right_x - 6, 6):
                cv2.line(seal_shaded, (kx, top_y + 2), (kx, top_y + seal_h - 2), (140, 140, 145), 1)
            final_canvas = np.where(seal_mask[:, :, None] > 0.5, seal_shaded, final_canvas)

            # Central label with finish
            mask_lbl = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(mask_lbl, (cx, cy), (int(pw * 0.28), int(ph * 0.18)), 0, 0, 360, 1.0, -1)
            final_canvas = self.facet_renderer.apply_finish_to_panel(final_canvas, mask_lbl, finish, lighting)

        elif cat in ("bottle", "jar"):
            is_jar = (cat == "jar")
            bw = geom["jw"] if is_jar else geom["bw"]
            bh = geom["jh"] if is_jar else geom["bh"]
            left_x, right_x = geom["left_x"], geom["right_x"]
            bot_y = geom["bot_y"]
            top_y = geom["top_y"] if is_jar else geom["shoulder_y"]
            cap_h = geom["cap_h"]

            # Floor shadow
            sh_mask = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(sh_mask, (cx, bot_y + 6), (int(bw * 0.52), int(bw * 0.12)), 0, 0, 360, 1.0, -1)
            final_canvas *= (1.0 - cv2.GaussianBlur(sh_mask, (35, 19), 8)[:, :, None] * 0.72)

            # Cylindrical body
            body_mask = np.zeros((h, w), dtype=np.float32)
            y_start = top_y + cap_h + 8 if is_jar else top_y + 22
            cv2.rectangle(body_mask, (left_x, y_start), (right_x, bot_y), 1.0, -1)
            cv2.ellipse(body_mask, (cx, bot_y), (bw // 2, 14), 0, 0, 180, 1.0, -1)

            x_coords = np.arange(w)[None, :]
            u = np.clip((x_coords - cx) / (bw / 2.0), -1.0, 1.0)
            nz = np.sqrt(np.maximum(0.01, 1.0 - u ** 2))
            diffuse = np.maximum(0.0, u * (-0.55) + nz * 0.65)
            light_map = 0.26 + diffuse * 0.85
            specular = np.exp(-((u + 0.32) ** 2) / 0.015) * 85.0
            rim = np.power(np.abs(u), 4) * 45.0

            shaded_cyl = sub_canvas * light_map[:, :, None] + specular[:, :, None] + rim[:, :, None]
            final_canvas = np.where(body_mask[:, :, None] > 0.5, np.clip(shaded_cyl, 0, 255), final_canvas)

            # Cap
            cap_w = geom["cap_w"] if is_jar else geom["neck_w"]
            cap_lx = cx - cap_w // 2
            cap_rx = cx + cap_w // 2
            cap_top_y = top_y if is_jar else geom["cap_y"]
            cap_mask = np.zeros((h, w), dtype=np.float32)
            cv2.rectangle(cap_mask, (cap_lx, cap_top_y), (cap_rx, cap_top_y + cap_h), 1.0, -1)
            cv2.ellipse(cap_mask, (cx, cap_top_y), (cap_w // 2, 8), 0, 0, 360, 1.0, -1)
            cap_sub = np.clip(sub_canvas * 1.18 + 20, 0, 255)
            final_canvas = np.where(cap_mask[:, :, None] > 0.5, cap_sub, final_canvas)
            cv2.ellipse(final_canvas, (cx, cap_top_y), (cap_w // 2, 8), 0, 0, 360, (255, 255, 255), 2)

            # Label with finish
            mask_lbl = np.zeros((h, w), dtype=np.float32)
            lbl_y1 = y_start + int(bh * 0.15)
            lbl_y2 = bot_y - int(bh * 0.18)
            cv2.rectangle(mask_lbl, (left_x + 16, lbl_y1), (right_x - 16, lbl_y2), 1.0, -1)
            final_canvas = self.facet_renderer.apply_finish_to_panel(final_canvas, mask_lbl, finish, lighting)

        elif cat == "can":
            cw, ch = geom["cw"], geom["ch"]
            left_x, right_x = geom["left_x"], geom["right_x"]
            top_y, bot_y = geom["top_y"], geom["bot_y"]
            chime_w = geom["chime_w"]

            # Floor shadow
            sh_mask = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(sh_mask, (cx, bot_y + 8), (int(cw * 0.52), int(cw * 0.12)), 0, 0, 360, 1.0, -1)
            final_canvas *= (1.0 - cv2.GaussianBlur(sh_mask, (35, 19), 8)[:, :, None] * 0.72)

            # Aluminum can body with specular highlight band
            body_mask = np.zeros((h, w), dtype=np.float32)
            cv2.rectangle(body_mask, (left_x, top_y + 32), (right_x, bot_y - 20), 1.0, -1)

            x_coords = np.arange(w)[None, :]
            u = np.clip((x_coords - cx) / (cw / 2.0), -1.0, 1.0)
            nz = np.sqrt(np.maximum(0.01, 1.0 - u ** 2))
            diffuse = np.maximum(0.0, u * (-0.55) + nz * 0.65)
            specular = np.exp(-((u + 0.32) ** 2) / 0.012) * 95.0
            rim = np.power(np.abs(u), 4) * 45.0
            shaded_can = sub_canvas * (0.28 + diffuse * 0.85)[:, :, None] + specular[:, :, None] + rim[:, :, None]
            final_canvas = np.where(body_mask[:, :, None] > 0.5, np.clip(shaded_can, 0, 255), final_canvas)

            # Top chime rim
            chime_mask = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(chime_mask, (cx, top_y + 10), (int(chime_w * 0.5), 9), 0, 0, 360, 1.0, -1)
            chime_sub = np.clip(sub_canvas * 1.25 + 25, 0, 255)
            final_canvas = np.where(chime_mask[:, :, None] > 0.5, chime_sub, final_canvas)
            cv2.ellipse(final_canvas, (cx, top_y + 10), (int(chime_w * 0.5), 9), 0, 0, 360, (255, 255, 255), 2)

            # Label band with finish
            mask_lbl = np.zeros((h, w), dtype=np.float32)
            cv2.rectangle(mask_lbl, (left_x + 8, top_y + int(ch * 0.28)), (right_x - 8, bot_y - int(ch * 0.26)), 1.0, -1)
            final_canvas = self.facet_renderer.apply_finish_to_panel(final_canvas, mask_lbl, finish, lighting)

        elif cat == "tube":
            tw, th = geom["tw"], geom["th"]
            top_y, bot_y = geom["top_y"], geom["bot_y"]
            crimp_w, crimp_h = geom["crimp_w"], geom["crimp_h"]
            crimp_lx = cx - crimp_w // 2
            crimp_rx = cx + crimp_w // 2
            cap_w = int(tw * 0.65)
            cap_lx = cx - cap_w // 2
            cap_rx = cx + cap_w // 2

            # Floor shadow under cap
            sh_mask = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(sh_mask, (cx, bot_y + 4), (int(cap_w * 0.60), 8), 0, 0, 360, 1.0, -1)
            final_canvas *= (1.0 - cv2.GaussianBlur(sh_mask, (25, 13), 6)[:, :, None] * 0.70)

            # Tube body lofting from flat top to round bottom
            tube_body_mask = np.zeros((h, w), dtype=np.float32)
            tube_poly = np.array([
                (crimp_lx, top_y + crimp_h), (crimp_rx, top_y + crimp_h),
                (cx + tw // 2, bot_y - 45), (cx - tw // 2, bot_y - 45)
            ], dtype=np.int32)
            cv2.fillPoly(tube_body_mask, [tube_poly], 1.0)

            x_coords = np.arange(w)[None, :]
            u = np.clip((x_coords - cx) / (tw / 2.0), -1.0, 1.0)
            nz = np.sqrt(np.maximum(0.01, 1.0 - u ** 2))
            diffuse = np.maximum(0.0, u * (-0.55) + nz * 0.65)
            specular = np.exp(-((u + 0.30) ** 2) / 0.02) * 70.0
            rim = np.power(np.abs(u), 4) * 35.0
            shaded_tube = sub_canvas * (0.30 + diffuse * 0.80)[:, :, None] + specular[:, :, None] + rim[:, :, None]
            final_canvas = np.where(tube_body_mask[:, :, None] > 0.5, np.clip(shaded_tube, 0, 255), final_canvas)

            # Crimped top seam
            crimp_mask = np.zeros((h, w), dtype=np.float32)
            cv2.rectangle(crimp_mask, (crimp_lx, top_y), (crimp_rx, top_y + crimp_h), 1.0, -1)
            crimp_shaded = sub_canvas * 1.15
            for kx in range(crimp_lx + 6, crimp_rx - 4, 4):
                cv2.line(crimp_shaded, (kx, top_y + 2), (kx, top_y + crimp_h - 2), (150, 150, 155), 1)
            final_canvas = np.where(crimp_mask[:, :, None] > 0.5, crimp_shaded, final_canvas)

            # Cap
            cap_mask = np.zeros((h, w), dtype=np.float32)
            cv2.rectangle(cap_mask, (cap_lx, bot_y - 32), (cap_rx, bot_y), 1.0, -1)
            final_canvas = np.where(cap_mask[:, :, None] > 0.5, np.clip(sub_canvas * 1.10, 0, 255), final_canvas)
            cv2.line(final_canvas, (cap_lx, bot_y - 16), (cap_rx, bot_y - 16), (60, 60, 65), 2)

            # Label with finish
            mask_lbl = np.zeros((h, w), dtype=np.float32)
            cv2.rectangle(mask_lbl, (cx - int(tw * 0.30), top_y + crimp_h + 30), (cx + int(tw * 0.30), bot_y - 65), 1.0, -1)
            final_canvas = self.facet_renderer.apply_finish_to_panel(final_canvas, mask_lbl, finish, lighting)

        # Subtle 35mm photographic film grain
        grain = np.random.normal(0, 1.8, (h, w, 3))
        final_bgr = np.clip(final_canvas + grain, 0, 255).astype(np.uint8)
        return Image.fromarray(cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB))

    def _generate_depth_map(self, geom: Dict[str, Any]) -> Image.Image:
        """Generates matching 3D structural pseudo-depth map."""
        w, h = self.size
        depth = np.full((h, w), 35, dtype=np.uint8)
        cat = geom["category"]
        cx, cy = geom["cx"], geom["cy"]

        # Pedestal / floor gradient
        depth[int(h * 0.76):, :] = np.linspace(60, 110, h - int(h * 0.76))[:, None].astype(np.uint8)

        if cat in ("carton", "box"):
            P_front_top = geom["P_front_top"]
            P_front_bot = geom["P_front_bot"]
            P_left_top = geom["P_left_top"]
            P_left_bot = geom["P_left_bot"]
            P_right_top = geom["P_right_top"]
            P_right_bot = geom["P_right_bot"]
            P_back_top = geom["P_back_top"]

            # Left face
            mask_left = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask_left, [np.array([P_left_top, P_front_top, P_front_bot, P_left_bot], dtype=np.int32)], 255)
            # Right face
            mask_right = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask_right, [np.array([P_front_top, P_right_top, P_right_bot, P_front_bot], dtype=np.int32)], 255)
            # Top face
            mask_top = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask_top, [np.array([P_left_top, P_back_top, P_right_top, P_front_top], dtype=np.int32)], 255)

            depth[mask_top > 0] = 205
            depth[mask_left > 0] = 240
            depth[mask_right > 0] = 185

        elif cat == "pouch":
            pw, ph = geom["pw"], geom["ph"]
            top_y, bot_y = geom["top_y"], geom["bot_y"]
            left_x, right_x = geom["left_x"], geom["right_x"]
            mask = np.zeros((h, w), dtype=np.uint8)
            poly = np.array([
                (left_x, top_y), (right_x, top_y),
                (right_x, bot_y - 25), (cx + pw // 4, bot_y),
                (cx - pw // 4, bot_y), (left_x, bot_y - 25)
            ], dtype=np.int32)
            cv2.fillPoly(mask, [poly], 255)
            x_coords = np.arange(w)[None, :]
            u = np.clip((x_coords - cx) / (pw / 2.0), -1.0, 1.0)
            curved_d = (195 + 50 * np.sqrt(np.maximum(0.01, 1.0 - u ** 2))).astype(np.uint8)
            depth[mask > 0] = np.tile(curved_d, (h, 1))[mask > 0]

        elif cat in ("bottle", "jar", "can", "tube"):
            cw = geom.get("cw") or geom.get("bw") or geom.get("jw") or geom.get("tw")
            top_y = geom.get("top_y") or geom.get("cap_y")
            bot_y = geom["bot_y"]
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(mask, (cx - cw // 2, top_y), (cx + cw // 2, bot_y), 255, -1)
            x_coords = np.arange(w)[None, :]
            u = np.clip((x_coords - cx) / (cw / 2.0), -1.0, 1.0)
            curved_d = (190 + 55 * np.sqrt(np.maximum(0.01, 1.0 - u ** 2))).astype(np.uint8)
            depth[mask > 0] = np.tile(curved_d, (h, 1))[mask > 0]

        return Image.fromarray(depth, mode="L")

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
        """Generates a complete paired sample: sketch, 3D facet render, depth map, and metadata."""
        rng = random.Random(seed)
        np.random.seed(seed)

        target_dir = self.train_dir if split == "train" else self.val_dir

        # 1. Compute parameterized 3D geometry
        geom = self._get_geometry(category, rng)

        # 2. Generate paired sketch
        sketch_pil = self._draw_geometric_sketch(geom, rng)
        sketch_filename = f"{sample_id}_sketch.png"
        sketch_path = target_dir / sketch_filename
        sketch_pil.save(sketch_path, format="PNG")

        # 3. Generate 3D pseudo-depth map
        depth_pil = self._generate_depth_map(geom)
        depth_filename = f"{sample_id}_depth.png"
        depth_path = target_dir / depth_filename
        depth_pil.save(depth_path, format="PNG")

        # 4. Render photorealistic packaging mockup via 3D Facet PBR engine
        render_pil = self._render_ground_truth(
            geom=geom,
            material=material,
            finish=finish,
            lighting=lighting,
            background=background,
            rng=rng,
        )
        render_filename = f"{sample_id}_render.png"
        render_path = target_dir / render_filename
        render_pil.save(render_path, format="PNG")

        # 5. Build structured packaging prompt
        brand_names = [
            "Aura Botanical", "Nordic Pure", "Lumina Luxe", "Verde Craft",
            "Solstice Organics", "Kyoto Studio", "Vesper Atelier", "Apex Roastery",
        ]
        brand_name = rng.choice(brand_names)
        palette = rng.choice([
            "minimalist earth tones", "deep luxury noir",
            "matte pastel sage", "warm terracotta", "monochrome slate",
        ])

        prompt_substrate = SUBSTRATE_TO_PROMPT.get(material, "coated SBS")
        prompt_finish = FINISH_TO_PROMPT.get(finish, "matte")
        prompt_lighting = LIGHTING_TO_PROMPT.get(lighting, "studio softbox")
        prompt_background = BACKGROUND_TO_PROMPT.get(background, "seamless studio")

        pos_prompt, neg_prompt = build_packaging_prompt(
            category=category,
            substrate=prompt_substrate,
            finish=prompt_finish,
            lighting=prompt_lighting,
            camera_angle="three-quarter",
            background=prompt_background,
            brand_name=brand_name,
            color_palette=palette,
        )

        # 6. Compile metadata record
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

    def _load_existing_records(self) -> List[Dict[str, Any]]:
        """Load records already present in the output-root manifests (for appends)."""
        meta_file = self.output_root / "metadata.jsonl"
        records: List[Dict[str, Any]] = []
        if not meta_file.exists():
            return records
        with open(meta_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def next_start_index(self) -> int:
        """Return the next free pv_pkg_NNNN index so new ids never collide."""
        max_idx = 0
        for rec in self._load_existing_records():
            sid = str(rec.get("sample_id", ""))
            if sid.startswith("pv_pkg_"):
                suffix = sid[len("pv_pkg_"):].split("_", 1)[0]
                if suffix.isdigit():
                    max_idx = max(max_idx, int(suffix))
        return max_idx + 1 if max_idx else 1

    def build_dataset(
        self,
        samples_per_category: int = 30,
        val_samples_per_category: int = 5,
        base_seed: int = 2026,
        start_index: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Generates a full paired dataset of at least 210 diverse high-quality paired triplets.

        ``start_index`` continues the pv_pkg_NNNN numbering (auto-detected from the
        existing metadata.jsonl when omitted) so a later run APPENDS new triplets
        instead of overwriting earlier ids. Existing manifest records are always
        preserved: manifests are merged (old records first, new appended), never
        truncated.
        """
        total_samples = samples_per_category * len(self.categories)
        total_val = val_samples_per_category * len(self.categories)
        total_train = total_samples - total_val

        if start_index is None:
            start_index = self.next_start_index()

        print(f"Generating PackVision AI custom paired dataset ({total_samples} samples across 7 categories)...")
        print(f"  sample ids: pv_pkg_{start_index:04d} .. pv_pkg_{start_index + total_samples - 1:04d}")
        records = []
        start_time = time.time()

        sample_idx = 0
        for cat_i, category in enumerate(self.categories):
            for i in range(samples_per_category):
                sample_idx += 1
                global_idx = start_index + sample_idx - 1
                sample_id = f"pv_pkg_{global_idx:04d}"
                split = "val" if i >= (samples_per_category - val_samples_per_category) else "train"
                seed = base_seed + global_idx * 37

                # Balanced material and finish rotation
                material = self.materials[(cat_i + i) % len(self.materials)]
                finish = self.finishes[(cat_i * 2 + i) % len(self.finishes)]
                lighting = self.lighting_setups[(i) % len(self.lighting_setups)]
                background = self.backgrounds[(cat_i + i) % len(self.backgrounds)]

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

                if sample_idx % 30 == 0 or sample_idx == total_samples:
                    elapsed = time.time() - start_time
                    print(f"  Generated {sample_idx}/{total_samples} samples ({elapsed:.1f}s)")

        # Write overall metadata.jsonl and split manifests.
        # Merge with any existing records so appends never clobber prior data.
        meta_file = self.output_root / "metadata.jsonl"
        train_file = self.output_root / "train_manifest.jsonl"
        val_file = self.output_root / "val_manifest.jsonl"

        existing = self._load_existing_records()
        existing_ids = {r.get("sample_id") for r in existing}
        merged = list(existing)
        skipped = 0
        for r in records:
            if r["sample_id"] in existing_ids:
                print(f"  [warn] sample_id {r['sample_id']} already exists; keeping the old record")
                skipped += 1
                continue
            existing_ids.add(r["sample_id"])
            merged.append(r)

        with open(meta_file, "w", encoding="utf-8") as f_all, \
             open(train_file, "w", encoding="utf-8") as f_train, \
             open(val_file, "w", encoding="utf-8") as f_val:
            for r in merged:
                line = json.dumps(r) + "\n"
                f_all.write(line)
                if r.get("split") == "train":
                    f_train.write(line)
                else:
                    f_val.write(line)

        print(f"Dataset generation complete!")
        print(f"  Generated: {len(records) - skipped} new samples ({total_train} train, {total_val} val)")
        print(f"  Manifest records: {len(existing)} -> {len(merged)} (merged)")
        print(f"  Manifest: {meta_file}")
        return records


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    default_categories = ["carton", "box", "pouch", "bottle", "jar", "can", "tube"]
    parser = argparse.ArgumentParser(
        description="Generate the paired PackVision AI packaging dataset "
                    "(sketch / render / depth triplets + manifests)."
    )
    parser.add_argument(
        "--samples", type=int, default=30,
        help="Samples generated per category (default: 30).",
    )
    parser.add_argument(
        "--val-samples", type=int, default=5,
        help="Of the per-category samples, how many go to the val split (default: 5).",
    )
    parser.add_argument(
        "--categories", type=str, default=",".join(default_categories),
        help="Comma-separated category list (default: carton,box,pouch,bottle,jar,can,tube).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base seed for deterministic generation (default: 42).",
    )
    parser.add_argument(
        "--start-index", type=int, default=None,
        help="First pv_pkg_NNNN index for this run (default: auto-detect the next "
             "free index from the existing metadata.jsonl so runs append safely).",
    )
    parser.add_argument(
        "--output-root", type=str, default="data/dataset",
        help="Output directory for triplets and manifests (default: data/dataset).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    cli_args = _parse_args()
    generator = PackagingDatasetGenerator(output_root=cli_args.output_root)
    categories = [c.strip() for c in cli_args.categories.split(",") if c.strip()]
    unknown = [c for c in categories if c not in generator.categories]
    if unknown:
        raise SystemExit(
            f"Unknown categories: {unknown}. "
            f"Supported geometry categories: {generator.categories}"
        )
    generator.categories = categories
    generator.build_dataset(
        samples_per_category=cli_args.samples,
        val_samples_per_category=cli_args.val_samples,
        base_seed=cli_args.seed,
        start_index=cli_args.start_index,
    )
