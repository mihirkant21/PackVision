"""
PackVision AI - Sample Asset Generator
======================================
Generates realistic packaging sketch assets and a brand logo:
1. carton_sketch.png - Tea folding carton sketch
2. pouch_sketch.png  - Stand-up flexible pouch sketch
3. bottle_sketch.png - Cosmetic dropper bottle sketch
4. sample_logo.png   - Sharp luxury brand logo with transparent alpha
"""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_carton_sketch(size: int = 1024) -> Image.Image:
    """Generate crisp, architectural 3D line sketch of a folding tea carton."""
    canvas = np.full((size, size, 3), 252, dtype=np.uint8)
    line_color = (25, 25, 25)

    # Perspective 3-point vertices for front, side, and top panels
    # Front panel
    p_tl = (340, 360)
    p_tr = (640, 320)
    p_br = (640, 800)
    p_bl = (340, 840)

    # Right side panel
    p_side_tr = (760, 270)
    p_side_br = (760, 710)

    # Top panel / tuck flap
    p_top_peak = (460, 210)
    p_top_back = (580, 190)

    # Draw front panel
    cv2.polylines(canvas, [np.array([p_tl, p_tr, p_br, p_bl])], True, line_color, 4, cv2.LINE_AA)

    # Draw right side panel
    cv2.polylines(canvas, [np.array([p_tr, p_side_tr, p_side_br, p_br])], True, line_color, 4, cv2.LINE_AA)

    # Draw top closure panel & tuck-in flap
    cv2.polylines(canvas, [np.array([p_tl, p_top_peak, p_side_tr, p_tr])], True, line_color, 3, cv2.LINE_AA)
    cv2.line(canvas, p_top_peak, p_top_back, line_color, 2, cv2.LINE_AA)
    cv2.line(canvas, p_top_back, p_side_tr, line_color, 2, cv2.LINE_AA)

    # Structural fold creases (dashed / subtle lines)
    cv2.line(canvas, (340, 420), (640, 380), (80, 80, 80), 2, cv2.LINE_AA)  # Top tuck line
    cv2.line(canvas, (640, 380), (760, 330), (80, 80, 80), 2, cv2.LINE_AA)

    # Subtle inner artwork guidelines / window zone
    cv2.rectangle(canvas, (400, 480), (580, 680), (120, 120, 120), 2, cv2.LINE_AA)
    cv2.putText(canvas, "TEA ATELIER", (420, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2, cv2.LINE_AA)
    cv2.putText(canvas, "ORGANIC HERBAL", (415, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 130, 130), 1, cv2.LINE_AA)

    # Bottom crease line
    cv2.line(canvas, (340, 820), (640, 780), (100, 100, 100), 1, cv2.LINE_AA)

    # Add subtle pencil stroke noise
    noise = np.random.normal(0, 1.5, canvas.shape).astype(np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def create_pouch_sketch(size: int = 1024) -> Image.Image:
    """Generate sketch of a stand-up flexible pouch (doypack)."""
    canvas = np.full((size, size, 3), 252, dtype=np.uint8)
    line_color = (25, 25, 25)

    # Pouch outer perimeter points
    pts_left = [(310, 260), (320, 380), (310, 760)]
    pts_right = [(714, 260), (704, 380), (714, 760)]

    # Top seal line
    cv2.line(canvas, (310, 260), (714, 260), line_color, 4, cv2.LINE_AA)
    cv2.line(canvas, (310, 310), (714, 310), (60, 60, 60), 2, cv2.LINE_AA)  # Heat seal band

    # Side heat-seal borders
    cv2.line(canvas, (310, 260), (310, 760), line_color, 4, cv2.LINE_AA)
    cv2.line(canvas, (330, 260), (330, 760), (110, 110, 110), 1, cv2.LINE_AA)

    cv2.line(canvas, (714, 260), (714, 760), line_color, 4, cv2.LINE_AA)
    cv2.line(canvas, (694, 260), (694, 760), (110, 110, 110), 1, cv2.LINE_AA)

    # Bottom stand-up gusset curve
    cv2.ellipse(canvas, (512, 760), (202, 45), 0, 0, 180, line_color, 4, cv2.LINE_AA)
    cv2.ellipse(canvas, (512, 750), (180, 35), 0, 0, 180, (90, 90, 90), 2, cv2.LINE_AA)  # Inner fold

    # Tear notches (left and right V-notches)
    cv2.line(canvas, (310, 285), (325, 285), line_color, 2, cv2.LINE_AA)
    cv2.line(canvas, (714, 285), (699, 285), line_color, 2, cv2.LINE_AA)

    # Press-to-close zipper guideline
    for x in range(330, 690, 12):
        cv2.line(canvas, (x, 340), (x + 6, 340), (100, 100, 100), 2, cv2.LINE_AA)

    # Front panel curvature lines indicating volume fullness
    cv2.ellipse(canvas, (512, 540), (150, 140), 0, 30, 150, (140, 140, 140), 1, cv2.LINE_AA)
    cv2.ellipse(canvas, (512, 540), (150, 140), 0, 210, 330, (140, 140, 140), 1, cv2.LINE_AA)

    # Brand typography guide
    cv2.putText(canvas, "ROAST & GROUND", (410, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 90, 90), 2, cv2.LINE_AA)
    cv2.putText(canvas, "STAND-UP DOYPACK 250G", (390, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)

    noise = np.random.normal(0, 1.5, canvas.shape).astype(np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def create_bottle_sketch(size: int = 1024) -> Image.Image:
    """Generate sketch of a cosmetic glass dropper bottle."""
    canvas = np.full((size, size, 3), 252, dtype=np.uint8)
    line_color = (25, 25, 25)

    # Cylindrical Body
    p_body_tl = (360, 440)
    p_body_tr = (664, 440)
    p_body_bl = (360, 830)
    p_body_br = (664, 830)

    # Outer body lines
    cv2.line(canvas, p_body_tl, p_body_bl, line_color, 4, cv2.LINE_AA)
    cv2.line(canvas, p_body_tr, p_body_br, line_color, 4, cv2.LINE_AA)

    # Base curve
    cv2.ellipse(canvas, (512, 830), (152, 28), 0, 0, 180, line_color, 4, cv2.LINE_AA)
    # Heavy glass base thickness
    cv2.ellipse(canvas, (512, 805), (135, 20), 0, 0, 180, (90, 90, 90), 2, cv2.LINE_AA)

    # Shoulders
    cv2.ellipse(canvas, (512, 440), (152, 25), 0, 180, 360, line_color, 4, cv2.LINE_AA)
    cv2.line(canvas, (360, 440), (445, 345), line_color, 4, cv2.LINE_AA)
    cv2.line(canvas, (664, 440), (579, 345), line_color, 4, cv2.LINE_AA)

    # Bottle Neck & Collar
    cv2.rectangle(canvas, (445, 275), (579, 345), line_color, 4, cv2.LINE_AA)
    cv2.line(canvas, (445, 305), (579, 305), (80, 80, 80), 2, cv2.LINE_AA)

    # Rubber Suction Bulb
    cv2.ellipse(canvas, (512, 230), (45, 48), 0, 0, 360, line_color, 4, cv2.LINE_AA)

    # Inner Glass Pipette tube
    cv2.line(canvas, (504, 345), (504, 780), (110, 110, 110), 2, cv2.LINE_AA)
    cv2.line(canvas, (520, 345), (520, 780), (110, 110, 110), 2, cv2.LINE_AA)
    cv2.ellipse(canvas, (512, 780), (8, 12), 0, 0, 360, (110, 110, 110), 2, cv2.LINE_AA)

    # Optical glass wall thickness lines
    cv2.line(canvas, (375, 440), (375, 805), (140, 140, 140), 1, cv2.LINE_AA)
    cv2.line(canvas, (649, 440), (649, 805), (140, 140, 140), 1, cv2.LINE_AA)

    # Label area indication
    cv2.rectangle(canvas, (410, 520), (614, 720), (120, 120, 120), 2, cv2.LINE_AA)
    cv2.putText(canvas, "SERUM 30ML", (445, 610), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 2, cv2.LINE_AA)

    noise = np.random.normal(0, 1.5, canvas.shape).astype(np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def create_sample_logo(size: int = 1024) -> Image.Image:
    """Generate high-resolution vector-style brand logo with transparent alpha channel."""
    logo_rgba = np.zeros((size, size, 4), dtype=np.uint8)

    cx, cy = size // 2, int(size * 0.40)
    primary_color = (40, 35, 30, 255)  # Refined charcoal-black (BGRA: B, G, R, A)
    gold_color = (30, 185, 220, 255)     # Luxury Gold (BGRA)

    # 1. Geometric Botanical Emblem (8-petal mandala ring)
    r_outer = 110
    r_inner = 70
    cv2.circle(logo_rgba, (cx, cy), r_outer, gold_color, 4, cv2.LINE_AA)
    cv2.circle(logo_rgba, (cx, cy), r_inner, primary_color, 2, cv2.LINE_AA)

    for angle_deg in range(0, 360, 45):
        rad = np.deg2rad(angle_deg)
        px = int(cx + r_outer * 0.7 * np.cos(rad))
        py = int(cy + r_outer * 0.7 * np.sin(rad))
        cv2.circle(logo_rgba, (px, py), 18, gold_color, 2, cv2.LINE_AA)

    cv2.circle(logo_rgba, (cx, cy), 12, gold_color, -1, cv2.LINE_AA)

    # 2. Brand Name Typography
    text_y = int(size * 0.66)
    font = cv2.FONT_HERSHEY_COMPLEX
    
    brand_text = "A U R O R A"
    text_size = cv2.getTextSize(brand_text, font, 2.2, 5)[0]
    tx = (size - text_size[0]) // 2
    cv2.putText(logo_rgba, brand_text, (tx, text_y), font, 2.2, primary_color, 5, cv2.LINE_AA)

    # Subtitle
    sub_text = "ORGANIC BOTANICALS"
    font_sub = cv2.FONT_HERSHEY_SIMPLEX
    sub_size = cv2.getTextSize(sub_text, font_sub, 0.9, 2)[0]
    sx = (size - sub_size[0]) // 2
    cv2.putText(logo_rgba, sub_text, (sx, text_y + 55), font_sub, 0.9, (80, 75, 70, 255), 2, cv2.LINE_AA)

    # Est. 2026 seal
    est_text = "NEW YORK * PARIS"
    est_size = cv2.getTextSize(est_text, font_sub, 0.6, 1)[0]
    ex = (size - est_size[0]) // 2
    cv2.putText(logo_rgba, est_text, (ex, text_y + 100), font_sub, 0.6, (140, 130, 120, 255), 1, cv2.LINE_AA)

    # Horizontal divider rules
    rule_y = text_y + 75
    cv2.line(logo_rgba, (cx - 160, rule_y), (cx - 40, rule_y), gold_color, 2, cv2.LINE_AA)
    cv2.line(logo_rgba, (cx + 40, rule_y), (cx + 160, rule_y), gold_color, 2, cv2.LINE_AA)
    cv2.circle(logo_rgba, (cx, rule_y), 4, gold_color, -1, cv2.LINE_AA)

    # Convert to RGBA PIL Image
    pil_img = Image.fromarray(cv2.cvtColor(logo_rgba, cv2.COLOR_BGRA2RGBA))
    return pil_img


def main() -> None:
    samples_dir = Path(__file__).resolve().parent
    samples_dir.mkdir(parents=True, exist_ok=True)

    print(f"[SampleAssets] Generating packaging sketches in {samples_dir}...")

    carton = create_carton_sketch()
    carton.save(samples_dir / "carton_sketch.png", quality=95)
    print("  -> Created carton_sketch.png")

    pouch = create_pouch_sketch()
    pouch.save(samples_dir / "pouch_sketch.png", quality=95)
    print("  -> Created pouch_sketch.png")

    bottle = create_bottle_sketch()
    bottle.save(samples_dir / "bottle_sketch.png", quality=95)
    print("  -> Created bottle_sketch.png")

    logo = create_sample_logo()
    logo.save(samples_dir / "sample_logo.png")
    print("  -> Created sample_logo.png")

    print("[SampleAssets] All sample assets successfully generated!")


if __name__ == "__main__":
    main()
