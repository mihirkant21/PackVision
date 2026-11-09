"""
PackVision AI - ML Workbench & Rapid Experimentation Suite
=========================================================
Standalone interactive Gradio application for packaging designers, ML researchers,
and brand creative directors to transform 2D packaging sketches into photoreal mockups.

Features:
- Canvas sketch upload & brush drawing / touch-up tool
- Industrial packaging taxonomy selectors (Category, Substrate, Finish, Lighting, Viewpoint, Backdrop)
- Vector logo overlay with projective homography & CIELAB luminance blending
- Interactive Before / After Split Comparison Slider
- Quantitative evaluation metrics (Edge SSIM fidelity, CLIP alignment score, inference time)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gradio as gr
import numpy as np
from PIL import Image, ImageOps

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe imports for PackVision Core ML modules
try:
    from src.pipeline import PackVisionRenderer, RenderResult
    from src.postprocessor import LogoCompositor
    from src.prompt_engine import PromptEngine, build_packaging_prompt
    from src.evaluation import PackagingEvaluator, compute_edge_ssim, compute_edge_iou
    CORE_AVAILABLE = True
except ImportError as err:
    print(f"[Warning] Core ML engine modules import warning: {err}")
    CORE_AVAILABLE = False

# Lazy-loaded singleton engine instances
_RENDERER: Optional[PackVisionRenderer] = None
_COMPOSITOR: Optional[LogoCompositor] = None
_EVALUATOR: Optional[PackagingEvaluator] = None


def get_renderer() -> PackVisionRenderer:
    global _RENDERER
    if _RENDERER is None:
        use_mock = os.getenv("USE_MOCK_INFERENCE", "true").lower() in ("true", "1", "yes")
        _RENDERER = PackVisionRenderer(use_mock_inference=use_mock)
    return _RENDERER


def get_compositor() -> LogoCompositor:
    global _COMPOSITOR
    if _COMPOSITOR is None:
        _COMPOSITOR = LogoCompositor()
    return _COMPOSITOR


def get_evaluator() -> PackagingEvaluator:
    global _EVALUATOR
    if _EVALUATOR is None:
        _EVALUATOR = PackagingEvaluator()
    return _EVALUATOR


def extract_pil_from_canvas(sketch_input: Any) -> Optional[Image.Image]:
    """Extract standard PIL Image from Gradio ImageEditor or Image component."""
    if sketch_input is None:
        return None
    if isinstance(sketch_input, Image.Image):
        return sketch_input
    if isinstance(sketch_input, dict):
        # Gradio ImageEditor returns dict with 'composite', 'background', 'layers'
        composite = sketch_input.get("composite")
        if composite is not None:
            if isinstance(composite, Image.Image):
                return composite
            elif isinstance(composite, np.ndarray):
                return Image.fromarray(composite)
        bg = sketch_input.get("background")
        if bg is not None:
            if isinstance(bg, Image.Image):
                return bg
            elif isinstance(bg, np.ndarray):
                return Image.fromarray(bg)
    if isinstance(sketch_input, np.ndarray):
        return Image.fromarray(sketch_input)
    if isinstance(sketch_input, str) and os.path.exists(sketch_input):
        return Image.open(sketch_input)
    return None


def auto_enrich_prompt(
    category: str,
    material: str,
    finish: str,
    lighting: str,
    camera: str,
    background: str,
    user_prompt: str,
) -> str:
    """Generate an enriched engineering prompt based on selected taxonomy."""
    try:
        from src.prompt_engine import PromptEngine
        engine = PromptEngine()
        pkg = engine.build_prompt(
            category=category.lower().replace(" ", "_"),
            substrate=material.lower().replace(" ", "_"),
            finish=finish.lower().replace(" ", "_"),
            lighting=lighting.lower().replace(" ", "_"),
            camera_angle=camera.lower().replace(" ", "_"),
            background=background.lower().replace(" ", "_"),
            user_prompt=user_prompt if user_prompt.strip() else None,
        )
        return pkg.positive_prompt
    except Exception:
        parts = [
            f"Commercial {category} packaging mockup",
            f"crafted from authentic {material}",
            f"featuring {finish} finish",
            f"lit by {lighting}",
            f"{camera} framing",
            f"set against {background}",
        ]
        if user_prompt.strip():
            parts.insert(1, user_prompt.strip())
        return ", ".join(parts) + ", 8k resolution, raytraced ambient occlusion, master product photo."


def invert_sketch_lines(sketch_input: Any) -> Any:
    """Invert sketch image contrast (black on white <-> white on black)."""
    pil_img = extract_pil_from_canvas(sketch_input)
    if pil_img is None:
        return sketch_input
    if pil_img.mode in ("RGBA", "LA"):
        # Separate alpha
        r, g, b, a = pil_img.split()
        rgb_img = Image.merge("RGB", (r, g, b))
        inv_rgb = ImageOps.invert(rgb_img)
        r2, g2, b2 = inv_rgb.split()
        inverted = Image.merge("RGBA", (r2, g2, b2, a))
    else:
        inverted = ImageOps.invert(pil_img.convert("RGB"))
    return inverted


def run_generation(
    sketch_input: Any,
    category: str,
    material: str,
    finish: str,
    lighting: str,
    camera: str,
    background: str,
    user_prompt: str,
    negative_prompt: str,
    logo_input: Optional[Any],
    logo_pos_x: float,
    logo_pos_y: float,
    logo_scale: float,
    blend_mode: str,
    steps: int,
    guidance_scale: float,
    controlnet_scale: float,
    seed: int,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[Any, str, str, str, str]:
    """Execute end-to-end rendering pipeline and return slider comparison + metrics."""
    pil_sketch = extract_pil_from_canvas(sketch_input)
    if pil_sketch is None:
        raise gr.Error("Please upload or draw a packaging sketch on the canvas first.")

    progress(0.1, desc="Queued: Initializing PackVision pipeline...")
    time.sleep(0.05)

    effective_seed = None if seed <= 0 else seed

    # 1. Pipeline Render
    progress(0.3, desc=f"Preprocessing sketch geometry & synthesizing {material} substrate...")
    renderer = get_renderer()

    norm_category = category.lower().replace(" ", "_")
    norm_material = material.lower().replace(" ", "_")
    norm_finish = finish.lower().replace(" ", "_")
    norm_lighting = lighting.lower().replace(" ", "_")
    norm_camera = camera.lower().replace(" ", "_")
    norm_bg = background.lower().replace(" ", "_")

    render_result: RenderResult = renderer.render(
        sketch=pil_sketch,
        category=norm_category,
        substrate=norm_material,
        finish=norm_finish,
        lighting=norm_lighting,
        camera_angle=norm_camera,
        background=norm_bg,
        user_prompt=user_prompt if user_prompt.strip() else None,
        conditioning_scale=controlnet_scale,
        seed=effective_seed,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
    )

    final_render = render_result.image
    orig_normalized = render_result.preprocessed_sketch

    # 2. Stage-2 Homography Logo Compositing (if logo provided)
    if logo_input is not None:
        progress(0.75, desc="Stage 2: Compositing brand logo with CIELAB luminance modulation...")
        pil_logo = extract_pil_from_canvas(logo_input)
        if pil_logo is not None:
            compositor = get_compositor()
            # Calculate quadrilateral destination coordinates from slider offsets
            w, h = final_render.size
            cx = (logo_pos_x / 100.0) * w
            cy = (logo_pos_y / 100.0) * h
            box_w = (logo_scale / 100.0) * (w * 0.45)
            box_h = box_w * (pil_logo.height / max(pil_logo.width, 1))

            dest_points = [
                (cx - box_w / 2, cy - box_h / 2),
                (cx + box_w / 2, cy - box_h / 2),
                (cx + box_w / 2, cy + box_h / 2),
                (cx - box_w / 2, cy + box_h / 2),
            ]

            comp_res = compositor.composite(
                package_image=final_render,
                logo_image=pil_logo,
                destination_points=dest_points,
                blend_mode=blend_mode.lower(),
                opacity=0.95,
            )
            final_render = comp_res.image

    progress(0.9, desc="Evaluating structural fidelity & CLIP semantic alignment...")

    # 3. Quantitative Evaluation
    evaluator = get_evaluator()
    ssim_val = compute_edge_ssim(orig_normalized, final_render)
    iou_val = compute_edge_iou(orig_normalized, final_render)
    clip_val = evaluator.compute_clip_score(
        final_render,
        f"{category} packaging in {material} with {finish} finish, {user_prompt}",
    )

    progress(1.0, desc="Render completed successfully!")

    # Format metrics display strings
    ssim_label = f"Edge SSIM: {ssim_val:.3f} ({'Excellent' if ssim_val > 0.8 else 'Good' if ssim_val > 0.65 else 'Moderate'})"
    iou_label = f"Edge IoU: {iou_val:.3f}"
    clip_label = f"CLIP Alignment: {clip_val:.3f}"
    latency_label = f"Time: {render_result.inference_time:.2f}s ({'Procedural PBR' if render_result.is_mock else 'SDXL ControlNet'})"

    # Gradio ImageSlider accepts (before, after) tuple
    slider_output = (orig_normalized, final_render)

    return slider_output, ssim_label, iou_label, clip_label, latency_label


# =============================================================================
# GRADIO INTERFACE CONSTRUCTION
# =============================================================================

CUSTOM_CSS = """
/* Luxury Packaging Dark Aesthetic */
body, .gradio-container {
    background-color: #0b0f19 !important;
    color: #e2e8f0 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
}

.brand-header {
    border-bottom: 1px solid rgba(212, 175, 55, 0.25);
    padding-bottom: 18px;
    margin-bottom: 24px;
}

.brand-logo-text {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #f59e0b 0%, #d4af37 50%, #fef08a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-flex;
    align-items: center;
    gap: 12px;
}

.badge-tag {
    display: inline-block;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border-radius: 9999px;
    background: rgba(212, 175, 55, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(212, 175, 55, 0.35);
}

.metrics-panel {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(212, 175, 55, 0.2);
    border-radius: 12px;
    padding: 14px 18px;
    margin-top: 14px;
}

.generate-btn {
    background: linear-gradient(135deg, #d97706 0%, #b45309 50%, #78350f 100%) !important;
    border: 1px solid rgba(245, 158, 11, 0.4) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 14px 0 rgba(217, 119, 6, 0.35) !important;
    transition: all 0.2s ease-in-out !important;
}

.generate-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px 0 rgba(217, 119, 6, 0.55) !important;
}
"""

with gr.Blocks(title="PackVision AI - Packaging Studio") as demo:

    with gr.Row(elem_classes=["brand-header"]):
        with gr.Column(scale=9):
            gr.HTML("""
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <div class="brand-logo-text">
                        <span>📦 PackVision AI</span>
                        <span class="badge-tag">STUDIO PRO v1.0</span>
                    </div>
                    <p style="color: #94a3b8; margin: 4px 0 0 0; font-size: 14px;">
                        Sketch-to-Photoreal Packaging Renderer • Conditional Structural Diffusion & PBR Shader Synthesis
                    </p>
                </div>
            </div>
            """)
        with gr.Column(scale=3, min_width=200):
            gr.HTML("""
            <div style="text-align: right; color: #64748b; font-size: 12px; padding-top: 8px;">
                <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #10b981; margin-right: 6px;"></span>
                Engine: <strong>Production Ready</strong>
            </div>
            """)

    with gr.Row():
        # LEFT COLUMN: INPUTS & CONTROLS
        with gr.Column(scale=5):
            gr.Markdown("### 1. Packaging Sketch Canvas")
            sketch_editor = gr.ImageEditor(
                type="pil",
                label="Upload Sketch or Draw / Touch-Up",
                sources=["upload", "clipboard"],
                brush=gr.Brush(colors=["#000000", "#ffffff", "#4b5563"], default_size=4),
                eraser=gr.Eraser(default_size=10),
                transforms=["crop"],
            )

            with gr.Row():
                btn_invert = gr.Button("🔄 Invert Sketch Lines", size="sm", variant="secondary")

            gr.Markdown("### 2. Packaging Taxonomy & Shaders")
            with gr.Row():
                category_dropdown = gr.Dropdown(
                    choices=["Carton", "Pouch", "Bottle", "Jar", "Can", "Box"],
                    value="Carton",
                    label="Packaging Category",
                )
                material_dropdown = gr.Dropdown(
                    choices=[
                        "Kraft",
                        "Corrugated",
                        "Matte Carton",
                        "Flexible Mylar",
                        "Frosted Glass",
                        "Brushed Aluminum",
                    ],
                    value="Kraft",
                    label="Substrate Material",
                )

            with gr.Row():
                finish_dropdown = gr.Dropdown(
                    choices=[
                        "Gold Foil",
                        "Silver Foil",
                        "Spot UV",
                        "Embossed",
                        "Debossed",
                        "Holographic",
                        "Matte",
                    ],
                    value="Gold Foil",
                    label="Print Finish & Foil",
                )
                lighting_dropdown = gr.Dropdown(
                    choices=[
                        "Three-Point Studio Softbox",
                        "Dramatic Luxury Rim Light",
                        "Commercial High-Key Clean",
                        "Natural Window Daylight",
                    ],
                    value="Three-Point Studio Softbox",
                    label="Studio Lighting",
                )

            with gr.Row():
                camera_dropdown = gr.Dropdown(
                    choices=[
                        "Front-Facing Hero",
                        "Three-Quarter Isometric",
                        "Low-Angle Dynamic",
                        "Top-Down Flat Lay",
                    ],
                    value="Three-Quarter Isometric",
                    label="Camera Viewpoint",
                )
                background_dropdown = gr.Dropdown(
                    choices=[
                        "Seamless White/Neutral Studio",
                        "Warm Beige Pedestal",
                        "Minimalist Dark Luxury Stone",
                        "Transparent/Isolated",
                    ],
                    value="Seamless White/Neutral Studio",
                    label="Backdrop Scene",
                )

            gr.Markdown("### 3. Creative Prompts")
            user_prompt_txt = gr.Textbox(
                label="Creative Prompt Directives",
                placeholder="e.g. Luxury herbal botanical tea packaging, minimalist gold typography, elegant aesthetic",
                value="Luxury artisanal tea box, gold foil stamped botanical motifs, minimalist aesthetic",
                lines=2,
            )
            btn_enrich = gr.Button("✨ Auto-Enrich Engineering Prompt", size="sm", variant="secondary")

            negative_prompt_txt = gr.Textbox(
                label="Negative Prompt",
                value="distorted geometry, broken dielines, uneven seams, low quality, warped text, cartoon, blur",
                lines=1,
            )

            # STAGE 2: LOGO COMPOSITING ACCORDION
            with gr.Accordion("🏷️ Stage-2 Brand Logo / Vector Overlay (Optional)", open=False):
                gr.Markdown("Warp vector logo onto 3D packaging face with CIELAB shadow & texture modulation.")
                logo_upload = gr.Image(
                    type="pil",
                    image_mode="RGBA",
                    label="Upload Flat Brand Logo (PNG with transparency)",
                )
                with gr.Row():
                    logo_x_slider = gr.Slider(
                        minimum=10.0, maximum=90.0, value=50.0, step=1.0, label="Position X (%)"
                    )
                    logo_y_slider = gr.Slider(
                        minimum=10.0, maximum=90.0, value=52.0, step=1.0, label="Position Y (%)"
                    )
                with gr.Row():
                    logo_scale_slider = gr.Slider(
                        minimum=10.0, maximum=100.0, value=50.0, step=2.0, label="Logo Scale (%)"
                    )
                    blend_mode_dropdown = gr.Dropdown(
                        choices=["Normal", "Multiply", "Overlay", "Hard_Light"],
                        value="Multiply",
                        label="Ink Blend Mode",
                    )

            # ADVANCED CONTROLS ACCORDION
            with gr.Accordion("⚙️ Advanced Inference Parameters", open=False):
                with gr.Row():
                    steps_slider = gr.Slider(
                        minimum=10, maximum=50, value=30, step=1, label="Denoising Steps"
                    )
                    guidance_slider = gr.Slider(
                        minimum=1.0, maximum=15.0, value=7.5, step=0.5, label="Guidance Scale"
                    )
                with gr.Row():
                    controlnet_slider = gr.Slider(
                        minimum=0.1, maximum=1.5, value=0.85, step=0.05, label="ControlNet Conditioning Scale"
                    )
                    seed_input = gr.Number(value=-1, label="Seed (-1 for random)", precision=0)

            btn_generate = gr.Button(
                "🚀 Generate Photoreal Mockup",
                size="lg",
                variant="primary",
                elem_classes=["generate-btn"],
            )

        # RIGHT COLUMN: PREVIEW STAGE & EVALUATION
        with gr.Column(scale=7):
            gr.Markdown("### Interactive Before / After Split Comparison")
            slider_output = gr.ImageSlider(
                label="Sketch (Left) vs. Photoreal Render (Right)",
                type="pil",
                show_label=True,
            )

            with gr.Row(elem_classes=["metrics-panel"]):
                metric_ssim = gr.Textbox(label="Structural Fidelity", value="Edge SSIM: --", interactive=False)
                metric_iou = gr.Textbox(label="Contour IoU", value="Edge IoU: --", interactive=False)
                metric_clip = gr.Textbox(label="Semantic Alignment", value="CLIP Score: --", interactive=False)
                metric_time = gr.Textbox(label="Inference Latency", value="Latency: --", interactive=False)

            # EXAMPLES GALLERY
            gr.Markdown("### Quick Load Sample Sketches")
            sample_dir = PROJECT_ROOT / "data" / "samples"
            samples = [
                [
                    str(sample_dir / "carton_sketch.png"),
                    "Carton",
                    "Kraft",
                    "Gold Foil",
                    "Three-Point Studio Softbox",
                    "Three-Quarter Isometric",
                    "Seamless White/Neutral Studio",
                    "Luxury herbal botanical tea packaging, gold foil embossed",
                ],
                [
                    str(sample_dir / "pouch_sketch.png"),
                    "Pouch",
                    "Flexible Mylar",
                    "Matte",
                    "Commercial High-Key Clean",
                    "Front-Facing Hero",
                    "Seamless White/Neutral Studio",
                    "Artisanal roasted coffee stand-up pouch, minimalist earthy aesthetic",
                ],
                [
                    str(sample_dir / "bottle_sketch.png"),
                    "Bottle",
                    "Frosted Glass",
                    "Gold Foil",
                    "Dramatic Luxury Rim Light",
                    "Front-Facing Hero",
                    "Minimalist Dark Luxury Stone",
                    "Luxury facial serum dropper bottle, frosted glass, metallic dropper",
                ],
            ]

            gr.Examples(
                examples=samples,
                inputs=[
                    sketch_editor,
                    category_dropdown,
                    material_dropdown,
                    finish_dropdown,
                    lighting_dropdown,
                    camera_dropdown,
                    background_dropdown,
                    user_prompt_txt,
                ],
                label="Click any sample to populate configuration and canvas",
            )

    # EVENT WIRING
    btn_invert.click(
        fn=invert_sketch_lines,
        inputs=[sketch_editor],
        outputs=[sketch_editor],
    )

    btn_enrich.click(
        fn=auto_enrich_prompt,
        inputs=[
            category_dropdown,
            material_dropdown,
            finish_dropdown,
            lighting_dropdown,
            camera_dropdown,
            background_dropdown,
            user_prompt_txt,
        ],
        outputs=[user_prompt_txt],
    )

    btn_generate.click(
        fn=run_generation,
        inputs=[
            sketch_editor,
            category_dropdown,
            material_dropdown,
            finish_dropdown,
            lighting_dropdown,
            camera_dropdown,
            background_dropdown,
            user_prompt_txt,
            negative_prompt_txt,
            logo_upload,
            logo_x_slider,
            logo_y_slider,
            logo_scale_slider,
            blend_mode_dropdown,
            steps_slider,
            guidance_slider,
            controlnet_slider,
            seed_input,
        ],
        outputs=[
            slider_output,
            metric_ssim,
            metric_iou,
            metric_clip,
            metric_time,
        ],
    )


if __name__ == "__main__":
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    host = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    print(f"Launching PackVision AI Gradio Workbench on http://{host}:{port}...")
    demo.launch(
        server_name=host,
        server_port=port,
        share=False,
        theme=gr.themes.Base(primary_hue="amber", neutral_hue="slate"),
        css=CUSTOM_CSS,
    )
