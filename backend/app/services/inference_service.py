import io
import time
import math
import random
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Tuple
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageEnhance

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


class InferenceService:
    """Service adapter bridging FastAPI backend jobs with AI rendering engines.
    
    Supports:
    - Real deep learning pipeline (PackVisionRenderer + LogoCompositor + PromptEngine)
    - High-fidelity Mock Renderer for development, CI/CD, and environments without GPU
    """

    def __init__(self):
        self.use_mock = settings.USE_MOCK_INFERENCE
        self._renderer = None
        self._prompt_engine = None
        self._logo_compositor = None
        self._initialized = False

    def _init_real_engine(self):
        """Lazy load deep learning pipelines when not in mock mode."""
        if self._initialized:
            return

        try:
            from src.pipeline import PackVisionRenderer
            from src.prompt_engine import build_packaging_prompt
            from src.postprocessor import LogoCompositor

            logger.info("Initializing PackVision deep learning inference pipeline...")
            self._renderer = PackVisionRenderer(
                base_model=settings.SDXL_BASE_MODEL,
                controlnet_model=settings.CONTROLNET_MODEL,
                device=settings.MODEL_DEVICE,
            )
            self._prompt_engine = build_packaging_prompt
            self._logo_compositor = LogoCompositor()
            self._initialized = True
            logger.info("PackVision deep learning pipeline initialized successfully.")
        except ImportError as e:
            logger.warning(
                f"Real inference dependencies not available ({str(e)}). Falling back to Mock Inference engine."
            )
            self.use_mock = True

    def generate(
        self,
        input_sketch_path: Path,
        package_type: str,
        material: Optional[str] = "cardboard",
        finish: Optional[str] = "matte",
        viewpoint: Optional[str] = "front",
        background: Optional[str] = "studio_clean",
        user_prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        reference_image_path: Optional[Path] = None,
        logo_image_path: Optional[Path] = None,
        logo_placement: Optional[Dict[str, Any]] = None,
        num_outputs: int = 1,
        seed: Optional[int] = None,
        guidance_scale: float = 7.5,
        controlnet_scale: float = 0.85,
        steps: int = 30,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute end-to-end rendering pipeline.
        
        Returns a list of dictionaries with:
        - "image": PIL.Image
        - "metrics": dict of performance/quality metrics
        """
        start_time = time.time()

        if seed is None:
            seed = random.randint(1000, 999999)

        if not self.use_mock:
            self._init_real_engine()

        if not self.use_mock and self._renderer is not None:
            return self._run_real_inference(
                input_sketch_path=input_sketch_path,
                package_type=package_type,
                material=material,
                finish=finish,
                viewpoint=viewpoint,
                background=background,
                user_prompt=user_prompt,
                negative_prompt=negative_prompt,
                reference_image_path=reference_image_path,
                logo_image_path=logo_image_path,
                logo_placement=logo_placement,
                num_outputs=num_outputs,
                seed=seed,
                guidance_scale=guidance_scale,
                controlnet_scale=controlnet_scale,
                steps=steps,
                progress_callback=progress_callback,
            )
        else:
            return self._run_mock_inference(
                input_sketch_path=input_sketch_path,
                package_type=package_type,
                material=material,
                finish=finish,
                viewpoint=viewpoint,
                background=background,
                user_prompt=user_prompt,
                logo_image_path=logo_image_path,
                logo_placement=logo_placement,
                num_outputs=num_outputs,
                seed=seed,
                progress_callback=progress_callback,
            )

    def _run_mock_inference(
        self,
        input_sketch_path: Path,
        package_type: str,
        material: Optional[str],
        finish: Optional[str],
        viewpoint: Optional[str],
        background: Optional[str],
        user_prompt: Optional[str],
        logo_image_path: Optional[Path],
        logo_placement: Optional[Dict[str, Any]],
        num_outputs: int,
        seed: int,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """High-fidelity mock renderer that produces realistic packaging mockups."""
        logger.info(
            f"Executing Mock Inference: package={package_type}, material={material}, finish={finish}, seed={seed}"
        )

        if progress_callback:
            progress_callback(40, "Preparing packaging geometry and lighting environment...")
        time.sleep(0.15)

        # Load input sketch
        sketch = Image.open(input_sketch_path).convert("RGBA")
        target_size = (1024, 1024)
        sketch_resized = ImageOps.fit(sketch, target_size, Image.Resampling.LANCZOS)

        results = []
        for i in range(num_outputs):
            current_seed = seed + i
            rng = random.Random(current_seed)

            if progress_callback:
                pct = 40 + int(35 * ((i + 1) / num_outputs))
                progress_callback(
                    pct,
                    f"Synthesizing photorealistic {material or 'paper'} render variant {i + 1}/{num_outputs}...",
                )

            # 1. Base Studio Canvas & Background
            canvas = self._create_mock_background(target_size, background, rng)

            # 2. Material Shading Layer
            material_layer = self._render_mock_packaging_body(
                target_size, package_type, material, finish, rng
            )

            # 3. Blend sketch contours with material shading
            rendered = Image.alpha_composite(canvas, material_layer)

            # Extract sketch edges (invert if light background)
            sketch_gray = sketch_resized.convert("L")
            sketch_edges = sketch_gray.filter(ImageFilter.FIND_EDGES)
            sketch_overlay = Image.new("RGBA", target_size, (20, 20, 25, 0))
            # subtle sketch contour blending
            alpha_mask = sketch_edges.point(lambda p: int(min(180, p * 0.4)))
            sketch_overlay.putalpha(alpha_mask)
            rendered = Image.alpha_composite(rendered, sketch_overlay)

            # 4. Logo Compositing (if logo provided)
            if logo_image_path and Path(logo_image_path).exists():
                rendered = self._apply_mock_logo_compositing(
                    rendered, logo_image_path, package_type, finish, logo_placement, rng
                )

            # 5. Final Material Gloss & Specular Highlights
            rendered = self._apply_mock_highlights(rendered, finish, rng)

            # Convert to RGB
            final_img = rendered.convert("RGB")

            metrics = {
                "inference_time_ms": round(rng.uniform(280.0, 450.0), 2),
                "fidelity_score": round(rng.uniform(0.92, 0.98), 3),
                "resolution": f"{target_size[0]}x{target_size[1]}",
                "seed": current_seed,
                "package_type": package_type,
                "material": material or "cardboard",
                "finish": finish or "matte",
                "is_mock": True,
            }

            results.append({"image": final_img, "metrics": metrics})

        return results

    def _create_mock_background(
        self, size: Tuple[int, int], background: Optional[str], rng: random.Random
    ) -> Image.Image:
        """Create photographic background canvas with gradient lighting."""
        w, h = size
        base = Image.new("RGBA", size, (245, 245, 248, 255))
        draw = ImageDraw.Draw(base)

        if background == "retail_shelf":
            # Soft warm ambient shelf gradient
            for y in range(h):
                ratio = y / h
                r = int(220 + 25 * ratio)
                g = int(215 + 20 * ratio)
                b = int(210 + 20 * ratio)
                draw.line([(0, y), (w, y)], fill=(r, g, b, 255))
            # Shelf shadow
            draw.rectangle([(0, int(h * 0.82)), (w, h)], fill=(185, 180, 175, 255))
        elif background == "minimalist_gradient":
            # Minimalist cool gradient
            for y in range(h):
                ratio = y / h
                r = int(230 - 30 * ratio)
                g = int(235 - 25 * ratio)
                b = int(245 - 20 * ratio)
                draw.line([(0, y), (w, y)], fill=(r, g, b, 255))
        else:
            # Studio Clean: soft radial vignette
            cx, cy = w // 2, int(h * 0.45)
            max_r = math.sqrt(cx**2 + cy**2)
            for y in range(0, h, 2):
                for x in range(0, w, 2):
                    dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                    factor = min(1.0, dist / max_r)
                    lum = int(250 - 45 * factor)
                    draw.point((x, y), fill=(lum, lum, lum + 2, 255))

        # Table surface ground reflection/shadow
        shadow = Image.new("RGBA", size, (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(shadow)
        sdraw.ellipse(
            [
                (int(w * 0.2), int(h * 0.78)),
                (int(w * 0.8), int(h * 0.88)),
            ],
            fill=(40, 40, 45, 75),
        )
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=25))
        return Image.alpha_composite(base, shadow)

    def _render_mock_packaging_body(
        self,
        size: Tuple[int, int],
        package_type: str,
        material: Optional[str],
        finish: Optional[str],
        rng: random.Random,
    ) -> Image.Image:
        """Render packaging shape with material color palette and gradient lighting."""
        w, h = size
        layer = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        # Palette selection based on material
        mat = (material or "cardboard").lower()
        if "glass" in mat:
            base_col = (200, 225, 230, 200)
            highlight_col = (245, 255, 255, 220)
            shadow_col = (140, 175, 185, 220)
        elif "aluminum" in mat or "metallic" in (finish or ""):
            base_col = (195, 200, 205, 255)
            highlight_col = (250, 252, 255, 255)
            shadow_col = (120, 125, 130, 255)
        elif "kraft" in mat or "cardboard" in mat:
            base_col = (205, 165, 125, 255)
            highlight_col = (225, 185, 145, 255)
            shadow_col = (150, 115, 80, 255)
        elif "plastic" in mat:
            base_col = (235, 235, 238, 255)
            highlight_col = (255, 255, 255, 255)
            shadow_col = (170, 172, 178, 255)
        else:
            base_col = (210, 210, 215, 255)
            highlight_col = (240, 240, 245, 255)
            shadow_col = (140, 140, 145, 255)

        # Draw package body based on package_type
        pkg = (package_type or "bottle").lower()
        if "box" in pkg:
            # 3D isometric box
            left, top, right, bottom = int(w * 0.28), int(h * 0.25), int(w * 0.72), int(h * 0.8)
            draw.rounded_rectangle(
                [(left, top), (right, bottom)], radius=16, fill=base_col
            )
            # Subtle side shadow
            draw.rectangle(
                [(int(w * 0.6), top), (right, bottom)], fill=shadow_col
            )
        elif "pouch" in pkg:
            # Standup pouch
            points = [
                (int(w * 0.35), int(h * 0.22)),
                (int(w * 0.65), int(h * 0.22)),
                (int(w * 0.74), int(h * 0.76)),
                (int(w * 0.26), int(h * 0.76)),
            ]
            draw.polygon(points, fill=base_col)
            # Pouch seal at top
            draw.rectangle(
                [(int(w * 0.33), int(h * 0.18)), (int(w * 0.67), int(h * 0.22))],
                fill=highlight_col,
            )
        elif "can" in pkg:
            # Aluminum beverage can
            draw.rounded_rectangle(
                [(int(w * 0.34), int(h * 0.22)), (int(w * 0.66), int(h * 0.78))],
                radius=32,
                fill=base_col,
            )
        else:
            # Default / Bottle shape
            # Neck
            draw.rounded_rectangle(
                [(int(w * 0.44), int(h * 0.15)), (int(w * 0.56), int(h * 0.32))],
                radius=8,
                fill=highlight_col,
            )
            # Cap
            draw.rounded_rectangle(
                [(int(w * 0.43), int(h * 0.12)), (int(w * 0.57), int(h * 0.16))],
                radius=4,
                fill=(45, 45, 50, 255),
            )
            # Main bottle body
            draw.rounded_rectangle(
                [(int(w * 0.32), int(h * 0.30)), (int(w * 0.68), int(h * 0.80))],
                radius=28,
                fill=base_col,
            )

        return layer

    def _apply_mock_logo_compositing(
        self,
        base_render: Image.Image,
        logo_path: Path,
        package_type: str,
        finish: Optional[str],
        logo_placement: Optional[Dict[str, Any]],
        rng: random.Random,
    ) -> Image.Image:
        """Compose logo onto rendered packaging surface with realistic perspective and blending."""
        try:
            logo = Image.open(logo_path).convert("RGBA")
            bw, bh = base_render.size

            # Determine scale & position
            scale_factor = (
                logo_placement.get("scale", 0.25) if logo_placement else 0.25
            )
            target_w = int(bw * scale_factor)
            aspect = logo.height / max(1, logo.width)
            target_h = int(target_w * aspect)

            logo_resized = logo.resize((target_w, target_h), Image.Resampling.LANCZOS)

            # Center on the package body
            pos_x = int((bw - target_w) / 2)
            pos_y = int(bh * 0.45)

            if logo_placement:
                if "x" in logo_placement:
                    pos_x = int(bw * logo_placement["x"])
                if "y" in logo_placement:
                    pos_y = int(bh * logo_placement["y"])

            # Emboss/foil finish effect
            if finish and ("foil" in finish or "metallic" in finish):
                enhancer = ImageEnhance.Brightness(logo_resized)
                logo_resized = enhancer.enhance(1.15)

            logo_layer = Image.new("RGBA", base_render.size, (0, 0, 0, 0))
            logo_layer.paste(logo_resized, (pos_x, pos_y), mask=logo_resized)

            return Image.alpha_composite(base_render, logo_layer)
        except Exception as e:
            logger.error(f"Error compositing mock logo: {e}")
            return base_render

    def _apply_mock_highlights(
        self, image: Image.Image, finish: Optional[str], rng: random.Random
    ) -> Image.Image:
        """Add realistic specular highlights and surface reflection sheen."""
        w, h = image.size
        fin = (finish or "matte").lower()
        if "glossy" not in fin and "metallic" not in fin and "glass" not in fin:
            return image

        highlight_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(highlight_layer)

        # Cylindrical/vertical highlight streak down the left third
        hx1 = int(w * 0.38)
        hx2 = int(w * 0.42)
        hy1 = int(h * 0.26)
        hy2 = int(h * 0.74)

        alpha = 80 if "glossy" in fin else 120
        draw.rounded_rectangle(
            [(hx1, hy1), (hx2, hy2)],
            radius=12,
            fill=(255, 255, 255, alpha),
        )

        highlight_blurred = highlight_layer.filter(ImageFilter.GaussianBlur(radius=10))
        return Image.alpha_composite(image, highlight_blurred)

    def _run_real_inference(
        self,
        input_sketch_path: Path,
        package_type: str,
        material: Optional[str],
        finish: Optional[str],
        viewpoint: Optional[str],
        background: Optional[str],
        user_prompt: Optional[str],
        negative_prompt: Optional[str],
        reference_image_path: Optional[Path],
        logo_image_path: Optional[Path],
        logo_placement: Optional[Dict[str, Any]],
        num_outputs: int,
        seed: int,
        guidance_scale: float,
        controlnet_scale: float,
        steps: int,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Run deep learning pipeline using PackVisionRenderer & LogoCompositor."""
        if progress_callback:
            progress_callback(30, "Compiling engineering prompt and conditioning tensors...")

        # 1. Build prompt
        full_prompt = self._prompt_engine(
            package_type=package_type,
            material=material,
            finish=finish,
            viewpoint=viewpoint,
            background=background,
            user_prompt=user_prompt,
        )

        if progress_callback:
            progress_callback(45, "Running ControlNet + SDXL diffusion model...")

        # 2. Diffusion rendering
        generated_images = self._renderer.render(
            sketch_path=input_sketch_path,
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            num_images=num_outputs,
            seed=seed,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_scale,
            num_inference_steps=steps,
        )

        results = []
        for i, img in enumerate(generated_images):
            if logo_image_path and self._logo_compositor:
                if progress_callback:
                    progress_callback(
                        80, f"Compositing vector logo onto packaging render {i + 1}..."
                    )
                img = self._logo_compositor.composite(
                    rendered_image=img,
                    logo_path=logo_image_path,
                    placement=logo_placement,
                )

            metrics = {
                "steps": steps,
                "guidance_scale": guidance_scale,
                "controlnet_scale": controlnet_scale,
                "seed": seed + i,
                "prompt": full_prompt,
                "is_mock": False,
            }
            results.append({"image": img, "metrics": metrics})

        return results


inference_service = InferenceService()
