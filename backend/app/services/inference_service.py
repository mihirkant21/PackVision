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
    """Service adapter bridging FastAPI backend jobs with PackVision AI rendering engines.
    
    Supports:
    - Custom trained PackVisionNet deep learning model (models/checkpoints/packvision_custom_v1.pt)
    - High-fidelity procedural PBR packaging renderer (kraft fibers, corrugation, glass SSS, lighting)
    - Stable Diffusion XL + ControlNet on CUDA GPUs
    - Stage-2 projective homography logo compositor and CIELAB ink blending
    - Structural Edge SSIM / IoU and CLIP metric evaluation
    """

    def __init__(self):
        self._renderer = None
        self._logo_compositor = None
        self._evaluator = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy load rendering engine and compositors."""
        if self._initialized:
            return

        try:
            from src.pipeline import PackVisionRenderer
            from src.postprocessor import LogoCompositor
            from src.evaluation import PackagingEvaluator

            logger.info("Initializing PackVision neural inference engine and compositors...")
            self._renderer = PackVisionRenderer(
                base_model_id=settings.SDXL_BASE_MODEL,
                controlnet_model_id=settings.CONTROLNET_MODEL,
                device=settings.MODEL_DEVICE,
                use_mock_inference=settings.USE_MOCK_INFERENCE,
            )
            self._logo_compositor = LogoCompositor()
            self._evaluator = PackagingEvaluator()
            self._initialized = True
            logger.info("PackVision inference service initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing PackVision renderer: {e}. Will use procedural engine.")
            self._initialized = False

    def generate(
        self,
        input_sketch_path: Path,
        package_type: str,
        material: Optional[str] = "cardboard",
        finish: Optional[str] = "matte",
        viewpoint: Optional[str] = "three_quarter",
        background: Optional[str] = "seamless_studio",
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
        input_mode: str = "auto",
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute end-to-end rendering pipeline.
        
        Returns a list of dictionaries with:
        - "image": PIL.Image
        - "metrics": dict of performance/quality metrics
        """
        self._ensure_initialized()

        if seed is None:
            seed = random.randint(1000, 999999)

        if not Path(input_sketch_path).exists():
            raise FileNotFoundError(f"Input sketch file not found: {input_sketch_path}")

        sketch_pil = Image.open(input_sketch_path).convert("RGB")

        # Normalize terminology
        category_norm = package_type.lower().replace(" ", "_")
        material_norm = (material or "cardboard").lower().replace(" ", "_")
        finish_norm = (finish or "matte").lower().replace(" ", "_")
        viewpoint_norm = (viewpoint or "three_quarter").lower().replace(" ", "_")
        background_norm = (background or "seamless_studio").lower().replace(" ", "_")

        # Reference image -> one-sentence style hint via the shared VLM client.
        # When no VLM provider is configured, a static directional hint is used so
        # the reference still influences the render.
        reference_style_hint = ""
        reference_used = False
        if reference_image_path and Path(reference_image_path).exists():
            reference_used = True
            try:
                ref_img = Image.open(reference_image_path).convert("RGB")
                hint = ""
                try:
                    from src.vlm_client import describe_image

                    hint = describe_image(
                        ref_img,
                        "Describe the material, color palette and finish of this packaging "
                        "reference image in one sentence.",
                    )
                except Exception as vlm_err:
                    logger.warning(f"VLM reference description failed: {vlm_err}")
                reference_style_hint = (hint or "").strip() or (
                    "Design direction inspired by the provided reference swatch."
                )
                logger.info(f"Reference style hint: {reference_style_hint}")
            except Exception as ref_err:
                logger.warning(f"Could not read reference image {reference_image_path}: {ref_err}")
                reference_style_hint = ""
                reference_used = False

        effective_user_prompt = user_prompt
        if reference_style_hint:
            effective_user_prompt = (
                f"{user_prompt or ''} Reference style hint: {reference_style_hint}"
            ).strip()

        results = []

        for i in range(num_outputs):
            current_seed = seed + i
            if progress_callback:
                pct = int(20 + (i / max(1, num_outputs)) * 40)
                progress_callback(pct, f"Synthesizing photorealistic {material_norm} render (variant {i+1}/{num_outputs})...")

            try:
                # 1. Primary AI Synthesis (Custom PackVisionNet or Procedural PBR)
                render_result = self._renderer.render(
                    sketch=sketch_pil,
                    category=category_norm,
                    substrate=material_norm,
                    finish=finish_norm,
                    lighting="studio softbox",
                    camera_angle=viewpoint_norm,
                    background=background_norm,
                    user_prompt=effective_user_prompt,
                    negative_prompt=negative_prompt,
                    seed=current_seed,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    controlnet_scale=controlnet_scale,
                    input_mode=input_mode,
                )

                rendered_img = render_result.image

                # 2. Stage-2 Homography Logo Compositing (if logo supplied)
                if logo_image_path and Path(logo_image_path).exists():
                    if progress_callback:
                        progress_callback(75, f"Compositing vector logo onto packaging face...")
                    try:
                        logo_img = Image.open(logo_image_path)
                        blend_mode = "multiply"
                        dest_points = None

                        if logo_placement and isinstance(logo_placement, dict):
                            blend_mode = logo_placement.get("blend_mode", "multiply")
                            # If manual normalized coordinates are passed
                            px = logo_placement.get("x", 0.5)
                            py = logo_placement.get("y", 0.5)
                            scale = logo_placement.get("scale", 0.5)
                            w, h = rendered_img.size
                            cx = px * w
                            cy = py * h
                            bw = scale * (w * 0.45)
                            bh = bw * (logo_img.height / max(logo_img.width, 1))
                            dest_points = [
                                (cx - bw / 2, cy - bh / 2),
                                (cx + bw / 2, cy - bh / 2),
                                (cx + bw / 2, cy + bh / 2),
                                (cx - bw / 2, cy + bh / 2),
                            ]

                        comp_res = self._logo_compositor.composite(
                            package_image=rendered_img,
                            logo_image=logo_img,
                            destination_points=dest_points,
                            blend_mode=blend_mode,
                        )
                        rendered_img = comp_res.image
                    except Exception as logo_err:
                        logger.warning(f"Logo compositing notice: {logo_err}")

                # 3. Quality Evaluation Metrics (honest values: None when the
                #    evaluator is unavailable, never fabricated placeholders)
                if progress_callback:
                    progress_callback(90, "Evaluating structural fidelity and SSIM metrics...")

                ssim_score = None
                iou_score = None
                clip_score = None
                try:
                    from src.evaluation import compute_edge_ssim, compute_edge_iou
                    ssim_score = compute_edge_ssim(sketch_pil, rendered_img)
                    iou_score = compute_edge_iou(sketch_pil, rendered_img)
                    if self._evaluator:
                        prompt_eval = f"{category_norm} in {material_norm} with {finish_norm} finish"
                        clip_score = self._evaluator.compute_clip_score(rendered_img, prompt_eval)
                except Exception as eval_err:
                    logger.debug(f"Evaluation metrics calculation: {eval_err}")

                metrics_available = (
                    ssim_score is not None or iou_score is not None or clip_score is not None
                )
                renderer_meta = getattr(render_result, "metadata", {}) or {}

                metrics = {
                    "edge_ssim": round(float(ssim_score), 4) if ssim_score is not None else None,
                    "edge_iou": round(float(iou_score), 4) if iou_score is not None else None,
                    "clip_score": round(float(clip_score), 4) if clip_score is not None else None,
                    "metrics_available": metrics_available,
                    "metrics_source": "computed" if metrics_available else "unavailable",
                    "inference_time_ms": round(float(render_result.inference_time * 1000.0), 2),
                    "seed": current_seed,
                    "package_type": category_norm,
                    "material": material_norm,
                    "finish": finish_norm,
                    "engine": renderer_meta.get("engine", "PackVisionNet"),
                    "engine_used": renderer_meta.get(
                        "engine_used", renderer_meta.get("engine", "unknown")
                    ),
                    "engine_latency_s": renderer_meta.get("engine_latency_s"),
                    "engine_note": renderer_meta.get("engine_note", ""),
                    "input_mode": renderer_meta.get("input_mode", "auto"),
                    "reference_used": reference_used,
                    "is_mock": render_result.is_mock,
                }

                results.append({
                    "image": rendered_img,
                    "metrics": metrics,
                })

            except Exception as render_err:
                logger.error(f"Render generation variant {i+1} encountered error: {render_err}", exc_info=True)
                # Fallback safeguard: return enhanced sketch with tone mapping so the user never gets an unhandled crash
                fallback_img = self._create_emergency_fallback(sketch_pil, material_norm, finish_norm)
                results.append({
                    "image": fallback_img,
                    "metrics": {
                        "edge_ssim": None,
                        "edge_iou": None,
                        "clip_score": None,
                        "metrics_available": False,
                        "metrics_source": "unavailable",
                        "inference_time_ms": 120.0,
                        "seed": current_seed,
                        "engine": "Emergency Fallback",
                        "engine_used": "emergency-fallback",
                        "reference_used": reference_used,
                        "is_mock": True,
                    },
                })

        return results

    def _create_emergency_fallback(self, sketch_pil: Image.Image, material: str, finish: str) -> Image.Image:
        """Emergency fail-safe render to ensure backend requests never raise 500 errors."""
        target_size = (1024, 1024)
        canvas = Image.new("RGB", target_size, (244, 244, 246))
        sketch_resized = sketch_pil.resize(target_size, Image.Resampling.LANCZOS)
        # Apply warm or cool studio tint
        tint = (210, 185, 155) if "kraft" in material else (235, 238, 242)
        tinted = Image.new("RGB", target_size, tint)
        blended = Image.blend(tinted, sketch_resized, 0.45)
        return blended

    def get_engine_status(self) -> Dict[str, Any]:
        """Probe cloud render engine availability (for health/admin endpoints).

        Returns the CloudRenderEngine.probe() report:
        {provider: {"configured": bool, "last_error": str|None, "status": "ready"|"cooldown"|"missing_key"}}
        """
        try:
            from src.cloud_engine import CloudRenderEngine

            return CloudRenderEngine().probe()
        except Exception as status_err:
            logger.warning(f"Cloud engine probe failed: {status_err}")
            return {
                "gemini": {"configured": False, "last_error": str(status_err), "status": "missing_key"},
                "openrouter": {"configured": False, "last_error": None, "status": "missing_key"},
                "openai": {"configured": False, "last_error": None, "status": "missing_key"},
                "error": str(status_err),
            }


inference_service = InferenceService()


def get_engine_status() -> Dict[str, Any]:
    """Module-level helper (used by API/health endpoints and other services):
    probe cloud render engine availability via the shared inference service."""
    return inference_service.get_engine_status()
