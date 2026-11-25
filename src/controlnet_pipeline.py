"""
PackVision AI - Stable Diffusion 1.5 + ControlNet Canny Inference Pipeline
==========================================================================
Genuine local neural inference pipeline tailored for hardware-constrained systems
(such as NVIDIA GTX 1650 Ti with 4GB VRAM) and optional Google Colab GPU bridge.

Inference Strategy:
1. Local CUDA GPU (SD 1.5 + ControlNet Canny in FP16 with attention slicing & CPU offload).
2. Remote Google Colab GPU Bridge (when COLAB_INFERENCE_URL is configured by user).
3. Clearly labeled procedural PBR packaging preview fallback when no GPU/weights are loaded.
"""

from __future__ import annotations

import gc
import io
import os
import time
import base64
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from backend.app.core.config import settings

logger = logging.getLogger("packvision.controlnet")


@dataclass
class HardwareProfile:
    has_cuda: bool
    device_name: str
    vram_total_mb: int
    vram_free_mb: int
    is_low_vram: bool
    recommended_resolution: int
    recommended_steps: int
    status_summary: str


def get_hardware_profile() -> HardwareProfile:
    """Inspects host machine GPU, VRAM, and returns hardware-aware defaults."""
    if not HAS_TORCH or not torch.cuda.is_available():
        return HardwareProfile(
            has_cuda=False,
            device_name="CPU Only",
            vram_total_mb=0,
            vram_free_mb=0,
            is_low_vram=True,
            recommended_resolution=512,
            recommended_steps=20,
            status_summary="No NVIDIA CUDA GPU detected in current Python environment. Local neural inference will require CPU or Colab GPU bridge.",
        )

    try:
        device_idx = 0
        device_name = torch.cuda.get_device_name(device_idx)
        props = torch.cuda.get_device_properties(device_idx)
        total_vram_mb = int(props.total_memory / (1024 * 1024))
        free_vram_mb = int(torch.cuda.mem_get_info()[0] / (1024 * 1024))
        is_low_vram = total_vram_mb <= 6144  # 6GB or less (e.g. GTX 1650 Ti 4GB)

        return HardwareProfile(
            has_cuda=True,
            device_name=device_name,
            vram_total_mb=total_vram_mb,
            vram_free_mb=free_vram_mb,
            is_low_vram=is_low_vram,
            recommended_resolution=512 if is_low_vram else 768,
            recommended_steps=25 if is_low_vram else 35,
            status_summary=f"CUDA GPU detected: {device_name} ({total_vram_mb} MB VRAM). Low-VRAM optimizations: {'ACTIVE' if is_low_vram else 'OFF'}.",
        )
    except Exception as e:
        logger.warning(f"Error querying CUDA hardware: {e}")
        return HardwareProfile(
            has_cuda=False,
            device_name="CUDA Query Failed",
            vram_total_mb=0,
            vram_free_mb=0,
            is_low_vram=True,
            recommended_resolution=512,
            recommended_steps=20,
            status_summary=f"CUDA detection error: {str(e)}",
        )


class LocalControlNetPipeline:
    """
    Manages loading, caching, and inference of Stable Diffusion 1.5 + ControlNet Canny.
    """

    SD15_BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
    CONTROLNET_CANNY_MODEL_ID = "lllyasviel/control_v11p_sd15_canny"

    def __init__(self):
        self.pipe = None
        self._loaded = False
        self._load_error = None
        self.hardware = get_hardware_profile()

    def is_available(self) -> bool:
        """Returns True if the pipeline is loaded or can be loaded."""
        return self._loaded or (self.hardware.has_cuda and HAS_TORCH)

    def load_pipeline(
        self,
        base_model_id: Optional[str] = None,
        controlnet_id: Optional[str] = None,
    ) -> bool:
        """Loads SD 1.5 + ControlNet Canny with memory-efficient settings."""
        if self._loaded and self.pipe is not None:
            return True

        if not HAS_TORCH or not self.hardware.has_cuda:
            self._load_error = "CUDA-enabled PyTorch is required for local GPU inference."
            return False

        try:
            from diffusers import (
                StableDiffusionControlNetPipeline,
                ControlNetModel,
                UniPCMultistepScheduler,
            )

            base_id = base_model_id or self.SD15_BASE_MODEL_ID
            cnet_id = controlnet_id or self.CONTROLNET_CANNY_MODEL_ID

            logger.info(f"Loading ControlNet Canny model: {cnet_id} (fp16)...")
            controlnet = ControlNetModel.from_pretrained(
                cnet_id,
                torch_dtype=torch.float16,
            )

            logger.info(f"Loading Stable Diffusion 1.5 base pipeline: {base_id} (fp16)...")
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                base_id,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                safety_checker=None,
            )

            # High-speed multi-step scheduler
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

            # Low-VRAM Optimizations for 4GB GPUs (GTX 1650 Ti)
            if self.hardware.is_low_vram:
                logger.info("Applying 4GB VRAM optimizations: model_cpu_offload + attention_slicing...")
                self.pipe.enable_model_cpu_offload()
                self.pipe.enable_attention_slicing(slice_size="auto")
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass
            else:
                self.pipe.to("cuda")

            self._loaded = True
            self._load_error = None
            logger.info("Local SD 1.5 + ControlNet Canny pipeline initialized successfully.")
            return True

        except Exception as e:
            self._load_error = str(e)
            logger.error(f"Failed to load local SD 1.5 + ControlNet pipeline: {e}", exc_info=True)
            self._loaded = False
            return False

    def generate(
        self,
        edge_image: Image.Image,
        prompt: str,
        negative_prompt: str,
        steps: int = 25,
        guidance_scale: float = 7.5,
        controlnet_scale: float = 0.85,
        seed: Optional[int] = None,
        width: int = 512,
        height: int = 512,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Image.Image:
        """Executes local neural diffusion pass conditioned on Canny edge map."""
        if not self._loaded:
            if not self.load_pipeline():
                raise RuntimeError(f"Local pipeline not available: {self._load_error}")

        import torch

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        # Ensure edge image is RGB PIL of target dimension
        cond_img = edge_image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)

        def step_callback(pipe, step_idx, timestep, callback_kwargs):
            if progress_callback:
                progress_callback(step_idx, steps, f"Diffusion step {step_idx}/{steps}")
            return callback_kwargs

        logger.info(f"Running SD 1.5 + ControlNet inference: {width}x{height}, {steps} steps, scale={controlnet_scale}...")
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=cond_img,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_scale,
            generator=generator,
            callback_on_step_end=step_callback,
        )

        # Free GPU memory cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        return result.images[0]


class ColabRemoteBridge:
    """
    Connects to an optional remote Google Colab GPU instance configured by the user.
    """

    def __init__(self, colab_url: Optional[str] = None):
        self.colab_url = (colab_url or os.getenv("COLAB_INFERENCE_URL", "")).rstrip("/")

    def is_configured(self) -> bool:
        return bool(self.colab_url and self.colab_url.startswith("http"))

    def check_health(self) -> Dict[str, Any]:
        """Checks if remote Colab worker is online."""
        if not self.is_configured():
            return {"online": False, "reason": "No COLAB_INFERENCE_URL configured."}

        import requests

        try:
            res = requests.get(f"{self.colab_url}/health", timeout=5.0)
            if res.status_code == 200:
                data = res.json()
                return {"online": True, "gpu": data.get("gpu", "Remote GPU"), "details": data}
            return {"online": False, "reason": f"HTTP {res.status_code}: {res.text}"}
        except Exception as e:
            return {"online": False, "reason": str(e)}

    def generate(
        self,
        edge_image: Image.Image,
        prompt: str,
        negative_prompt: str,
        steps: int = 30,
        guidance_scale: float = 7.5,
        controlnet_scale: float = 0.85,
        seed: Optional[int] = None,
        width: int = 512,
        height: int = 512,
    ) -> Image.Image:
        """Sends inference request to the remote Colab GPU worker."""
        import requests

        # Base64 encode the edge image
        buf = io.BytesIO()
        edge_image.convert("RGB").save(buf, format="PNG")
        b64_image = base64.b64encode(buf.getvalue()).decode("utf-8")

        payload = {
            "image": b64_image,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "controlnet_scale": controlnet_scale,
            "seed": seed,
            "width": width,
            "height": height,
        }

        url = f"{self.colab_url}/render"
        logger.info(f"Dispatching inference to Colab GPU: {url}")
        res = requests.post(url, json=payload, timeout=120.0)
        if res.status_code != 200:
            raise RuntimeError(f"Colab GPU error ({res.status_code}): {res.text}")

        res_data = res.json()
        out_b64 = res_data.get("image")
        if not out_b64:
            raise RuntimeError("Colab GPU response missing 'image' data.")

        img_bytes = base64.b64decode(out_b64)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")


# Global singletons
local_controlnet = LocalControlNetPipeline()
colab_bridge = ColabRemoteBridge()
