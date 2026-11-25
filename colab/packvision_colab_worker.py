"""
PackVision AI - Google Colab GPU Worker Server
===============================================
Run this script inside a free Google Colab GPU instance to provide a remote
Stable Diffusion 1.5 + ControlNet Canny GPU worker for your local PackVision AI studio.

How to use:
1. Open Google Colab (https://colab.research.google.com)
2. Switch runtime to T4 GPU: Runtime -> Change runtime type -> T4 GPU.
3. Install dependencies:
   !pip install diffusers transformers accelerate torch torchvision fastapi uvicorn pyngrok pydantic pillow
4. Run this script:
   !python packvision_colab_worker.py --port 8000 --ngrok-token YOUR_TOKEN
   (Or use localtunnel: npx localtunnel --port 8000)
5. Copy the generated public URL (e.g. https://xxxx.ngrok-free.app) and paste it
   into your local PackVision AI Studio Settings -> "Google Colab URL".
"""

import io
import os
import gc
import base64
import argparse
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import torch
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)

app = FastAPI(title="PackVision AI - Colab GPU Worker")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipe = None
device = "cuda" if torch.cuda.is_available() else "cpu"


class RenderPayload(BaseModel):
    image: str  # Base64 encoded PNG/JPG edge map
    prompt: str
    negative_prompt: Optional[str] = "distorted geometry, warped text, ugly, blurry, deformed edges"
    steps: int = 30
    guidance_scale: float = 7.5
    controlnet_scale: float = 0.85
    seed: Optional[int] = None
    width: int = 512
    height: int = 512


@app.on_event("startup")
def load_models():
    global pipe
    print(f"[Colab Worker] Loading Stable Diffusion 1.5 + ControlNet Canny on {device}...")
    cnet_id = "lllyasviel/control_v11p_sd15_canny"
    base_id = "runwayml/stable-diffusion-v1-5"

    controlnet = ControlNetModel.from_pretrained(
        cnet_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_id,
        controlnet=controlnet,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    if device == "cuda":
        pipe.enable_attention_slicing()
    print("[Colab Worker] Models loaded and ready for rendering.")


@app.get("/health")
def health():
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram_mb = int(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)) if torch.cuda.is_available() else 0
    return {
        "status": "online",
        "service": "PackVision Colab GPU Worker",
        "gpu": gpu_name,
        "vram_mb": vram_mb,
        "models_loaded": pipe is not None,
    }


@app.post("/render")
def render(payload: RenderPayload):
    if pipe is None:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    try:
        # Decode base64 conditioning edge image
        img_bytes = base64.b64decode(payload.image)
        edge_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        edge_img = edge_img.resize((payload.width, payload.height), Image.Resampling.LANCZOS)

        generator = None
        if payload.seed is not None:
            generator = torch.Generator(device=device).manual_seed(payload.seed)

        print(f"[Colab Worker] Rendering: {payload.width}x{payload.height}, {payload.steps} steps...")
        output = pipe(
            prompt=payload.prompt,
            negative_prompt=payload.negative_prompt,
            image=edge_img,
            num_inference_steps=payload.steps,
            guidance_scale=payload.guidance_scale,
            controlnet_conditioning_scale=payload.controlnet_scale,
            generator=generator,
        )

        rendered_img = output.images[0]

        # Convert back to base64
        buf = io.BytesIO()
        rendered_img.save(buf, format="PNG")
        b64_out = base64.b64encode(buf.getvalue()).decode("utf-8")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        return {"success": True, "image": b64_out}
    except Exception as e:
        print(f"[Colab Worker] Error during render: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
