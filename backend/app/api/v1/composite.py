import io
import uuid
import logging
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import cv2

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.db.session import get_db, DEFAULT_SYSTEM_USER_ID
from backend.app.db.models import Asset
from backend.app.core.config import settings
from backend.app.services.asset_service import AssetService
from src.postprocessor import LogoCompositor

logger = logging.getLogger(__name__)

router = APIRouter()


class CompositeRequest(BaseModel):
    base_asset_id: str = Field(..., description="UUID of rendered packaging or background image")
    logo_asset_id: str = Field(..., description="UUID of logo or graphic artwork asset")
    x: float = Field(default=0.5, ge=0.0, le=1.0, description="Center horizontal position ratio (0.0 to 1.0)")
    y: float = Field(default=0.5, ge=0.0, le=1.0, description="Center vertical position ratio (0.0 to 1.0)")
    scale: float = Field(default=0.35, ge=0.05, le=1.5, description="Scale relative to base image width")
    rotation: float = Field(default=0.0, ge=-180.0, le=180.0, description="Rotation angle in degrees")
    opacity: float = Field(default=1.0, ge=0.0, le=1.0, description="Logo opacity (0.0 to 1.0)")
    blend_mode: str = Field(default="normal", description="Blend mode: normal, multiply, overlay, hard_light")
    use_cielab_texture: bool = Field(default=True, description="Blend with underlying substrate lighting and grain")
    save_as_asset: bool = Field(default=True, description="Persist composite as a new asset in database")


def apply_blend_mode(base: np.ndarray, overlay: np.ndarray, mode: str) -> np.ndarray:
    """Apply blend modes in float32 [0, 1] RGB."""
    b = base.astype(np.float32) / 255.0
    o = overlay.astype(np.float32) / 255.0

    if mode == "multiply":
        result = b * o
    elif mode == "overlay":
        result = np.where(b < 0.5, 2.0 * b * o, 1.0 - 2.0 * (1.0 - b) * (1.0 - o))
    elif mode == "hard_light":
        result = np.where(o < 0.5, 2.0 * b * o, 1.0 - 2.0 * (1.0 - b) * (1.0 - o))
    else:  # normal
        result = o

    return np.clip(result * 255.0, 0, 255).astype(np.uint8)


def composite_logo_on_image(
    base_pil: Image.Image,
    logo_pil: Image.Image,
    x: float,
    y: float,
    scale: float,
    rotation: float,
    opacity: float,
    blend_mode: str,
    use_cielab_texture: bool = True,
) -> Image.Image:
    """Places, transforms, and blends a logo onto a base packaging render."""
    base_w, base_h = base_pil.size
    logo_rgba = logo_pil.convert("RGBA")

    # 1. Calculate scaled logo dimensions preserving aspect ratio
    target_logo_w = max(16, int(base_w * scale))
    aspect_ratio = logo_rgba.height / max(1, logo_rgba.width)
    target_logo_h = max(16, int(target_logo_w * aspect_ratio))

    resized_logo = logo_rgba.resize((target_logo_w, target_logo_h), Image.Resampling.LANCZOS)

    # 2. Rotate logo if requested
    if abs(rotation) > 0.01:
        resized_logo = resized_logo.rotate(
            rotation, resample=Image.Resampling.BICUBIC, expand=True
        )

    # 3. Apply opacity to alpha channel
    if opacity < 1.0:
        r, g, b, a = resized_logo.split()
        a = a.point(lambda p: int(p * opacity))
        resized_logo = Image.merge("RGBA", (r, g, b, a))

    lw, lh = resized_logo.size

    # 4. Compute bounding box coordinates centered at (x * base_w, y * base_h)
    center_px_x = int(x * base_w)
    center_px_y = int(y * base_h)
    paste_x = center_px_x - lw // 2
    paste_y = center_px_y - lh // 2

    # 5. Composite onto a transparent layer matching base image
    overlay_canvas = Image.new("RGBA", (base_w, base_h), (0, 0, 0, 0))
    overlay_canvas.paste(resized_logo, (paste_x, paste_y), mask=resized_logo)

    # Convert to numpy for advanced blending if needed
    base_np = np.array(base_pil.convert("RGB"))
    overlay_np = np.array(overlay_canvas)

    overlay_rgb = overlay_np[:, :, :3]
    alpha_mask = (overlay_np[:, :, 3].astype(np.float32) / 255.0)[:, :, None]

    if blend_mode in ("multiply", "overlay", "hard_light"):
        blended_rgb = apply_blend_mode(base_np, overlay_rgb, blend_mode)
    else:
        blended_rgb = overlay_rgb

    # Substrate texture integration (subtle ambient shadow transmission)
    if use_cielab_texture and np.any(alpha_mask > 0.05):
        gray_base = cv2.cvtColor(base_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        shading = cv2.GaussianBlur(gray_base, (21, 21), 0)
        # Moderate shading multiplier (0.85 to 1.15)
        shading_mod = np.clip(shading / (np.mean(shading) + 1e-6), 0.85, 1.15)[:, :, None]
        blended_rgb = np.clip(blended_rgb.astype(np.float32) * shading_mod, 0, 255).astype(np.uint8)

    # Final alpha composite
    final_np = (base_np.astype(np.float32) * (1.0 - alpha_mask) + blended_rgb.astype(np.float32) * alpha_mask).astype(np.uint8)
    return Image.fromarray(final_np)


@router.post(
    "/composite",
    summary="Composite logo / graphic artwork onto packaging render",
    description="Transforms and deterministically blends brand logos onto the render with position, scale, rotation, opacity, and blend modes.",
)
def create_composite(
    payload: CompositeRequest,
    user_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    # 1. Fetch base asset
    base_asset = db.query(Asset).filter(Asset.id == payload.base_asset_id, Asset.deleted_at.is_(None)).first()
    if not base_asset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Base asset '{payload.base_asset_id}' not found.")
    base_path = AssetService.get_asset_file_path(base_asset)
    if not base_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Base asset file missing from storage.")

    # 2. Fetch logo asset
    logo_asset = db.query(Asset).filter(Asset.id == payload.logo_asset_id, Asset.deleted_at.is_(None)).first()
    if not logo_asset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Logo asset '{payload.logo_asset_id}' not found.")
    logo_path = AssetService.get_asset_file_path(logo_asset)
    if not logo_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Logo asset file missing from storage.")

    base_pil = Image.open(base_path)
    logo_pil = Image.open(logo_path)

    # 3. Composite
    composited_pil = composite_logo_on_image(
        base_pil=base_pil,
        logo_pil=logo_pil,
        x=payload.x,
        y=payload.y,
        scale=payload.scale,
        rotation=payload.rotation,
        opacity=payload.opacity,
        blend_mode=payload.blend_mode.lower(),
        use_cielab_texture=payload.use_cielab_texture,
    )

    # 4. Save result
    composite_id = str(uuid.uuid4())
    comp_dir = settings.base_storage_dir / "composites"
    comp_dir.mkdir(parents=True, exist_ok=True)
    out_path = comp_dir / f"{composite_id}.png"
    composited_pil.save(out_path, format="PNG", quality=95)

    created_asset_id = None
    if payload.save_as_asset:
        # Save as new output asset
        new_asset = Asset(
            id=composite_id,
            user_id=user_id or DEFAULT_SYSTEM_USER_ID,
            storage_key=f"composites/{composite_id}.png",
            file_name=f"composite_{payload.base_asset_id[:8]}_{composite_id[:6]}.png",
            asset_type="output",
            mime_type="image/png",
            width=composited_pil.width,
            height=composited_pil.height,
            file_size=out_path.stat().st_size,
        )
        db.add(new_asset)
        db.commit()
        db.refresh(new_asset)
        created_asset_id = new_asset.id

    return {
        "success": True,
        "composite_id": composite_id,
        "asset_id": created_asset_id,
        "url": f"/api/v1/composite/{composite_id}",
        "width": composited_pil.width,
        "height": composited_pil.height,
        "parameters": payload.model_dump(),
    }


@router.get(
    "/composite/{composite_id}",
    summary="Serve composited image",
)
def get_composite_image(composite_id: str):
    from fastapi.responses import FileResponse

    file_path = settings.base_storage_dir / "composites" / f"{composite_id}.png"
    if not file_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Composite image not found.")

    return FileResponse(str(file_path), media_type="image/png")
