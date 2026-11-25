import io
import uuid
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form, status
from sqlalchemy.orm import Session
from PIL import Image

from backend.app.db.session import get_db
from backend.app.db.models import Asset
from backend.app.core.config import settings
from backend.app.services.asset_service import AssetService
from src.preprocessor import SketchPreprocessor, PreprocessResult

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/preprocess",
    summary="Preprocess sketch and generate Canny, Line-art, and Pseudo-Depth maps",
    description="Processes an existing asset or uploaded sketch, returning separate image URLs for the original, Canny edge map, line-art map, and pseudo-depth map.",
)
async def preprocess_sketch(
    asset_id: Optional[str] = Form(None, description="Existing asset ID"),
    file: Optional[UploadFile] = File(None, description="Optional image file if not using existing asset"),
    low_threshold: Optional[int] = Form(None, ge=1, le=255, description="Canny lower threshold"),
    high_threshold: Optional[int] = Form(None, ge=1, le=255, description="Canny upper threshold"),
    dilation: int = Form(1, ge=0, le=5, description="Stroke dilation kernel size"),
    target_size: int = Form(1024, ge=256, le=1024, description="Target square resolution"),
    db: Session = Depends(get_db),
):
    # 1. Resolve source image
    sketch_pil: Optional[Image.Image] = None
    source_asset_id: Optional[str] = asset_id

    if asset_id:
        asset = db.query(Asset).filter(Asset.id == asset_id, Asset.deleted_at.is_(None)).first()
        if not asset:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Asset '{asset_id}' not found.")
        file_path = AssetService.get_asset_file_path(asset)
        if not file_path.exists():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset file missing from storage.")
        sketch_pil = Image.open(file_path).convert("RGB")
    elif file:
        content = await file.read()
        sketch_pil = Image.open(io.BytesIO(content)).convert("RGB")
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide either 'asset_id' or a 'file' upload.",
        )

    # 2. Run SketchPreprocessor with parameters
    preprocessor = SketchPreprocessor(
        target_size=(target_size, target_size),
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        dilation_kernel_size=dilation,
    )
    result: PreprocessResult = preprocessor.preprocess(sketch_pil)

    # 3. Save preprocessed outputs in storage/preprocessed/{uuid}/
    prep_id = str(uuid.uuid4())
    prep_dir = settings.base_storage_dir / "preprocessed" / prep_id
    prep_dir.mkdir(parents=True, exist_ok=True)

    normalized_path = prep_dir / "normalized_sketch.png"
    canny_path = prep_dir / "canny_edge_map.png"
    depth_path = prep_dir / "pseudo_depth_map.png"
    mask_path = prep_dir / "silhouette_mask.png"

    result.image.save(normalized_path, format="PNG")
    result.canny.save(canny_path, format="PNG")
    result.depth.save(depth_path, format="PNG")
    result.mask.save(mask_path, format="PNG")

    # Generate Line-art inverted map (clean dark lines on white for lineart viewing)
    lineart_img = Image.eval(result.canny, lambda v: 255 - v)
    lineart_path = prep_dir / "lineart_map.png"
    lineart_img.save(lineart_path, format="PNG")

    base_url = f"/api/v1/preprocess/{prep_id}"

    # Calculate edge density metrics
    canny_arr = result.to_dict().get("metadata", {})
    
    return {
        "success": True,
        "preprocess_id": prep_id,
        "source_asset_id": source_asset_id,
        "target_size": [target_size, target_size],
        "original_size": list(result.original_size),
        "canny_low_threshold": result.metadata.get("canny_low_threshold"),
        "canny_high_threshold": result.metadata.get("canny_high_threshold"),
        "is_dark_background": result.metadata.get("is_dark_background", False),
        "edge_pixel_ratio": result.metadata.get("edge_pixel_ratio", 0.0),
        "maps": {
            "original_url": f"{base_url}/normalized",
            "canny_url": f"{base_url}/canny",
            "lineart_url": f"{base_url}/lineart",
            "depth_url": f"{base_url}/depth",
            "mask_url": f"{base_url}/mask",
        },
    }


@router.get(
    "/preprocess/{prep_id}/{map_type}",
    summary="Serve preprocessed map images",
)
def get_preprocessed_map(prep_id: str, map_type: str):
    from fastapi.responses import FileResponse

    filename_map = {
        "normalized": "normalized_sketch.png",
        "canny": "canny_edge_map.png",
        "lineart": "lineart_map.png",
        "depth": "pseudo_depth_map.png",
        "mask": "silhouette_mask.png",
    }
    filename = filename_map.get(map_type.lower())
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid map_type '{map_type}'. Valid types: {list(filename_map.keys())}",
        )

    file_path = settings.base_storage_dir / "preprocessed" / prep_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Map file not found.")

    return FileResponse(str(file_path), media_type="image/png")
