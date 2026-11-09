import logging
from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.schemas.asset import UploadResponse, AssetResponse
from backend.app.services.asset_service import AssetService

logger = logging.getLogger(__name__)

router = APIRouter()

VALID_UPLOAD_ASSET_TYPES = {"sketch", "reference", "logo"}


@router.post(
    "/uploads",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload image asset (sketch, reference moodboard, or vector logo)",
    description="Validates image magic bytes, MIME type, dimensions (128-4096px), auto-corrects EXIF orientation, creates 256x256 thumbnail, and persists to storage.",
)
async def upload_asset(
    file: UploadFile = File(..., description="Image binary file to upload"),
    asset_type: str = Form(
        "sketch",
        description="Type of asset: 'sketch', 'reference', or 'logo'",
    ),
    user_id: Optional[str] = Form(
        None, description="Optional user identifier associated with asset"
    ),
    db: Session = Depends(get_db),
) -> UploadResponse:
    norm_type = asset_type.strip().lower()
    if norm_type not in VALID_UPLOAD_ASSET_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid asset_type '{asset_type}'. Must be one of: {sorted(list(VALID_UPLOAD_ASSET_TYPES))}",
        )

    asset = await AssetService.process_and_save_upload(
        db=db,
        upload_file=file,
        asset_type=norm_type,
        user_id=user_id,
    )

    return UploadResponse(
        success=True,
        message=f"{norm_type.capitalize()} uploaded and processed successfully.",
        asset=AssetResponse.from_orm_model(asset),
    )
