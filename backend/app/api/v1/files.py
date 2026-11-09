import logging
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.db.models import Asset
from backend.app.services.asset_service import AssetService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/files/{asset_id}",
    summary="Download or stream an image asset or its thumbnail",
    description="Serves the raw image file for the given asset ID, or its 256x256 thumbnail when `thumb=true`.",
)
def get_file(
    asset_id: str,
    thumb: bool = Query(False, description="Whether to fetch the 256x256 thumbnail"),
    db: Session = Depends(get_db),
):
    asset = (
        db.query(Asset)
        .filter(Asset.id == asset_id, Asset.deleted_at.is_(None))
        .first()
    )
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Asset '{asset_id}' not found.",
        )

    if thumb:
        thumb_path = AssetService.get_thumbnail_file_path(asset)
        if thumb_path.exists():
            return FileResponse(
                path=str(thumb_path),
                media_type="image/png",
                filename=f"{asset.id}_thumb.png",
            )
        # If thumbnail doesn't exist yet, fall back to main asset file

    file_path = AssetService.get_asset_file_path(asset)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Asset file '{asset.file_name}' does not exist on disk.",
        )

    return FileResponse(
        path=str(file_path),
        media_type=asset.mime_type,
        filename=asset.file_name,
    )
