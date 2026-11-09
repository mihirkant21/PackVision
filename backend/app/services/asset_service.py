import io
import uuid
import logging
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image, ImageOps
from fastapi import UploadFile, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.db.models import Asset
from backend.app.db.session import DEFAULT_SYSTEM_USER_ID

logger = logging.getLogger(__name__)

MIME_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
}


class AssetService:
    @staticmethod
    def detect_and_validate_magic_bytes(content: bytes, declared_mime: str) -> str:
        """Validate header magic bytes and match with declared MIME type."""
        if len(content) < 12:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is too small or empty.",
            )

        detected_mime: Optional[str] = None
        if content.startswith(b"\xff\xd8\xff"):
            detected_mime = "image/jpeg"
        elif content.startswith(b"\x89PNG\r\n\x1a\n"):
            detected_mime = "image/png"
        elif content[:4] == b"RIFF" and content[8:12] == b"WEBP":
            detected_mime = "image/webp"
        elif content.startswith(b"BM"):
            detected_mime = "image/bmp"

        if not detected_mime:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported or corrupted image file format (invalid magic bytes).",
            )

        # Allow image/jpg as alias for image/jpeg
        norm_declared = "image/jpeg" if declared_mime == "image/jpg" else declared_mime
        if norm_declared not in settings.ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"MIME type '{declared_mime}' is not permitted. Allowed: {settings.ALLOWED_IMAGE_TYPES}",
            )

        if detected_mime != norm_declared and not (
            detected_mime == "image/jpeg" and norm_declared == "image/jpeg"
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Declared MIME '{declared_mime}' does not match detected format '{detected_mime}'.",
            )

        return detected_mime

    @classmethod
    def validate_and_process_image(
        cls, content: bytes, declared_mime: str
    ) -> Tuple[Image.Image, str, int, int]:
        """Validate size, magic bytes, dimensions, and apply EXIF orientation."""
        # 1. File size check
        max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds maximum size limit of {settings.MAX_UPLOAD_MB} MB.",
            )

        # 2. Magic bytes validation
        valid_mime = cls.detect_and_validate_magic_bytes(content, declared_mime)

        # 3. Open image with PIL & apply EXIF transpose
        try:
            image = Image.open(io.BytesIO(content))
            # EXIF orientation correction
            image = ImageOps.exif_transpose(image)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to decode image data: {str(e)}",
            )

        # 4. Dimension checks
        w, h = image.size
        min_dim = settings.MIN_IMAGE_DIMENSION
        max_dim = settings.MAX_IMAGE_DIMENSION
        if w < min_dim or h < min_dim:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image dimensions ({w}x{h}) are below minimum {min_dim}x{min_dim}px.",
            )
        if w > max_dim or h > max_dim:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image dimensions ({w}x{h}) exceed maximum {max_dim}x{max_dim}px.",
            )

        return image, valid_mime, w, h

    @classmethod
    def save_thumbnail(cls, image: Image.Image, asset_id: str) -> Path:
        """Create 256x256 thumbnail maintaining aspect ratio and save as PNG."""
        thumb_path = settings.thumbnails_dir / f"{asset_id}.png"
        thumb_img = image.copy()
        thumb_img.thumbnail((256, 256), Image.Resampling.LANCZOS)
        # Convert RGBA if needed or RGB
        if thumb_img.mode not in ("RGB", "RGBA"):
            thumb_img = thumb_img.convert("RGBA")
        thumb_img.save(thumb_path, format="PNG", optimize=True)
        return thumb_path

    @classmethod
    async def process_and_save_upload(
        cls,
        db: Session,
        upload_file: UploadFile,
        asset_type: str = "sketch",
        user_id: Optional[str] = None,
    ) -> Asset:
        """Process an uploaded file from FastAPI endpoint, validate, create thumbnail, and persist."""
        content = await upload_file.read()
        declared_mime = upload_file.content_type or "application/octet-stream"

        image, valid_mime, width, height = cls.validate_and_process_image(
            content, declared_mime
        )

        asset_id = str(uuid.uuid4())
        ext = MIME_EXTENSIONS.get(valid_mime, ".png")
        file_name = f"{asset_id}{ext}"

        dest_dir = (
            settings.outputs_dir if asset_type == "output" else settings.inputs_dir
        )
        dest_path = dest_dir / file_name

        # Save primary file
        with open(dest_path, "wb") as f:
            f.write(content)

        # Generate thumbnail
        cls.save_thumbnail(image, asset_id)

        # Storage key relative to storage root
        sub_folder = "outputs" if asset_type == "output" else "inputs"
        storage_key = f"{sub_folder}/{file_name}"

        effective_user_id = user_id or DEFAULT_SYSTEM_USER_ID

        asset = Asset(
            id=asset_id,
            user_id=effective_user_id,
            storage_key=storage_key,
            file_name=upload_file.filename or file_name,
            asset_type=asset_type,
            mime_type=valid_mime,
            width=width,
            height=height,
            file_size=len(content),
        )
        db.add(asset)
        db.commit()
        db.refresh(asset)
        return asset

    @classmethod
    def save_output_image(
        cls,
        db: Session,
        image: Image.Image,
        file_name_label: str = "packvision_render.png",
        user_id: Optional[str] = None,
    ) -> Asset:
        """Save a generated PIL output image to storage/outputs and register in DB."""
        asset_id = str(uuid.uuid4())
        file_name = f"{asset_id}.png"
        dest_path = settings.outputs_dir / file_name

        # Ensure RGB or RGBA mode
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")

        image.save(dest_path, format="PNG", optimize=True)
        file_size = dest_path.stat().st_size
        width, height = image.size

        # Create thumbnail
        cls.save_thumbnail(image, asset_id)

        effective_user_id = user_id or DEFAULT_SYSTEM_USER_ID

        asset = Asset(
            id=asset_id,
            user_id=effective_user_id,
            storage_key=f"outputs/{file_name}",
            file_name=file_name_label,
            asset_type="output",
            mime_type="image/png",
            width=width,
            height=height,
            file_size=file_size,
        )
        db.add(asset)
        db.commit()
        db.refresh(asset)
        return asset

    @staticmethod
    def get_asset_file_path(asset: Asset) -> Path:
        """Resolve absolute filesystem path for an asset."""
        return (settings.base_storage_dir / asset.storage_key).resolve()

    @staticmethod
    def get_thumbnail_file_path(asset: Asset) -> Path:
        """Resolve absolute filesystem path for an asset thumbnail."""
        return (settings.thumbnails_dir / f"{asset.id}.png").resolve()
