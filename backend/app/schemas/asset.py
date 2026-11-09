from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class AssetBase(BaseModel):
    file_name: str
    asset_type: str  # 'sketch', 'reference', 'logo', 'output'
    mime_type: str
    width: Optional[int] = None
    height: Optional[int] = None
    file_size: int


class AssetResponse(AssetBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: Optional[str] = None
    storage_key: str
    created_at: datetime
    deleted_at: Optional[datetime] = None
    url: str = Field(description="URL to access raw file asset")
    thumbnail_url: Optional[str] = Field(default=None, description="URL to access thumbnail")

    @classmethod
    def from_orm_model(cls, asset, api_prefix: str = "/api/v1") -> "AssetResponse":
        return cls(
            id=asset.id,
            user_id=asset.user_id,
            storage_key=asset.storage_key,
            file_name=asset.file_name,
            asset_type=asset.asset_type,
            mime_type=asset.mime_type,
            width=asset.width,
            height=asset.height,
            file_size=asset.file_size,
            created_at=asset.created_at,
            deleted_at=asset.deleted_at,
            url=f"{api_prefix}/files/{asset.id}",
            thumbnail_url=f"{api_prefix}/files/{asset.id}?thumb=true",
        )


class UploadResponse(BaseModel):
    success: bool = True
    message: str = "Asset uploaded successfully"
    asset: AssetResponse
