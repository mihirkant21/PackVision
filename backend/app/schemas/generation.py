from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, ConfigDict, Field
from backend.app.schemas.asset import AssetResponse


class GenerationCreate(BaseModel):
    input_asset_id: str = Field(description="UUID of sketch or base asset")
    reference_asset_id: Optional[str] = Field(
        default=None, description="UUID of reference/moodboard asset"
    )
    logo_asset_id: Optional[str] = Field(
        default=None, description="UUID of logo asset for branding overlay"
    )
    package_type: str = Field(
        default="bottle",
        description="Packaging category e.g. bottle, box, can, pouch, jar, tube",
    )
    material: Optional[str] = Field(
        default="glass",
        description="Material type e.g. glass, cardboard, aluminum, plastic, kraft_paper",
    )
    finish: Optional[str] = Field(
        default="glossy",
        description="Material finish e.g. matte, glossy, metallic, embossed, foil_stamping",
    )
    viewpoint: Optional[str] = Field(
        default="front",
        description="Camera viewpoint e.g. front, isometric, three_quarter, top_down",
    )
    background: Optional[str] = Field(
        default="studio_clean",
        description="Background scene e.g. studio_clean, retail_shelf, minimalist_gradient",
    )
    prompt: Optional[str] = Field(
        default=None, description="Additional user prompt details or custom descriptors"
    )
    negative_prompt: Optional[str] = Field(
        default=None, description="Elements to exclude from rendering"
    )
    input_mode: str = Field(
        default="auto",
        pattern="^(auto|sketch|photo)$",
        description=(
            "Input treatment: 'auto' detects photo vs sketch, 'sketch' forces "
            "line-art handling, 'photo' treats the upload as a real product photo"
        ),
    )
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    num_outputs: int = Field(
        default=1, ge=1, le=4, description="Number of candidate variations to generate"
    )
    guidance_scale: float = Field(
        default=7.5, ge=1.0, le=20.0, description="Classifier-free guidance scale"
    )
    controlnet_scale: float = Field(
        default=0.85, ge=0.0, le=1.5, description="ControlNet conditioning scale"
    )
    steps: int = Field(default=30, ge=10, le=100, description="Denoising steps")
    logo_placement: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional positioning configuration for logo composting (x, y, scale, blend_mode)",
    )


class GenerationOutputDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    generation_id: str
    asset_id: str
    is_selected: bool = False
    metrics: Optional[Dict[str, Any]] = Field(default=None)
    created_at: datetime
    asset: Optional[AssetResponse] = None


class JobStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    status: str
    progress: int
    stage_message: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class GenerationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: Optional[str] = None
    input_asset_id: str
    reference_asset_id: Optional[str] = None
    logo_asset_id: Optional[str] = None
    status: str
    package_type: str
    material: Optional[str] = None
    finish: Optional[str] = None
    viewpoint: Optional[str] = None
    background: Optional[str] = None
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    model_name: str
    model_revision: str
    parameters_json: Optional[Dict[str, Any]] = None
    progress: int
    stage_message: str
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    outputs: List[GenerationOutputDetail] = []
    input_asset: Optional[AssetResponse] = None
    reference_asset: Optional[AssetResponse] = None
    logo_asset: Optional[AssetResponse] = None


class GenerationListResponse(BaseModel):
    total: int
    items: List[GenerationResponse]
