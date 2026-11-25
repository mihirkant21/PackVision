from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, ConfigDict, Field
from backend.app.schemas.generation import GenerationResponse


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Project name")
    description: Optional[str] = Field(default=None, description="Optional project description or brief")
    package_type: Optional[str] = Field(default="carton", description="Packaging category")
    material: Optional[str] = Field(default="cardboard", description="Material substrate")
    finish: Optional[str] = Field(default="matte", description="Surface finish")
    prompt: Optional[str] = Field(default=None, description="Creative prompt")
    negative_prompt: Optional[str] = Field(default=None, description="Negative prompt directives")
    thumbnail_url: Optional[str] = Field(default=None, description="Thumbnail URL")


class ProjectUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    description: Optional[str] = None
    package_type: Optional[str] = None
    material: Optional[str] = None
    finish: Optional[str] = None
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    thumbnail_url: Optional[str] = None


class ProjectResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: Optional[str] = None
    name: str
    description: Optional[str] = None
    package_type: Optional[str] = None
    material: Optional[str] = None
    finish: Optional[str] = None
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    thumbnail_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    generations_count: int = 0


class ProjectDetailResponse(ProjectResponse):
    generations: List[GenerationResponse] = []


class ProjectListResponse(BaseModel):
    total: int
    items: List[ProjectResponse]
