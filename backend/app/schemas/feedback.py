from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, ConfigDict, Field


class FeedbackCreate(BaseModel):
    rating: int = Field(ge=1, le=5, description="User rating from 1 (poor) to 5 (excellent)")
    issue_categories: Optional[List[str]] = Field(
        default=None,
        description="List of detected issues e.g. ['geometry_distortion', 'logo_misalignment', 'material_inaccuracy']",
    )
    comment: Optional[str] = Field(
        default=None, description="Optional textual user feedback or critique"
    )


class FeedbackResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    generation_id: str
    user_id: Optional[str] = None
    rating: int
    issue_categories: Optional[List[str]] = None
    comment: Optional[str] = None
    created_at: datetime
