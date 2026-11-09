from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class TaxonomyItem(BaseModel):
    id: str
    label: str
    description: Optional[str] = None
    icon: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MaterialItem(BaseModel):
    id: str
    label: str
    default_finish: str
    compatible_finishes: List[str]
    description: Optional[str] = None


class TaxonomyResponse(BaseModel):
    package_types: List[TaxonomyItem]
    materials: List[MaterialItem]
    finishes: List[TaxonomyItem]
    viewpoints: List[TaxonomyItem]
    backgrounds: List[TaxonomyItem]
    lighting_presets: List[TaxonomyItem]
