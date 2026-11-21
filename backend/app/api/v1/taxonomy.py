from fastapi import APIRouter
from backend.app.schemas.taxonomy import (
    TaxonomyResponse,
    TaxonomyItem,
    MaterialItem,
)

from src.taxonomy import taxonomy_response as canonical_taxonomy_response

router = APIRouter()


def _build_taxonomy() -> TaxonomyResponse:
    """
    Construct the API response from the canonical taxonomy (src/taxonomy.py).

    The canonical taxonomy is the single source of truth shared with
    src/prompt_engine.py and the Gradio UI. Legacy ids consumed by earlier
    clients (e.g. "box", "glass", "front", "studio_clean") are preserved
    additively by taxonomy_response(), so the response contract only grows -
    no existing field, id, or label is removed or renamed.
    """
    data = canonical_taxonomy_response()
    return TaxonomyResponse(
        package_types=[TaxonomyItem(**item) for item in data["package_types"]],
        materials=[MaterialItem(**item) for item in data["materials"]],
        finishes=[TaxonomyItem(**item) for item in data["finishes"]],
        viewpoints=[TaxonomyItem(**item) for item in data["viewpoints"]],
        backgrounds=[TaxonomyItem(**item) for item in data["backgrounds"]],
        lighting_presets=[TaxonomyItem(**item) for item in data["lighting_presets"]],
    )


TAXONOMY_DATA = _build_taxonomy()


@router.get(
    "/taxonomy",
    response_model=TaxonomyResponse,
    summary="Get packaging design taxonomy",
    description="Returns available package types, materials, finishes, viewpoints, backgrounds, and lighting presets.",
)
def get_taxonomy() -> TaxonomyResponse:
    return TAXONOMY_DATA
