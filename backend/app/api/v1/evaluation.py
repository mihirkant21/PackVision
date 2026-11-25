import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from PIL import Image

from backend.app.db.session import get_db
from backend.app.db.models import Asset
from backend.app.services.asset_service import AssetService
from src.evaluation import PackagingEvaluator

logger = logging.getLogger(__name__)

router = APIRouter()


class EvaluationRequest(BaseModel):
    sketch_asset_id: str = Field(..., description="UUID of original sketch asset")
    render_asset_id: str = Field(..., description="UUID of generated render asset")
    prompt: Optional[str] = Field(default=None, description="Prompt text for semantic alignment check")


@router.post(
    "/evaluate",
    summary="Evaluate structural preservation and quality metrics",
    description="Computes Edge SSIM, Edge IoU, structural preservation, and prompt fidelity comparing sketch with generated render.",
)
def evaluate_render(
    payload: EvaluationRequest,
    db: Session = Depends(get_db),
):
    # 1. Fetch sketch asset
    sketch_asset = db.query(Asset).filter(Asset.id == payload.sketch_asset_id, Asset.deleted_at.is_(None)).first()
    if not sketch_asset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Sketch asset '{payload.sketch_asset_id}' not found.")
    sketch_path = AssetService.get_asset_file_path(sketch_asset)
    if not sketch_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sketch file missing from storage.")

    # 2. Fetch render asset
    render_asset = db.query(Asset).filter(Asset.id == payload.render_asset_id, Asset.deleted_at.is_(None)).first()
    if not render_asset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Render asset '{payload.render_asset_id}' not found.")
    render_path = AssetService.get_asset_file_path(render_asset)
    if not render_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Render file missing from storage.")

    try:
        sketch_pil = Image.open(sketch_path).convert("RGB")
        render_pil = Image.open(render_path).convert("RGB")

        evaluator = PackagingEvaluator()
        report = evaluator.evaluate(
            sketch_image=sketch_pil,
            rendered_image=render_pil,
            prompt_text=payload.prompt or "",
        )

        edge_ssim = report.edge_ssim if report.edge_ssim is not None else 0.0
        edge_iou = report.edge_iou if report.edge_iou is not None else 0.0
        clip_score = report.clip_score

        # Composite structural preservation score (0 - 100)
        preservation_score = round(((float(edge_ssim) * 0.6) + (float(edge_iou) * 0.4)) * 100, 1)
        quality_rating = report.fidelity_rating or "Good"

        return {
            "success": True,
            "sketch_asset_id": payload.sketch_asset_id,
            "render_asset_id": payload.render_asset_id,
            "metrics": {
                "edge_ssim": round(float(edge_ssim), 4),
                "edge_iou": round(float(edge_iou), 4),
                "clip_score": round(float(clip_score), 4) if clip_score is not None else None,
                "structural_preservation_score": preservation_score,
                "structural_rating": quality_rating,
            },
            "provenance": {
                "edge_ssim": report.metric_sources.get("edge_ssim", "library"),
                "edge_iou": report.metric_sources.get("edge_iou", "library"),
                "clip_score": report.clip_source or "heuristic",
            },
            "summary": (
                f"Structural consistency score: {preservation_score}%. "
                f"Edge SSIM: {round(float(edge_ssim), 3)}, Edge IoU: {round(float(edge_iou), 3)}. "
                f"Packaging geometry preservation is classified as {quality_rating}."
            ),
        }
    except Exception as e:
        logger.error(f"Error computing evaluation metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}",
        )
