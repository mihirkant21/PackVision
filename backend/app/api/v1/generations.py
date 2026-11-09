import logging
import uuid
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session, joinedload

from backend.app.db.session import get_db, DEFAULT_SYSTEM_USER_ID
from backend.app.db.models import Generation, GenerationOutput, Asset, Feedback
from backend.app.schemas.generation import (
    GenerationCreate,
    GenerationResponse,
    JobStatusResponse,
    GenerationOutputDetail,
    GenerationListResponse,
)
from backend.app.schemas.asset import AssetResponse
from backend.app.schemas.feedback import FeedbackCreate, FeedbackResponse
from backend.app.services.job_service import job_service

logger = logging.getLogger(__name__)

router = APIRouter()


def serialize_generation(gen: Generation) -> GenerationResponse:
    """Helper to transform SQLAlchemy Generation model to GenerationResponse schema."""
    outputs_detail: List[GenerationOutputDetail] = []
    if gen.outputs:
        for out in gen.outputs:
            asset_resp = (
                AssetResponse.from_orm_model(out.asset) if out.asset else None
            )
            outputs_detail.append(
                GenerationOutputDetail(
                    id=out.id,
                    generation_id=out.generation_id,
                    asset_id=out.asset_id,
                    is_selected=out.is_selected,
                    metrics=out.metrics_json,
                    created_at=out.created_at,
                    asset=asset_resp,
                )
            )

    return GenerationResponse(
        id=gen.id,
        user_id=gen.user_id,
        input_asset_id=gen.input_asset_id,
        reference_asset_id=gen.reference_asset_id,
        logo_asset_id=gen.logo_asset_id,
        status=gen.status,
        package_type=gen.package_type,
        material=gen.material,
        finish=gen.finish,
        viewpoint=gen.viewpoint,
        background=gen.background,
        prompt=gen.prompt,
        negative_prompt=gen.negative_prompt,
        seed=gen.seed,
        model_name=gen.model_name,
        model_revision=gen.model_revision,
        parameters_json=gen.parameters_json,
        progress=gen.progress,
        stage_message=gen.stage_message,
        error_code=gen.error_code,
        error_message=gen.error_message,
        created_at=gen.created_at,
        started_at=gen.started_at,
        completed_at=gen.completed_at,
        outputs=outputs_detail,
        input_asset=(
            AssetResponse.from_orm_model(gen.input_asset)
            if gen.input_asset
            else None
        ),
        reference_asset=(
            AssetResponse.from_orm_model(gen.reference_asset)
            if gen.reference_asset
            else None
        ),
        logo_asset=(
            AssetResponse.from_orm_model(gen.logo_asset)
            if gen.logo_asset
            else None
        ),
    )


@router.post(
    "/generations",
    response_model=GenerationResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create packaging generation job",
    description="Validates input assets, enqueues rendering job to background worker queue, and immediately returns job status.",
)
def create_generation(
    payload: GenerationCreate,
    user_id: Optional[str] = Query(None, description="Optional requesting user ID"),
    db: Session = Depends(get_db),
) -> GenerationResponse:
    # 1. Validate input asset
    input_asset = (
        db.query(Asset)
        .filter(Asset.id == payload.input_asset_id, Asset.deleted_at.is_(None))
        .first()
    )
    if not input_asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Input asset '{payload.input_asset_id}' not found.",
        )

    # 2. Validate optional reference asset
    if payload.reference_asset_id:
        ref_asset = (
            db.query(Asset)
            .filter(
                Asset.id == payload.reference_asset_id, Asset.deleted_at.is_(None)
            )
            .first()
        )
        if not ref_asset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Reference asset '{payload.reference_asset_id}' not found.",
            )

    # 3. Validate optional logo asset
    if payload.logo_asset_id:
        logo_asset = (
            db.query(Asset)
            .filter(Asset.id == payload.logo_asset_id, Asset.deleted_at.is_(None))
            .first()
        )
        if not logo_asset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Logo asset '{payload.logo_asset_id}' not found.",
            )

    # 4. Construct parameters JSON
    parameters = {
        "num_outputs": payload.num_outputs,
        "guidance_scale": payload.guidance_scale,
        "controlnet_scale": payload.controlnet_scale,
        "steps": payload.steps,
        "logo_placement": payload.logo_placement,
    }

    effective_user_id = user_id or DEFAULT_SYSTEM_USER_ID

    generation = Generation(
        id=str(uuid.uuid4()),
        user_id=effective_user_id,
        input_asset_id=payload.input_asset_id,
        reference_asset_id=payload.reference_asset_id,
        logo_asset_id=payload.logo_asset_id,
        status="queued",
        package_type=payload.package_type,
        material=payload.material,
        finish=payload.finish,
        viewpoint=payload.viewpoint,
        background=payload.background,
        prompt=payload.prompt,
        negative_prompt=payload.negative_prompt,
        seed=payload.seed,
        parameters_json=parameters,
        progress=0,
        stage_message="Job queued for processing",
    )
    db.add(generation)
    db.commit()
    db.refresh(generation)

    # 5. Enqueue background rendering task
    job_service.submit_generation_job(generation.id)

    return serialize_generation(generation)


@router.get(
    "/generations",
    response_model=GenerationListResponse,
    summary="List recent packaging generation jobs",
)
def list_generations(
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status"),
    db: Session = Depends(get_db),
) -> GenerationListResponse:
    query = db.query(Generation)
    if status_filter:
        query = query.filter(Generation.status == status_filter)

    total = query.count()
    generations = (
        query.options(
            joinedload(Generation.outputs).joinedload(GenerationOutput.asset),
            joinedload(Generation.input_asset),
            joinedload(Generation.reference_asset),
            joinedload(Generation.logo_asset),
        )
        .order_by(Generation.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return GenerationListResponse(
        total=total,
        items=[serialize_generation(g) for g in generations],
    )


@router.get(
    "/generations/{generation_id}",
    response_model=GenerationResponse,
    summary="Get generation job status, progress, and results",
)
def get_generation(
    generation_id: str,
    db: Session = Depends(get_db),
) -> GenerationResponse:
    gen = (
        db.query(Generation)
        .options(
            joinedload(Generation.outputs).joinedload(GenerationOutput.asset),
            joinedload(Generation.input_asset),
            joinedload(Generation.reference_asset),
            joinedload(Generation.logo_asset),
        )
        .filter(Generation.id == generation_id)
        .first()
    )
    if not gen:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Generation job '{generation_id}' not found.",
        )
    return serialize_generation(gen)


@router.post(
    "/generations/{generation_id}/retry",
    response_model=JobStatusResponse,
    summary="Retry a failed generation job",
)
def retry_generation(
    generation_id: str,
    db: Session = Depends(get_db),
) -> JobStatusResponse:
    gen = db.query(Generation).filter(Generation.id == generation_id).first()
    if not gen:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Generation job '{generation_id}' not found.",
        )

    retried_gen = job_service.retry_generation(db, generation_id)
    if not retried_gen:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retry generation job.",
        )

    return JobStatusResponse(
        id=retried_gen.id,
        status=retried_gen.status,
        progress=retried_gen.progress,
        stage_message=retried_gen.stage_message,
        error_code=retried_gen.error_code,
        error_message=retried_gen.error_message,
        created_at=retried_gen.created_at,
        started_at=retried_gen.started_at,
        completed_at=retried_gen.completed_at,
    )


@router.post(
    "/generations/{generation_id}/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit user quality rating and feedback for a generation",
)
def submit_feedback(
    generation_id: str,
    payload: FeedbackCreate,
    user_id: Optional[str] = Query(None, description="Optional user ID submitting feedback"),
    db: Session = Depends(get_db),
) -> FeedbackResponse:
    gen = db.query(Generation).filter(Generation.id == generation_id).first()
    if not gen:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Generation job '{generation_id}' not found.",
        )

    effective_user_id = user_id or gen.user_id or DEFAULT_SYSTEM_USER_ID

    feedback = Feedback(
        id=str(uuid.uuid4()),
        generation_id=generation_id,
        user_id=effective_user_id,
        rating=payload.rating,
        issue_categories=payload.issue_categories,
        comment=payload.comment,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)

    return FeedbackResponse(
        id=feedback.id,
        generation_id=feedback.generation_id,
        user_id=feedback.user_id,
        rating=feedback.rating,
        issue_categories=feedback.issue_categories,
        comment=feedback.comment,
        created_at=feedback.created_at,
    )


@router.delete(
    "/generations/{generation_id}",
    summary="Delete a generation job and associated outputs",
)
def delete_generation(
    generation_id: str,
    db: Session = Depends(get_db),
):
    gen = db.query(Generation).filter(Generation.id == generation_id).first()
    if not gen:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Generation job '{generation_id}' not found.",
        )

    # Set status to cancelled to prevent background task continuation if running
    gen.status = "cancelled"
    db.commit()

    db.delete(gen)
    db.commit()
    return {
        "success": True,
        "message": f"Generation '{generation_id}' deleted successfully.",
    }
