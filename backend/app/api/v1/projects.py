import logging
import uuid
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session, joinedload

from backend.app.db.session import get_db, DEFAULT_SYSTEM_USER_ID
from backend.app.db.models import Project, Generation, GenerationOutput
from backend.app.schemas.project import (
    ProjectCreate,
    ProjectUpdate,
    ProjectResponse,
    ProjectListResponse,
    ProjectDetailResponse,
)
from backend.app.api.v1.generations import serialize_generation

logger = logging.getLogger(__name__)

router = APIRouter()


def serialize_project(proj: Project) -> ProjectResponse:
    return ProjectResponse(
        id=proj.id,
        user_id=proj.user_id,
        name=proj.name,
        description=proj.description,
        package_type=proj.package_type,
        material=proj.material,
        finish=proj.finish,
        prompt=proj.prompt,
        negative_prompt=proj.negative_prompt,
        thumbnail_url=proj.thumbnail_url,
        created_at=proj.created_at,
        updated_at=proj.updated_at,
        generations_count=len(proj.generations) if proj.generations else 0,
    )


@router.post(
    "/projects",
    response_model=ProjectResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new packaging project",
)
def create_project(
    payload: ProjectCreate,
    user_id: Optional[str] = Query(None, description="Optional requesting user ID"),
    db: Session = Depends(get_db),
) -> ProjectResponse:
    effective_user_id = user_id or DEFAULT_SYSTEM_USER_ID
    project = Project(
        id=str(uuid.uuid4()),
        user_id=effective_user_id,
        name=payload.name.strip(),
        description=payload.description,
        package_type=payload.package_type,
        material=payload.material,
        finish=payload.finish,
        prompt=payload.prompt,
        negative_prompt=payload.negative_prompt,
        thumbnail_url=payload.thumbnail_url,
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return serialize_project(project)


@router.get(
    "/projects",
    response_model=ProjectListResponse,
    summary="List all packaging projects",
)
def list_projects(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    search: Optional[str] = Query(None, description="Search by name"),
    db: Session = Depends(get_db),
) -> ProjectListResponse:
    query = db.query(Project)
    if search:
        query = query.filter(Project.name.ilike(f"%{search.strip()}%"))

    total = query.count()
    projects = (
        query.options(joinedload(Project.generations))
        .order_by(Project.updated_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return ProjectListResponse(
        total=total,
        items=[serialize_project(p) for p in projects],
    )


@router.get(
    "/projects/{project_id}",
    response_model=ProjectDetailResponse,
    summary="Get project details and its generations",
)
def get_project(
    project_id: str,
    db: Session = Depends(get_db),
) -> ProjectDetailResponse:
    project = (
        db.query(Project)
        .options(
            joinedload(Project.generations)
            .joinedload(Generation.outputs)
            .joinedload(GenerationOutput.asset),
            joinedload(Project.generations).joinedload(Generation.input_asset),
            joinedload(Project.generations).joinedload(Generation.logo_asset),
        )
        .filter(Project.id == project_id)
        .first()
    )
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project '{project_id}' not found.",
        )

    generations_serialized = [
        serialize_generation(g)
        for g in sorted(project.generations or [], key=lambda x: x.created_at, reverse=True)
    ]

    return ProjectDetailResponse(
        id=project.id,
        user_id=project.user_id,
        name=project.name,
        description=project.description,
        package_type=project.package_type,
        material=project.material,
        finish=project.finish,
        prompt=project.prompt,
        negative_prompt=project.negative_prompt,
        thumbnail_url=project.thumbnail_url,
        created_at=project.created_at,
        updated_at=project.updated_at,
        generations_count=len(project.generations or []),
        generations=generations_serialized,
    )


@router.put(
    "/projects/{project_id}",
    response_model=ProjectResponse,
    summary="Update project details",
)
def update_project(
    project_id: str,
    payload: ProjectUpdate,
    db: Session = Depends(get_db),
) -> ProjectResponse:
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project '{project_id}' not found.",
        )

    if payload.name is not None:
        project.name = payload.name.strip()
    if payload.description is not None:
        project.description = payload.description
    if payload.package_type is not None:
        project.package_type = payload.package_type
    if payload.material is not None:
        project.material = payload.material
    if payload.finish is not None:
        project.finish = payload.finish
    if payload.prompt is not None:
        project.prompt = payload.prompt
    if payload.negative_prompt is not None:
        project.negative_prompt = payload.negative_prompt
    if payload.thumbnail_url is not None:
        project.thumbnail_url = payload.thumbnail_url

    db.commit()
    db.refresh(project)
    return serialize_project(project)


@router.delete(
    "/projects/{project_id}",
    summary="Delete a project",
)
def delete_project(
    project_id: str,
    db: Session = Depends(get_db),
):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project '{project_id}' not found.",
        )

    # Disassociate generations rather than deleting renders
    for gen in project.generations:
        gen.project_id = None

    db.delete(project)
    db.commit()
    return {"success": True, "message": f"Project '{project_id}' deleted."}
