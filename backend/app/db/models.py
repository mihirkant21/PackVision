import uuid
from datetime import datetime, timezone
from typing import Optional, List, Any
from sqlalchemy import (
    String,
    Integer,
    Boolean,
    Text,
    DateTime,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    email: Mapped[str] = mapped_column(
        String(255), unique=True, index=True, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    # Relationships
    assets: Mapped[List["Asset"]] = relationship("Asset", back_populates="user")
    generations: Mapped[List["Generation"]] = relationship(
        "Generation", back_populates="user"
    )
    feedbacks: Mapped[List["Feedback"]] = relationship("Feedback", back_populates="user")
    projects: Mapped[List["Project"]] = relationship("Project", back_populates="user")

    def __repr__(self) -> str:
        return f"<User {self.email}>"


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=True, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    package_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    material: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    finish: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    negative_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    thumbnail_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    # Relationships
    user: Mapped[Optional["User"]] = relationship("User", back_populates="projects")
    generations: Mapped[List["Generation"]] = relationship(
        "Generation", back_populates="project"
    )

    def __repr__(self) -> str:
        return f"<Project {self.name} ({self.id})>"


class Asset(Base):
    __tablename__ = "assets"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=True, index=True
    )
    project_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True, index=True
    )
    storage_key: Mapped[str] = mapped_column(String(500), nullable=False)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    # asset_type: 'sketch', 'reference', 'logo', 'output'
    asset_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    user: Mapped[Optional["User"]] = relationship("User", back_populates="assets")
    outputs: Mapped[List["GenerationOutput"]] = relationship(
        "GenerationOutput", back_populates="asset", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Asset {self.id} ({self.asset_type})>"


class Generation(Base):
    __tablename__ = "generations"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=True, index=True
    )
    project_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True, index=True
    )
    input_asset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("assets.id"), nullable=False, index=True
    )
    reference_asset_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("assets.id"), nullable=True
    )
    logo_asset_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("assets.id"), nullable=True
    )
    # status: 'queued', 'preprocessing', 'generating', 'postprocessing', 'completed', 'failed', 'cancelled'
    status: Mapped[str] = mapped_column(
        String(50), default="queued", index=True, nullable=False
    )
    package_type: Mapped[str] = mapped_column(String(100), nullable=False)
    material: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    finish: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    viewpoint: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    background: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    negative_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    seed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    model_name: Mapped[str] = mapped_column(
        # Neutral pending-label; overwritten with the engine that actually
        # rendered (e.g. "packvisionnet-v2", "cloud-gemini") on completion.
        String(100), default="pending-engine-selection", nullable=False
    )
    model_revision: Mapped[str] = mapped_column(
        String(50), default="unspecified", nullable=False
    )
    parameters_json: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    stage_message: Mapped[str] = mapped_column(
        String(255), default="Job queued", nullable=False
    )
    error_code: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    user: Mapped[Optional["User"]] = relationship("User", back_populates="generations")
    project: Mapped[Optional["Project"]] = relationship("Project", back_populates="generations")
    input_asset: Mapped["Asset"] = relationship(
        "Asset", foreign_keys=[input_asset_id], backref="input_for_generations"
    )
    reference_asset: Mapped[Optional["Asset"]] = relationship(
        "Asset", foreign_keys=[reference_asset_id], backref="reference_for_generations"
    )
    logo_asset: Mapped[Optional["Asset"]] = relationship(
        "Asset", foreign_keys=[logo_asset_id], backref="logo_for_generations"
    )
    outputs: Mapped[List["GenerationOutput"]] = relationship(
        "GenerationOutput", back_populates="generation", cascade="all, delete-orphan"
    )
    feedbacks: Mapped[List["Feedback"]] = relationship(
        "Feedback", back_populates="generation", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Generation {self.id} status={self.status}>"


class GenerationOutput(Base):
    __tablename__ = "generation_outputs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    generation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("generations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    asset_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("assets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    is_selected: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    metrics_json: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )

    # Relationships
    generation: Mapped["Generation"] = relationship("Generation", back_populates="outputs")
    asset: Mapped["Asset"] = relationship("Asset", back_populates="outputs")

    def __repr__(self) -> str:
        return f"<GenerationOutput {self.id} gen={self.generation_id} asset={self.asset_id}>"


class Feedback(Base):
    __tablename__ = "feedbacks"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    generation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("generations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=True
    )
    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    issue_categories: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )

    # Relationships
    generation: Mapped["Generation"] = relationship("Generation", back_populates="feedbacks")
    user: Mapped[Optional["User"]] = relationship("User", back_populates="feedbacks")

    def __repr__(self) -> str:
        return f"<Feedback {self.id} rating={self.rating}>"
