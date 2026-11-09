import logging
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from backend.app.core.config import settings
from backend.app.db.models import Base, User

logger = logging.getLogger(__name__)

# Configure connect_args for SQLite
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

DEFAULT_SYSTEM_USER_ID = "00000000-0000-0000-0000-000000000001"
DEFAULT_SYSTEM_USER_EMAIL = "designer@packvision.ai"


def get_db() -> Generator[Session, None, None]:
    """Dependency that yields a SQLAlchemy database session and ensures cleanup."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initialize database tables and seed default user if not exists."""
    logger.info("Initializing database tables...")
    Base.metadata.create_all(bind=engine)

    with SessionLocal() as db:
        user = db.query(User).filter(User.id == DEFAULT_SYSTEM_USER_ID).first()
        if not user:
            logger.info("Seeding default system user...")
            user = User(
                id=DEFAULT_SYSTEM_USER_ID,
                email=DEFAULT_SYSTEM_USER_EMAIL,
            )
            db.add(user)
            db.commit()
    logger.info("Database initialization completed successfully.")
