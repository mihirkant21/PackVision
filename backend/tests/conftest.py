import io
import tempfile
import pytest
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.core.config import settings
from backend.app.db.models import Base, User
from backend.app.db.session import (
    SessionLocal,
    get_db,
    DEFAULT_SYSTEM_USER_ID,
    DEFAULT_SYSTEM_USER_EMAIL,
)
from backend.app.main import app


@pytest.fixture(scope="session")
def test_storage():
    """Create a temporary storage directory for tests and configure settings."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        tmp_path = Path(tmp_dir)
        orig_storage = settings.STORAGE_PATH
        settings.STORAGE_PATH = str(tmp_path)
        settings.ensure_storage_dirs()
        yield tmp_path
        settings.STORAGE_PATH = orig_storage


@pytest.fixture(scope="session")
def test_engine(test_storage):
    """Create test SQLite database engine and bind to global SessionLocal."""
    db_path = test_storage / "test_packvision.db"
    settings.DATABASE_URL = f"sqlite:///{db_path}"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)

    # Rebind SessionLocal to test_engine so background tasks use test DB!
    orig_bind = SessionLocal.kw.get("bind")
    SessionLocal.configure(bind=engine)

    # Seed default user
    with SessionLocal() as session:
        user = session.query(User).filter(User.id == DEFAULT_SYSTEM_USER_ID).first()
        if not user:
            user = User(
                id=DEFAULT_SYSTEM_USER_ID,
                email=DEFAULT_SYSTEM_USER_EMAIL,
            )
            session.add(user)
            session.commit()

    yield engine

    from backend.app.services.job_service import job_service
    job_service.executor.shutdown(wait=True, cancel_futures=True)

    engine.dispose()
    if orig_bind:
        SessionLocal.configure(bind=orig_bind)


@pytest.fixture
def db_session(test_engine):
    """Provide a database session for a test."""
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(test_engine, db_session):
    """FastAPI TestClient with overridden get_db dependency."""
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    def override_get_db():
        session = TestingSession()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def valid_png_bytes() -> bytes:
    """Generate a valid 256x256 test PNG image in-memory."""
    buf = io.BytesIO()
    img = Image.new("RGBA", (256, 256), color=(60, 120, 240, 255))
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def valid_jpeg_bytes() -> bytes:
    """Generate a valid 256x256 test JPEG image in-memory."""
    buf = io.BytesIO()
    img = Image.new("RGB", (256, 256), color=(240, 100, 50))
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def corrupted_image_bytes() -> bytes:
    """Return invalid binary data masquerading as an image."""
    return b"NotARealImageFileCorruptedHeaderData123456789"


@pytest.fixture
def small_image_bytes() -> bytes:
    """Generate an image below the 128x128px minimum dimension bound."""
    buf = io.BytesIO()
    img = Image.new("RGB", (64, 64), color=(100, 100, 100))
    img.save(buf, format="PNG")
    return buf.getvalue()
