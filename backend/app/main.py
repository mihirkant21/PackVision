import os
import sys
import subprocess
from pathlib import Path

# Self-relauncher: if executed with global python lacking dependencies,
# automatically delegate to the local project virtual environment (.venv).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_VENV_PYTHON = _PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
if _VENV_PYTHON.exists() and Path(sys.executable).resolve() != _VENV_PYTHON.resolve():
    try:
        import fastapi
        import uvicorn
    except ImportError:
        print(f"[PackVision AI] Delegating execution to virtual environment: {_VENV_PYTHON}")
        sys.exit(subprocess.run([str(_VENV_PYTHON)] + sys.argv).returncode)

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from backend.app.core.config import settings
from backend.app.db.session import init_db
from backend.app.api.v1.router import api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("packvision")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager handling application startup and shutdown."""
    logger.info("Starting up PackVision AI backend server...")
    # Ensure storage paths exist
    settings.ensure_storage_dirs()
    # Initialize DB schemas and default seeds
    init_db()
    logger.info("PackVision AI backend startup complete.")
    yield
    logger.info("Shutting down PackVision AI backend server...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Photorealistic Packaging Renderer & Neural Design Synthesis Backend API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API v1 router
app.include_router(api_router, prefix=settings.API_V1_PREFIX)

# Static files for modern packaging studio workbench
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", summary="Web Workbench UI or API metadata")
def root(request: Request):
    accept_header = request.headers.get("accept", "")
    index_file = FRONTEND_DIR / "index.html"
    if "text/html" in accept_header and index_file.exists():
        return FileResponse(str(index_file))
    
    return {
        "service": settings.PROJECT_NAME,
        "version": "1.0.0",
        "status": "operational",
        "workbench_url": "/",
        "docs_url": "/docs",
        "api_v1_prefix": settings.API_V1_PREFIX,
        "mock_inference": settings.USE_MOCK_INFERENCE,
    }


@app.get("/workbench", include_in_schema=False)
def serve_workbench():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"error": "Frontend workbench not found."}


if __name__ == "__main__":
    import uvicorn
    import socket

    target_port = settings.API_PORT
    # Check if target port is occupied; if so, allocate next available port
    for p in range(target_port, target_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", p)) != 0:
                target_port = p
                break
        logger.warning(f"Port {p} is currently busy, testing next port...")

    logger.info(f"Starting PackVision Server on http://{settings.API_HOST}:{target_port}")
    uvicorn.run(
        "backend.app.main:app",
        host=settings.API_HOST,
        port=target_port,
        reload=settings.DEBUG,
    )
