import os
import sys
import subprocess
from pathlib import Path

# Ensure project root and backend dir are at the top of sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_BACKEND_DIR = _PROJECT_ROOT / "backend"
for _p in [str(_BACKEND_DIR), str(_PROJECT_ROOT)]:
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

# Self-relauncher: if executed with global python lacking dependencies,
# automatically delegate to the local project virtual environment (.venv).
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
from backend.app.core.rate_limit import (
    upload_limiter,
    rate_limit_response_body,
)
from backend.app.db.session import init_db
from backend.app.api.v1.router import api_router
from backend.app.services.job_service import job_service

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
    # Recover jobs orphaned by a previous process (crash / restart): any job
    # still in an active state that predates this process start is marked
    # failed with error_code "interrupted" so clients can retry.
    recovered = job_service.recover_stuck_jobs()
    if recovered:
        logger.warning(
            f"Recovered {recovered} orphaned job(s) left behind by a previous run."
        )
    logger.info("PackVision AI backend startup complete.")
    yield
    logger.info("Shutting down PackVision AI backend server...")
    # Signal in-flight workers to stop at their next stage boundary (including
    # the checkpoint right after the generation stage, so results rendered
    # during teardown are discarded and those jobs end up 'cancelled').
    job_service.cancel_active_jobs()
    # NOTE: job_service is a MODULE-LEVEL SINGLETON whose ThreadPoolExecutor
    # is shared for the whole process. FastAPI TestClient runs this lifespan
    # teardown after every test, and executor.shutdown() permanently poisons
    # the singleton (later submit() raises "cannot schedule new futures after
    # shutdown"), breaking every subsequent test that creates a job. So the
    # executor is only shut down at real process exit — tests set
    # PACKVISION_TESTING=1 in conftest BEFORE importing this module, and here
    # we skip the shutdown for them. Process exit reaps the worker threads.
    if os.getenv("PACKVISION_TESTING", "").strip().lower() not in ("1", "true", "yes"):
        job_service.shutdown(wait=False, cancel_running=False)


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Photorealistic Packaging Renderer & Neural Design Synthesis Backend API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware.
# CORS spec constraint: a wildcard origin ("*") is INVALID when combined with
# allow_credentials=True — browsers reject credentialed responses from a
# wildcard origin, and Starlette's CORSMiddleware will not echo "*" for
# credentialed requests. So the two modes below are mutually exclusive:
#   - CORS_ORIGINS unset/"*" -> allow_origins=["*"] with credentials disabled.
#   - CORS_ORIGINS="https://a.com,https://b.com" -> explicit allow-list with
#     credentials enabled.
# Read directly from the environment (not config.py) to keep this file
# self-contained; comma-separated list, defaults to wildcard.
_cors_origins_raw = os.getenv("CORS_ORIGINS", "*")
_cors_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()] or ["*"]
if _cors_origins == ["*"]:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Rate limiting for the upload endpoint.
# backend/app/api/v1/uploads.py applies per-endpoint rate limiting via FastAPI
# dependencies for the endpoints it owns (generation create/read limits live in
# backend/app/api/v1/generations.py). The uploads router is not owned here, so
# POST /api/v1/uploads is limited with this lightweight HTTP middleware using
# the shared upload_limiter (10 req/min/IP by default, env-overridable via
# RATE_LIMIT_UPLOADS_PER_MIN). Only the remote IP is consulted, in-memory only.
@app.middleware("http")
async def uploads_rate_limit_middleware(request: Request, call_next):
    path = request.url.path.rstrip("/")
    if (
        request.method == "POST"
        and path == f"{settings.API_V1_PREFIX}/uploads"
        and upload_limiter.enabled
    ):
        host = request.client.host if request.client else "unknown"
        allowed, retry_after = upload_limiter.check(host)
        if not allowed:
            message, seconds = rate_limit_response_body(retry_after)
            return JSONResponse(
                status_code=429,
                content={"detail": message},
                headers={"Retry-After": str(seconds)},
            )
    return await call_next(request)

# Mount API v1 router
app.include_router(api_router, prefix=settings.API_V1_PREFIX)

# Static files for modern packaging studio workbench
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
FRONTEND_DIST = FRONTEND_DIR / "dist"
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", summary="Web Workbench UI or API metadata")
def root(request: Request):
    accept_header = request.headers.get("accept", "")
    dist_index = FRONTEND_DIST / "index.html"
    index_file = dist_index if dist_index.exists() else (FRONTEND_DIR / "index.html")
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
