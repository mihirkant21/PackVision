import logging
from contextlib import asynccontextmanager
from pathlib import Path
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

    uvicorn.run(
        "backend.app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )
