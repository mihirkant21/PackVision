from fastapi import APIRouter

from backend.app.api.v1.uploads import router as uploads_router
from backend.app.api.v1.generations import router as generations_router
from backend.app.api.v1.files import router as files_router
from backend.app.api.v1.taxonomy import router as taxonomy_router
from backend.app.api.v1.health import router as health_router
from backend.app.api.v1.samples import router as samples_router
from backend.app.api.v1.projects import router as projects_router
from backend.app.api.v1.preprocess import router as preprocess_router
from backend.app.api.v1.composite import router as composite_router
from backend.app.api.v1.evaluation import router as evaluation_router

api_router = APIRouter()

api_router.include_router(health_router, tags=["Health"])
api_router.include_router(taxonomy_router, tags=["Taxonomy"])
api_router.include_router(samples_router, tags=["Samples"])
api_router.include_router(projects_router, tags=["Projects"])
api_router.include_router(preprocess_router, tags=["Preprocess"])
api_router.include_router(uploads_router, tags=["Uploads"])
api_router.include_router(generations_router, tags=["Generations"])
api_router.include_router(composite_router, tags=["Composite"])
api_router.include_router(evaluation_router, tags=["Evaluation"])
api_router.include_router(files_router, tags=["Files"])
