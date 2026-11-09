import os
import shutil
from typing import Dict, Any
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from sqlalchemy import text

from backend.app.core.config import settings
from backend.app.db.session import get_db

router = APIRouter()


@router.get(
    "/health",
    summary="Service health and hardware diagnostic check",
    description="Returns API service health, database connectivity, storage accessibility, and hardware accelerator status.",
)
def check_health(db: Session = Depends(get_db)) -> Dict[str, Any]:
    # 1. Database check
    db_ok = False
    db_error = None
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception as e:
        db_error = str(e)

    # 2. Storage writeability check
    storage_ok = True
    storage_info: Dict[str, Any] = {}
    for name, path in [
        ("inputs", settings.inputs_dir),
        ("outputs", settings.outputs_dir),
        ("thumbnails", settings.thumbnails_dir),
    ]:
        exists = path.exists()
        writable = os.access(path, os.W_OK) if exists else False
        storage_info[name] = {"exists": exists, "writable": writable, "path": str(path)}
        if not (exists and writable):
            storage_ok = False

    # Disk usage
    try:
        total, used, free = shutil.disk_usage(settings.base_storage_dir)
        storage_info["disk"] = {
            "total_gb": round(total / (1024**3), 2),
            "free_gb": round(free / (1024**3), 2),
            "used_pct": round((used / total) * 100, 1),
        }
    except Exception:
        pass

    # 3. Hardware device status
    device_info: Dict[str, Any] = {
        "configured_device": settings.MODEL_DEVICE,
        "cuda_available": False,
        "device_name": "CPU",
    }
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        device_info["cuda_available"] = cuda_avail
        if cuda_avail:
            device_info["device_name"] = torch.cuda.get_device_name(0)
            device_info["device_count"] = torch.cuda.device_count()
            device_info["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            )
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_info["device_name"] = "Apple Silicon MPS"
    except ImportError:
        device_info["torch_installed"] = False

    overall_healthy = db_ok and storage_ok

    return {
        "status": "healthy" if overall_healthy else "degraded",
        "service": settings.PROJECT_NAME,
        "environment": settings.APP_ENV,
        "mock_inference_mode": settings.USE_MOCK_INFERENCE,
        "database": {"connected": db_ok, "error": db_error},
        "storage": {"status": "ok" if storage_ok else "degraded", "details": storage_info},
        "hardware": device_info,
    }
