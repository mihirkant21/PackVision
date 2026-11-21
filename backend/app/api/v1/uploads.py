import io
import time
import logging
import os
import tempfile
from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.schemas.asset import UploadResponse, AssetResponse
from backend.app.db.models import Asset
from backend.app.services.asset_service import AssetService

logger = logging.getLogger(__name__)

router = APIRouter()

VALID_UPLOAD_ASSET_TYPES = {"sketch", "reference", "logo"}


@router.post(
    "/uploads",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload image asset (sketch, reference moodboard, or vector logo)",
    description="Validates image magic bytes, MIME type, dimensions (128-4096px), auto-corrects EXIF orientation, creates 256x256 thumbnail, and persists to storage.",
)
async def upload_asset(
    file: UploadFile = File(..., description="Image binary file to upload"),
    asset_type: str = Form(
        "sketch",
        description="Type of asset: 'sketch', 'reference', or 'logo'",
    ),
    user_id: Optional[str] = Form(
        None, description="Optional user identifier associated with asset"
    ),
    db: Session = Depends(get_db),
) -> UploadResponse:
    norm_type = asset_type.strip().lower()
    if norm_type not in VALID_UPLOAD_ASSET_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid asset_type '{asset_type}'. Must be one of: {sorted(list(VALID_UPLOAD_ASSET_TYPES))}",
        )

    asset = await AssetService.process_and_save_upload(
        db=db,
        upload_file=file,
        asset_type=norm_type,
        user_id=user_id,
    )

    return UploadResponse(
        success=True,
        message=f"{norm_type.capitalize()} uploaded and processed successfully.",
        asset=AssetResponse.from_orm_model(asset),
    )


# ---------------------------------------------------------------------------
# Auto-detection: inspect an uploaded image and infer the packaging config.
# Uses the shared VLM client (Gemini free tier) + the sketch/photo detector.
# ---------------------------------------------------------------------------

_analyze_limiter_state: dict = {}


@router.post(
    "/uploads/{asset_id}/analyze",
    summary="Auto-detect packaging configuration from a stored asset",
    description=(
        "Sends the stored image to the vision model and infers input_mode "
        "(sketch vs photo), package form, material, finish, lighting, viewpoint "
        "and background as canonical taxonomy keys. Falls back to the local "
        "sketch/photo detector for input_mode when the VLM is unavailable."
    ),
)
async def analyze_stored_asset(
    asset_id: str,
    db: Session = Depends(get_db),
):
    import asyncio

    asset = db.query(Asset).filter(Asset.id == asset_id).first()
    if asset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found.")
    if (asset.deleted_at is not None):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset was deleted.")

    image_path = AssetService.get_asset_file_path(asset)
    if not image_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset file missing from storage.")

    # Run the (blocking) VLM call off the event loop.
    return await asyncio.to_thread(_analyze_image_sync, image_path)


@router.post(
    "/uploads/analyze",
    summary="Auto-detect packaging configuration from a direct image upload",
    description=(
        "Layer 1 (always, <200ms): local trained classifier fills input_mode, "
        "package_type, material/finish (photo inputs) + a suggested_prompt. "
        "Layer 2 (enrich=true, slower, best-effort): the vision model adds "
        "lighting/viewpoint/background and upgrades material/finish for "
        "sketches. The frontend calls layer 1 first for an instant result, "
        "then layer 2 as a background polish."
    ),
)
async def analyze_uploaded_image(
    file: UploadFile = File(..., description="Image binary to analyze"),
    enrich: bool = Form(True, description="Run the (slower) VLM enrichment layer"),
    db: Session = Depends(get_db),
):
    import asyncio

    from PIL import Image as PILImage

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty upload.")
    try:
        PILImage.open(io.BytesIO(contents)).verify()  # magic-byte sanity
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported or corrupt image.")

    fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix="packvision_analyze_")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(contents)
        # Run the (blocking) classifier/VLM call off the event loop.
        return await asyncio.to_thread(_analyze_image_sync, tmp_path, enrich)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _analyze_image_sync(image_path, enrich: bool = True):
    from PIL import Image

    from src import taxonomy
    from src.config_classifier import predict_config
    from src.photo_mode import detect_input_mode
    from src.vlm_client import ask_vlm, vlm_available

    # Downscale: identification needs ~384px; keeps the request well inside
    # free-tier per-minute token budgets (512-1024px images get 429-throttled).
    img = Image.open(image_path).convert("RGB").resize((384, 384), Image.Resampling.LANCZOS) if min(Image.open(image_path).size) > 384 else Image.open(image_path).convert("RGB")

    # Layer 1 — instant local detection (no network, always available):
    #   input_mode from the sketch/photo detector, category from the trained
    #   config classifier (98.6% val accuracy). material/finish from the
    #   classifier are reliable for photos; sketches carry no texture signal,
    #   so those stay None until the VLM layer enriches them.
    detected_mode = detect_input_mode(img)
    classified = predict_config(img)

    result = {
        "input_mode": detected_mode,
        "package_type": classified.get("category"),
        "material": classified.get("material") if detected_mode == "photo" else None,
        "finish": classified.get("finish") if detected_mode == "photo" else None,
        "lighting": None,
        "viewpoint": None,
        "background": None,
        "source": "classifier",
        "detected_summary": "",
        "category_confidence": classified.get("category_confidence"),
        "material_confidence": classified.get("material_confidence") if detected_mode == "photo" else None,
        "finish_confidence": classified.get("finish_confidence") if detected_mode == "photo" else None,
    }
    matched = [x for x in (result["package_type"], result["material"], result["finish"]) if x]
    result["detected_summary"] = " · ".join(x.replace("_", " ") for x in matched[:3])

    # Suggested prompt composed from the detected configuration — the studio
    # auto-fills the prompt field with this so the render matches the image.
    fragments = []
    pt_label = (taxonomy.entry("PACKAGE_TYPES", result["package_type"]) or {}).get("prompt_fragment", "")
    if pt_label:
        fragments.append(f"a {result['package_type'].replace('_', ' ')}")
    mat_label = (taxonomy.entry("MATERIALS", result["material"]) or {}).get("prompt_fragment", "")
    if mat_label:
        fragments.append(mat_label)
    fin_label = (taxonomy.entry("FINISHES", result["finish"]) or {}).get("prompt_fragment", "")
    if fin_label:
        fragments.append(f"{fin_label} finish")
    result["suggested_prompt"] = (
        "Photorealistic commercial packaging render of " + ", ".join(fragments)
        + ", studio product lighting, seamless background, sharp focus, 8k quality."
        if fragments
        else ""
    )

    # Layer 2 — VLM enrichment (best-effort, only when enrich=true): fills
    # lighting/viewpoint/background (which the classifier cannot see) and
    # improves material/finish for sketch inputs, where lineart carries no
    # texture signal. Everything here is optional: the classifier results
    # above already stand and are returned immediately when enrich=false.
    vlm_enriched = False
    if not enrich:
        result["note"] = None
        return result
    if not vlm_available():
        result["note"] = "Vision model unavailable — keep material/finish chosen manually."
        return result

    def keys_of(category: str) -> str:
        return ", ".join(taxonomy.all_keys(category))

    prompt = (
        "You are a packaging-design expert. Look at this image and identify the packaging "
        "configuration. Answer with ONLY a JSON object (no markdown, no explanation) using "
        "exactly these keys and values chosen from the allowed lists:\n"
        f'{{"input_mode": "sketch|photo", '
        f'"package_type": one of [{keys_of("PACKAGE_TYPES")}], '
        f'"material": one of [{keys_of("MATERIALS")}], '
        f'"finish": one of [{keys_of("FINISHES")}], '
        f'"lighting": one of [{keys_of("LIGHTING")}], '
        f'"viewpoint": one of [{keys_of("CAMERA_ANGLES")}], '
        f'"background": one of [{keys_of("BACKGROUNDS")}]}}\n'
        "input_mode is \"sketch\" if the image is a line drawing or hand sketch, \"photo\" "
        "if it is a real photograph or photorealistic render. Choose the closest match for "
        "every field; never invent values."
    )

    # Free-tier VLM providers are intermittently rate-limited; one bounded
    # retry (after a short pause that lets the 60s negative cache expire for
    # fresh requests) dramatically improves the odds of a full detection.
    raw = ask_vlm(prompt, image=img, json_mode=True, max_tokens=300)
    if not raw:
        time.sleep(12)  # let upstream per-minute windows reset
        raw = ask_vlm(prompt, image=img, json_mode=True, max_tokens=300)
    if not raw:
        result["note"] = "Vision model unavailable — keep material/finish chosen manually."
        return result

    # Robust JSON extraction (models sometimes wrap in fences).
    import json as _json
    import re as _re

    cleaned = _re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=_re.MULTILINE).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        result["note"] = "Vision model returned unreadable output."
        return result
    try:
        parsed = _json.loads(cleaned[start : end + 1])
    except _json.JSONDecodeError:
        result["note"] = "Vision model returned unreadable output."
        return result

    # Merge: VLM fills lighting/viewpoint/background (classifier can't know
    # them) and upgrades material/finish for SKETCH inputs (no texture signal
    # locally). For photos, classifier material/finish already stand.
    field_map = {
        "package_type": "PACKAGE_TYPES",
        "material": "MATERIALS",
        "finish": "FINISHES",
        "lighting": "LIGHTING",
        "viewpoint": "CAMERA_ANGLES",
        "background": "BACKGROUNDS",
        "input_mode": None,
    }
    matched = []
    for field, category in field_map.items():
        value = str(parsed.get(field) or "").strip()
        if not value:
            continue
        if category:
            # only fill fields the classifier could not determine
            if result.get(field):
                continue
            canonical = taxonomy.normalize(value, category)
            if canonical:
                result[field] = canonical
                matched.append(canonical.replace("_", " "))
        elif value.lower() in ("sketch", "photo"):
            result[field] = value.lower()

    if matched:
        vlm_enriched = True
        result["source"] = "classifier+vlm"
        result["detected_summary"] = " · ".join(x.replace("_", " ") for x in matched[:3])
    result["note"] = None if vlm_enriched else "Vision model output unusable — keep material/finish chosen manually."
    return result

    return result
