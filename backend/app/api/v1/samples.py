from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()

SAMPLES_DIR = Path("data/samples").resolve()

SAMPLE_CATALOG = [
    {
        "id": "carton",
        "name": "Luxury Folding Carton",
        "package_type": "box",
        "material": "cardboard",
        "finish": "gold_foil",
        "viewpoint": "three_quarter",
        "background": "studio_clean",
        "prompt": "Luxury herbal tea carton packaging, gold hot-foil debossed botanical logo",
        "filename": "carton_sketch.png",
        "url": "/api/v1/samples/carton_sketch.png",
    },
    {
        "id": "pouch",
        "name": "Stand-Up Matte Pouch",
        "package_type": "pouch",
        "material": "kraft_paper",
        "finish": "matte",
        "viewpoint": "front",
        "background": "studio_clean",
        "prompt": "Artisanal organic ground coffee stand-up pouch, earth tones, minimalist label",
        "filename": "pouch_sketch.png",
        "url": "/api/v1/samples/pouch_sketch.png",
    },
    {
        "id": "bottle",
        "name": "Cosmetic Serum Bottle",
        "package_type": "bottle",
        "material": "glass",
        "finish": "frosted",
        "viewpoint": "front",
        "background": "studio_clean",
        "prompt": "Luxury cosmetic dropper bottle, frosted amber glass, gold metallic collar",
        "filename": "bottle_sketch.png",
        "url": "/api/v1/samples/bottle_sketch.png",
    },
]


@router.get(
    "/samples",
    summary="List available sample sketches for quick testing",
)
def list_samples():
    return {
        "samples": SAMPLE_CATALOG,
        "sample_logo": {
            "name": "Aura Botanica Logo",
            "filename": "sample_logo.png",
            "url": "/api/v1/samples/sample_logo.png",
        },
    }


@router.get(
    "/samples/{filename}",
    summary="Get sample sketch or logo image file",
)
def get_sample_file(filename: str):
    file_path = SAMPLES_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Sample file '{filename}' not found.")
    
    media_type = "image/png" if filename.endswith(".png") else "image/jpeg"
    return FileResponse(path=str(file_path), media_type=media_type, filename=filename)
