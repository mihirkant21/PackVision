import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient


def create_test_image(size=(256, 256), color=(255, 255, 255)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_projects_crud(client: TestClient):
    # 1. Create Project
    payload = {
        "name": "Luxury Perfume Bottle 2026",
        "description": "Premium cosmetic glass packaging project",
        "package_type": "bottle",
        "material": "glass",
        "finish": "glossy",
        "prompt": "Luxury perfume bottle, gold accents, studio lighting",
    }
    create_res = client.post("/api/v1/projects", json=payload)
    assert create_res.status_code == 201
    proj_data = create_res.json()
    proj_id = proj_data["id"]
    assert proj_data["name"] == "Luxury Perfume Bottle 2026"
    assert proj_data["package_type"] == "bottle"

    # 2. List Projects
    list_res = client.get("/api/v1/projects")
    assert list_res.status_code == 200
    list_data = list_res.json()
    assert list_data["total"] >= 1
    assert any(p["id"] == proj_id for p in list_data["items"])

    # 3. Get Project Detail
    get_res = client.get(f"/api/v1/projects/{proj_id}")
    assert get_res.status_code == 200
    detail = get_res.json()
    assert detail["id"] == proj_id
    assert "generations" in detail

    # 4. Update Project
    update_res = client.put(f"/api/v1/projects/{proj_id}", json={"name": "Updated Perfume Bottle"})
    assert update_res.status_code == 200
    assert update_res.json()["name"] == "Updated Perfume Bottle"

    # 5. Delete Project
    del_res = client.delete(f"/api/v1/projects/{proj_id}")
    assert del_res.status_code == 200

    # 6. Verify 404 after deletion
    get_deleted = client.get(f"/api/v1/projects/{proj_id}")
    assert get_deleted.status_code == 404


def test_preprocess_endpoint(client: TestClient, valid_png_bytes: bytes):
    # 1. Upload sketch
    files = {"file": ("sketch.png", valid_png_bytes, "image/png")}
    upload_res = client.post("/api/v1/uploads", files=files, data={"asset_type": "sketch"})
    assert upload_res.status_code == 201
    asset_id = upload_res.json()["asset"]["id"]

    # 2. Call /api/v1/preprocess
    prep_res = client.post(
        "/api/v1/preprocess",
        data={
            "asset_id": asset_id,
            "low_threshold": 50,
            "high_threshold": 150,
            "dilation": 1,
            "target_size": 512,
        },
    )
    assert prep_res.status_code == 200
    prep_data = prep_res.json()
    assert prep_data["success"] is True
    assert "maps" in prep_data
    assert "canny_url" in prep_data["maps"]
    assert "lineart_url" in prep_data["maps"]
    assert "depth_url" in prep_data["maps"]
    assert "original_url" in prep_data["maps"]

    # 3. Fetch preprocessed map images
    canny_res = client.get(prep_data["maps"]["canny_url"])
    assert canny_res.status_code == 200
    assert canny_res.headers["content-type"] == "image/png"


def test_composite_endpoint(client: TestClient, valid_png_bytes: bytes):
    # 1. Upload base image and logo
    base_file = {"file": ("render.png", valid_png_bytes, "image/png")}
    base_res = client.post("/api/v1/uploads", files=base_file, data={"asset_type": "sketch"})
    base_id = base_res.json()["asset"]["id"]

    logo_file = {"file": ("logo.png", valid_png_bytes, "image/png")}
    logo_res = client.post("/api/v1/uploads", files=logo_file, data={"asset_type": "logo"})
    logo_id = logo_res.json()["asset"]["id"]

    # 2. Call /api/v1/composite
    comp_payload = {
        "base_asset_id": base_id,
        "logo_asset_id": logo_id,
        "x": 0.5,
        "y": 0.45,
        "scale": 0.3,
        "rotation": 0.0,
        "opacity": 0.9,
        "blend_mode": "multiply",
        "use_cielab_texture": True,
        "save_as_asset": True,
    }
    comp_res = client.post("/api/v1/composite", json=comp_payload)
    assert comp_res.status_code == 200
    comp_data = comp_res.json()
    assert comp_data["success"] is True
    assert "url" in comp_data
    assert comp_data["asset_id"] is not None

    # 3. Fetch composited image
    fetch_res = client.get(comp_data["url"])
    assert fetch_res.status_code == 200
    assert fetch_res.headers["content-type"] == "image/png"


def test_evaluation_endpoint(client: TestClient, valid_png_bytes: bytes):
    # 1. Upload sketch and render
    sketch_file = {"file": ("sketch.png", valid_png_bytes, "image/png")}
    sketch_res = client.post("/api/v1/uploads", files=sketch_file, data={"asset_type": "sketch"})
    sketch_id = sketch_res.json()["asset"]["id"]

    render_file = {"file": ("render.png", valid_png_bytes, "image/png")}
    render_res = client.post("/api/v1/uploads", files=render_file, data={"asset_type": "sketch"})
    render_id = render_res.json()["asset"]["id"]

    # 2. Call /api/v1/evaluate
    eval_payload = {
        "sketch_asset_id": sketch_id,
        "render_asset_id": render_id,
        "prompt": "Premium folding carton packaging",
    }
    eval_res = client.post("/api/v1/evaluate", json=eval_payload)
    assert eval_res.status_code == 200
    eval_data = eval_res.json()
    assert eval_data["success"] is True
    assert "metrics" in eval_data
    assert "edge_ssim" in eval_data["metrics"]
    assert "edge_iou" in eval_data["metrics"]
    assert "structural_preservation_score" in eval_data["metrics"]
