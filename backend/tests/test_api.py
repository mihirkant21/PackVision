import time
import pytest
from fastapi.testclient import TestClient


def test_root_endpoint(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "PackVision AI"
    assert data["status"] == "operational"
    assert data["docs_url"] == "/docs"


def test_health_check(client: TestClient):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert "database" in data
    assert data["database"]["connected"] is True
    assert "storage" in data
    assert data["storage"]["status"] == "ok"
    assert "hardware" in data
    assert "mock_inference_mode" in data


def test_taxonomy_endpoint(client: TestClient):
    response = client.get("/api/v1/taxonomy")
    assert response.status_code == 200
    data = response.json()
    assert "package_types" in data
    assert len(data["package_types"]) >= 5
    assert any(pt["id"] == "bottle" for pt in data["package_types"])
    assert any(pt["id"] == "box" for pt in data["package_types"])

    assert "materials" in data
    assert len(data["materials"]) >= 4
    assert any(m["id"] == "glass" for m in data["materials"])

    assert "finishes" in data
    assert any(f["id"] == "matte" for f in data["finishes"])

    assert "viewpoints" in data
    assert any(v["id"] == "front" for v in data["viewpoints"])

    assert "backgrounds" in data
    assert any(b["id"] == "studio_clean" for b in data["backgrounds"])

    assert "lighting_presets" in data


def test_upload_valid_png(client: TestClient, valid_png_bytes: bytes):
    files = {"file": ("test_sketch.png", valid_png_bytes, "image/png")}
    data = {"asset_type": "sketch"}
    response = client.post("/api/v1/uploads", files=files, data=data)
    assert response.status_code == 201
    res_data = response.json()
    assert res_data["success"] is True
    asset = res_data["asset"]
    assert asset["asset_type"] == "sketch"
    assert asset["mime_type"] == "image/png"
    assert asset["width"] == 256
    assert asset["height"] == 256
    assert asset["file_size"] > 0
    assert "url" in asset
    assert "thumbnail_url" in asset


def test_upload_valid_jpeg(client: TestClient, valid_jpeg_bytes: bytes):
    files = {"file": ("moodboard.jpg", valid_jpeg_bytes, "image/jpeg")}
    data = {"asset_type": "reference"}
    response = client.post("/api/v1/uploads", files=files, data=data)
    assert response.status_code == 201
    res_data = response.json()
    assert res_data["asset"]["asset_type"] == "reference"
    assert res_data["asset"]["mime_type"] == "image/jpeg"


def test_upload_corrupted_magic_bytes(client: TestClient, corrupted_image_bytes: bytes):
    files = {"file": ("fake.png", corrupted_image_bytes, "image/png")}
    data = {"asset_type": "sketch"}
    response = client.post("/api/v1/uploads", files=files, data=data)
    assert response.status_code == 400
    assert "magic bytes" in response.json()["detail"].lower()


def test_upload_unsupported_mime(client: TestClient):
    files = {"file": ("document.pdf", b"%PDF-1.5 test document", "application/pdf")}
    data = {"asset_type": "sketch"}
    response = client.post("/api/v1/uploads", files=files, data=data)
    assert response.status_code == 400


def test_upload_too_small_dimensions(client: TestClient, small_image_bytes: bytes):
    files = {"file": ("tiny.png", small_image_bytes, "image/png")}
    data = {"asset_type": "sketch"}
    response = client.post("/api/v1/uploads", files=files, data=data)
    assert response.status_code == 400
    assert "minimum" in response.json()["detail"].lower()


def test_upload_invalid_asset_type(client: TestClient, valid_png_bytes: bytes):
    files = {"file": ("test.png", valid_png_bytes, "image/png")}
    data = {"asset_type": "invalid_type"}
    response = client.post("/api/v1/uploads", files=files, data=data)
    assert response.status_code == 400
    assert "invalid asset_type" in response.json()["detail"].lower()


def test_file_and_thumbnail_serving(client: TestClient, valid_png_bytes: bytes):
    # 1. Upload asset
    files = {"file": ("serve_test.png", valid_png_bytes, "image/png")}
    upload_res = client.post(
        "/api/v1/uploads", files=files, data={"asset_type": "sketch"}
    )
    asset_id = upload_res.json()["asset"]["id"]

    # 2. Fetch raw file
    file_res = client.get(f"/api/v1/files/{asset_id}")
    assert file_res.status_code == 200
    assert file_res.headers["content-type"] == "image/png"
    assert len(file_res.content) == len(valid_png_bytes)

    # 3. Fetch thumbnail
    thumb_res = client.get(f"/api/v1/files/{asset_id}?thumb=true")
    assert thumb_res.status_code == 200
    assert thumb_res.headers["content-type"] == "image/png"
    assert len(thumb_res.content) > 0


def test_get_nonexistent_file(client: TestClient):
    response = client.get("/api/v1/files/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


def test_generation_lifecycle(client: TestClient, valid_png_bytes: bytes):
    # 1. Upload sketch asset
    files = {"file": ("input_sketch.png", valid_png_bytes, "image/png")}
    upload_res = client.post(
        "/api/v1/uploads", files=files, data={"asset_type": "sketch"}
    )
    assert upload_res.status_code == 201
    sketch_id = upload_res.json()["asset"]["id"]

    # 2. Upload optional logo asset
    logo_files = {"file": ("brand_logo.png", valid_png_bytes, "image/png")}
    logo_res = client.post(
        "/api/v1/uploads", files=logo_files, data={"asset_type": "logo"}
    )
    assert logo_res.status_code == 201
    logo_id = logo_res.json()["asset"]["id"]

    # 3. Create Generation Job
    gen_payload = {
        "input_asset_id": sketch_id,
        "logo_asset_id": logo_id,
        "package_type": "bottle",
        "material": "glass",
        "finish": "glossy",
        "viewpoint": "front",
        "background": "studio_clean",
        "prompt": "Luxury cold-brew glass bottle with sleek amber tint",
        "seed": 42,
        "num_outputs": 2,
    }
    create_res = client.post("/api/v1/generations", json=gen_payload)
    assert create_res.status_code == 202
    gen_data = create_res.json()
    gen_id = gen_data["id"]
    assert gen_data["package_type"] == "bottle"
    assert gen_data["status"] in ("queued", "preprocessing", "generating", "completed")

    # 4. Poll until job reaches terminal state ('completed' or 'failed')
    max_wait_seconds = 10
    start_time = time.time()
    final_status = None
    gen_details = None

    while time.time() - start_time < max_wait_seconds:
        status_res = client.get(f"/api/v1/generations/{gen_id}")
        assert status_res.status_code == 200
        gen_details = status_res.json()
        final_status = gen_details["status"]
        if final_status in ("completed", "failed"):
            break
        time.sleep(0.3)

    assert final_status == "completed", f"Job ended in status: {final_status}"
    assert gen_details["progress"] == 100
    assert len(gen_details["outputs"]) == 2
    for out in gen_details["outputs"]:
        assert out["asset_id"] is not None
        assert out["asset"] is not None
        assert "url" in out["asset"]
        assert "thumbnail_url" in out["asset"]

    # 5. List generations
    list_res = client.get("/api/v1/generations")
    assert list_res.status_code == 200
    list_data = list_res.json()
    assert list_data["total"] >= 1
    assert any(item["id"] == gen_id for item in list_data["items"])

    # 6. Submit feedback
    feedback_payload = {
        "rating": 5,
        "issue_categories": [],
        "comment": "Incredible glass reflection rendering!",
    }
    fb_res = client.post(
        f"/api/v1/generations/{gen_id}/feedback", json=feedback_payload
    )
    assert fb_res.status_code == 201
    fb_data = fb_res.json()
    assert fb_data["rating"] == 5
    assert fb_data["generation_id"] == gen_id

    # 7. Retry generation
    retry_res = client.post(f"/api/v1/generations/{gen_id}/retry")
    assert retry_res.status_code == 200
    assert retry_res.json()["status"] in ("queued", "preprocessing")

    # 8. Delete generation
    del_res = client.delete(f"/api/v1/generations/{gen_id}")
    assert del_res.status_code == 200
    assert del_res.json()["success"] is True

    # 9. Verify 404 after deletion
    get_del_res = client.get(f"/api/v1/generations/{gen_id}")
    assert get_del_res.status_code == 404


def test_generation_invalid_input_asset(client: TestClient):
    payload = {
        "input_asset_id": "00000000-0000-0000-0000-000000000000",
        "package_type": "box",
    }
    response = client.post("/api/v1/generations", json=payload)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_generation_feedback_validation(client: TestClient, valid_png_bytes: bytes):
    # Upload sketch & create generation
    files = {"file": ("sketch.png", valid_png_bytes, "image/png")}
    upload_res = client.post(
        "/api/v1/uploads", files=files, data={"asset_type": "sketch"}
    )
    sketch_id = upload_res.json()["asset"]["id"]
    gen_res = client.post(
        "/api/v1/generations",
        json={"input_asset_id": sketch_id, "package_type": "can"},
    )
    gen_id = gen_res.json()["id"]

    # Rating out of bounds (> 5) -> should fail validation
    invalid_fb = {"rating": 6, "comment": "Too high"}
    bad_res = client.post(f"/api/v1/generations/{gen_id}/feedback", json=invalid_fb)
    assert bad_res.status_code == 422

    # Clean up generation
    client.delete(f"/api/v1/generations/{gen_id}")
