# PackVision AI: API Reference

Base URL: `http://localhost:8000/api/v1`

---

## 1. System Endpoints

### GET `/health`
Returns system status, active inference device, and storage metrics.

**Response `200 OK`**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "device": "cpu",
  "cuda_available": false,
  "mock_inference": true,
  "storage": {
    "inputs_count": 12,
    "outputs_count": 8
  }
}
```

### GET `/taxonomy`
Retrieves all supported packaging categories, materials, finishes, lighting environments, and camera angles.

**Response `200 OK`**:
```json
{
  "categories": [
    {"id": "folding_carton", "name": "Folding Carton", "description": "Standard paperboard consumer goods box"},
    {"id": "standup_pouch", "name": "Stand-Up Pouch", "description": "Flexible barrier packaging for snacks and powders"},
    {"id": "cosmetic_bottle", "name": "Cosmetic Bottle", "description": "Glass or plastic dropper/pump bottle"}
  ],
  "materials": [
    {"id": "kraft_cardboard", "name": "Kraft Paperboard", "finish_type": "matte"},
    {"id": "corrugated_box", "name": "E-Flute Corrugated", "finish_type": "textured"},
    {"id": "matte_carton", "name": "Matte Folding Carton", "finish_type": "soft_touch"},
    {"id": "frosted_glass", "name": "Frosted Glass", "finish_type": "translucent"}
  ],
  "finishes": [
    {"id": "gold_foil", "name": "Gold Hot Foil Stamping"},
    {"id": "spot_uv", "name": "Spot UV Gloss Varnish"},
    {"id": "embossed", "name": "3D Embossed Relief"}
  ]
}
```

---

## 2. Asset Upload Endpoint

### POST `/uploads`
Uploads an image asset (sketch, reference swatch, or brand logo).

**Request**: `multipart/form-data`
- `file`: Binary image file (PNG, JPG, WEBP, max 25MB)
- `asset_type`: `sketch` | `reference` | `logo`

**Response `201 Created`**:
```json
{
  "asset_id": "a9b8c7d6-1234-5678-9abc-def012345678",
  "file_name": "tea_sketch.png",
  "asset_type": "sketch",
  "mime_type": "image/png",
  "width": 1024,
  "height": 1024,
  "file_size": 184520,
  "url": "/api/v1/files/a9b8c7d6-1234-5678-9abc-def012345678",
  "thumbnail_url": "/api/v1/files/a9b8c7d6-1234-5678-9abc-def012345678?thumb=true"
}
```

---

## 3. Generation Lifecycle Endpoints

### POST `/generations`
Queues a new sketch-to-photoreal rendering job.

**Request `application/json`**:
```json
{
  "input_asset_id": "a9b8c7d6-1234-5678-9abc-def012345678",
  "logo_asset_id": "b1c2d3e4-5678-90ab-cdef-1234567890ab",
  "package_type": "folding_carton",
  "material": "kraft_cardboard",
  "finish": "gold_foil",
  "viewpoint": "three_quarter",
  "background": "neutral_studio",
  "prompt": "Luxury herbal tea box, minimalist botanical accents",
  "negative_prompt": "extra flaps, distorted seams",
  "seed": 42,
  "num_inference_steps": 25,
  "controlnet_conditioning_scale": 0.85
}
```

**Response `202 Accepted`**:
```json
{
  "job_id": "c1d2e3f4-5678-90ab-cdef-123456789012",
  "status": "queued",
  "progress": 0,
  "stage_message": "Generation job queued",
  "created_at": "2026-11-09T23:14:00Z"
}
```

---

### GET `/generations/{job_id}`
Polls job status, progress percentage, stage message, and completed output assets.

**Response `200 OK` (when running)**:
```json
{
  "job_id": "c1d2e3f4-5678-90ab-cdef-123456789012",
  "status": "generating",
  "progress": 60,
  "stage_message": "Synthesizing kraft_cardboard substrate and studio lighting...",
  "outputs": []
}
```

**Response `200 OK` (when completed)**:
```json
{
  "job_id": "c1d2e3f4-5678-90ab-cdef-123456789012",
  "status": "completed",
  "progress": 100,
  "stage_message": "Render completed successfully",
  "outputs": [
    {
      "output_id": "d1e2f3a4-1234-5678-90ab-cdef12345678",
      "asset_id": "e2f3a4b5-5678-90ab-cdef-123456789012",
      "url": "/api/v1/files/e2f3a4b5-5678-90ab-cdef-123456789012",
      "thumbnail_url": "/api/v1/files/e2f3a4b5-5678-90ab-cdef-123456789012?thumb=true",
      "metrics": {
        "edge_ssim": 0.864,
        "clip_score": 0.328
      }
    }
  ],
  "completed_at": "2026-11-09T23:14:04Z"
}
```

---

### POST `/generations/{job_id}/feedback`
Captures reviewer rating and structured quality assessment.

**Request `application/json`**:
```json
{
  "rating": 5,
  "issue_categories": {
    "shape_preserved": true,
    "material_realistic": true,
    "finish_accurate": true,
    "lighting_convincing": true,
    "logo_crisp": true
  },
  "comment": "Outstanding foil reflection and realistic kraft paper texture."
}
```

**Response `200 OK`**:
```json
{
  "feedback_id": "f1a2b3c4-5678-90ab-cdef-1234567890ab",
  "status": "recorded"
}
```
