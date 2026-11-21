# PackVision AI: API Reference

Base URL: `http://localhost:8000/api/v1`

All request/response shapes below are taken from the FastAPI routers in
[`backend/app/api/v1/`](file:///c:/Users/Lenovo/Downloads/GenAI%20Project/backend/app/api/v1)
and the Pydantic schemas in
[`backend/app/schemas/`](file:///c:/Users/Lenovo/Downloads/GenAI%20Project/backend/app/schemas).

> Endpoints marked **[new in working tree]** (cancel, plus the rate limiting
> behavior described in section 8) were added in the current working tree and
> are documented ahead of their next release.

---

## 1. System Endpoints

### GET `/health`
Returns system status, active inference device, and storage metrics
(`backend/app/api/v1/health.py`).

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
Retrieves the packaging taxonomy (shared source `src/taxonomy.py`).

**Response `200 OK`** (`TaxonomyResponse`, values abbreviated):
```json
{
  "package_types": [
    {"id": "folding_carton", "label": "Folding Carton", "description": "...", "icon": "..."}
  ],
  "materials": [
    {"id": "kraft_cardboard", "label": "Kraft Paperboard", "default_finish": "matte", "compatible_finishes": ["matte", "..."], "description": "..."}
  ],
  "finishes": [
    {"id": "gold_foil", "label": "Gold Hot Foil Stamping"}
  ],
  "viewpoints": [ {"id": "three_quarter", "label": "..."} ],
  "backgrounds": [ {"id": "seamless_studio", "label": "..."} ],
  "lighting_presets": [ {"id": "studio_softbox", "label": "..."} ]
}
```

---

## 2. Asset Endpoints

### POST `/uploads`
Uploads an image asset (sketch, reference swatch, or brand logo).

**Rate limited:** 10 requests/minute/IP (see section 8).

**Request**: `multipart/form-data`
- `file`: Binary image file (PNG, JPG, WEBP, max 25MB)
- `asset_type`: `sketch` | `reference` | `logo`

**Response `201 Created`** (`UploadResponse`):
```json
{
  "success": true,
  "message": "Asset uploaded successfully",
  "asset": {
    "id": "a9b8c7d6-1234-5678-9abc-def012345678",
    "user_id": null,
    "file_name": "tea_sketch.png",
    "asset_type": "sketch",
    "mime_type": "image/png",
    "width": 1024,
    "height": 1024,
    "file_size": 184520,
    "storage_key": "inputs/a9b8c7d6....png",
    "created_at": "2026-11-11T20:00:00Z",
    "deleted_at": null,
    "url": "/api/v1/files/a9b8c7d6-1234-5678-9abc-def012345678",
    "thumbnail_url": "/api/v1/files/a9b8c7d6-1234-5678-9abc-def012345678?thumb=true"
  }
}
```

Errors: `400` for unsupported mime type / oversize file.

### GET `/files/{asset_id}`
Serves the raw image file for an asset, or its 256x256 thumbnail when
`thumb=true` (`backend/app/api/v1/files.py`). If the thumbnail has not been
generated yet, the main file is served instead.

**Query parameters**:
- `thumb` (bool, default `false`): fetch the 256x256 PNG thumbnail.

**Response `200 OK`**: binary image (`FileResponse`) with the asset's stored
`mime_type` and `file_name`.

**Errors**: `404` if the asset does not exist (or is soft-deleted), or if its
file is missing on disk.

### GET `/samples`
Lists bundled example sketches for quick testing, plus the bundled sample
logo.

**Response `200 OK`**:
```json
{
  "samples": [
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
      "url": "/api/v1/samples/bottle_sketch.png"
    }
  ],
  "sample_logo": {
    "name": "Aura Botanica Logo",
    "filename": "sample_logo.png",
    "url": "/api/v1/samples/sample_logo.png"
  }
}
```

### GET `/samples/{filename}`
Serves a bundled sample sketch or the sample logo image file. `404` if the
filename does not exist on disk.

---

## 3. Generation Lifecycle Endpoints

### POST `/generations`
Queues a new sketch-to-photoreal rendering job and returns the full job
resource immediately (`status_code=202`).

**Rate limited:** 10 requests/minute/IP (see section 8).

**Request `application/json`** (`GenerationCreate`; defaults shown):
```json
{
  "input_asset_id": "a9b8c7d6-1234-5678-9abc-def012345678",
  "reference_asset_id": null,
  "logo_asset_id": null,
  "package_type": "bottle",
  "material": "glass",
  "finish": "glossy",
  "viewpoint": "front",
  "background": "studio_clean",
  "prompt": null,
  "negative_prompt": null,
  "seed": null,
  "num_outputs": 1,
  "guidance_scale": 7.5,
  "controlnet_scale": 0.85,
  "steps": 30,
  "logo_placement": null
}
```

Field constraints: `num_outputs` 1-4, `guidance_scale` 1.0-20.0,
`controlnet_scale` 0.0-1.5, `steps` 10-100.

**Response `202 Accepted`** (`GenerationResponse`, abbreviated):
```json
{
  "id": "c1d2e3f4-5678-90ab-cdef-123456789012",
  "user_id": "system",
  "input_asset_id": "a9b8c7d6-...",
  "reference_asset_id": null,
  "logo_asset_id": null,
  "status": "queued",
  "package_type": "bottle",
  "material": "glass",
  "finish": "glossy",
  "viewpoint": "front",
  "background": "studio_clean",
  "prompt": null,
  "negative_prompt": null,
  "seed": null,
  "model_name": "...",
  "model_revision": "...",
  "parameters_json": {"num_outputs": 1, "guidance_scale": 7.5, "controlnet_scale": 0.85, "steps": 30, "logo_placement": null},
  "progress": 0,
  "stage_message": "Job queued for processing",
  "error_code": null,
  "error_message": null,
  "created_at": "2026-11-11T20:00:00Z",
  "started_at": null,
  "completed_at": null,
  "outputs": [],
  "input_asset": { "...": "AssetResponse" }
}
```

**Errors**: `404` if `input_asset_id` / `reference_asset_id` / `logo_asset_id`
does not resolve to an existing (non-deleted) asset; `429` when rate limited.

---

### GET `/generations`
Lists recent generation jobs, newest first, with pagination and an optional
status filter.

**Rate limited:** 120 requests/minute/IP.

**Query parameters**:
- `limit` (int, 1-100, default `20`): items per page
- `offset` (int, default `0`): pagination offset
- `status` (optional string): filter by job status (e.g. `queued`, `running`, `completed`, `failed`, `cancelled`)

**Response `200 OK`** (`GenerationListResponse`):
```json
{
  "total": 42,
  "items": [ { "...": "GenerationResponse, same shape as POST /generations" } ]
}
```

---

### GET `/generations/{generation_id}`
Polls job status, progress percentage, stage message, error fields, and
completed output assets.

**Rate limited:** 120 requests/minute/IP.

**Response `200 OK` (when running)**:
```json
{
  "id": "c1d2e3f4-...",
  "status": "running",
  "progress": 60,
  "stage_message": "Synthesizing substrate and studio lighting...",
  "error_code": null,
  "error_message": null,
  "outputs": [],
  "...": "remaining GenerationResponse fields"
}
```

**Response `200 OK` (when completed)** — each output is a
`GenerationOutputDetail` whose nested `asset` carries the file URLs:
```json
{
  "id": "c1d2e3f4-...",
  "status": "completed",
  "progress": 100,
  "stage_message": "Render completed successfully",
  "completed_at": "2026-11-11T20:00:04Z",
  "outputs": [
    {
      "id": "d1e2f3a4-1234-5678-90ab-cdef12345678",
      "generation_id": "c1d2e3f4-...",
      "asset_id": "e2f3a4b5-5678-90ab-cdef-123456789012",
      "is_selected": true,
      "metrics": {"edge_ssim": 0.864, "clip_score": 0.328},
      "created_at": "2026-11-11T20:00:04Z",
      "asset": {
        "id": "e2f3a4b5-...",
        "url": "/api/v1/files/e2f3a4b5-5678-90ab-cdef-123456789012",
        "thumbnail_url": "/api/v1/files/e2f3a4b5-5678-90ab-cdef-123456789012?thumb=true",
        "...": "remaining AssetResponse fields"
      }
    }
  ]
}
```

**Errors**: `404` if the generation id is unknown.

---

### POST `/generations/{generation_id}/retry` **[new in working tree]**
Retries a failed generation job. Retry attempts are capped (max 3 per job,
persisted in the job's `parameters_json`).

**Response `200 OK`** (`JobStatusResponse`):
```json
{
  "id": "c1d2e3f4-...",
  "status": "queued",
  "progress": 0,
  "stage_message": "Job queued for processing",
  "error_code": null,
  "error_message": null,
  "created_at": "2026-11-11T20:00:00Z",
  "started_at": null,
  "completed_at": null
}
```

**Errors**: `404` unknown id; `409` when the retry cap is exhausted
(`RetryLimitExceeded`).

---

### POST `/generations/{generation_id}/cancel` **[new in working tree]**
Requests cooperative cancellation of a queued or running job. The background
worker stops at the next stage boundary and the job transitions to the
terminal `cancelled` state. The response carries the job's current DB status,
which may briefly remain active until the worker observes the cancel flag.

**Response `200 OK`**: `JobStatusResponse` (same shape as retry).

**Errors**: `404` unknown id; `409` if the job is already in a terminal state
(`completed`, `failed`, `cancelled`) and cannot be cancelled.

---

### DELETE `/generations/{generation_id}`
Deletes a generation job and its associated output rows. If a worker is still
processing the job, a cooperative cancel is signalled first (best-effort) and
the row is then removed.

**Response `200 OK`**:
```json
{
  "success": true,
  "message": "Generation 'c1d2e3f4-...' deleted successfully."
}
```

**Errors**: `404` unknown id.

---

### POST `/generations/{generation_id}/feedback`
Captures reviewer rating and structured quality assessment.

**Request `application/json`** (`FeedbackCreate`):
```json
{
  "rating": 5,
  "issue_categories": ["geometry_distortion", "logo_misalignment"],
  "comment": "Outstanding foil reflection and realistic kraft paper texture."
}
```

Field constraints: `rating` int 1-5; `issue_categories` optional list of
strings; `comment` optional string.

**Response `201 Created`** (`FeedbackResponse`):
```json
{
  "id": "f1a2b3c4-5678-90ab-cdef-1234567890ab",
  "generation_id": "c1d2e3f4-...",
  "user_id": "system",
  "rating": 5,
  "issue_categories": ["geometry_distortion"],
  "comment": "Outstanding foil reflection and realistic kraft paper texture.",
  "created_at": "2026-11-11T20:05:00Z"
}
```

**Errors**: `404` unknown generation id; `422` for schema violations.

---

## 4. Job Status Values

| Status | Meaning |
| :--- | :--- |
| `queued` | Job accepted and waiting for the background worker |
| `running` | Worker is rendering (`progress` 0-100, `stage_message` updates) |
| `completed` | Render finished; `outputs` populated |
| `failed` | Worker error; `error_code` / `error_message` set; retryable via `/retry` |
| `cancelled` | Terminal state after a successful cancellation |

---

## 5. Rate Limiting **[new in working tree]**

The backend applies an in-memory, dependency-free sliding-window rate limiter
keyed by client IP (`backend/app/core/rate_limit.py`). State is process-local
and resets on restart.

| Scope | Limit (default) | Env override |
| :--- | :---: | :--- |
| `POST /uploads` | 10 req/min | `RATE_LIMIT_UPLOADS_PER_MIN` |
| `POST /generations` | 10 req/min | `RATE_LIMIT_GENERATIONS_PER_MIN` |
| All other (read) endpoints | 120 req/min | `RATE_LIMIT_READ_PER_MIN` |

When the limit is exhausted the endpoint responds **`429 Too Many Requests`**
with a `Retry-After` header (seconds until the sliding window frees up).
