"""Job lifecycle robustness tests: startup recovery, cooperative cancellation,
retry cap, rate limiting, and concurrent job handling.

No real inference is executed here: the heavy rendering engine is bypassed by
monkeypatching inference_service.generate with deterministic fakes, keeping the
whole module well under the 60s budget.
"""

import threading
import time
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.db.models import Generation
from backend.app.services.inference_service import inference_service
from backend.app.services.job_service import job_service

TERMINAL = ("completed", "failed", "cancelled")
TERMINAL_POLL_TIMEOUT = 20.0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_generation(
    db_session,
    *,
    status: str = "failed",
    created_at: datetime = None,
    parameters_json: dict = None,
) -> Generation:
    """Insert a Generation row directly (bypassing the API) for state tests."""
    gen = Generation(
        id=str(uuid.uuid4()),
        user_id=None,
        # FK constraints are not enforced by the SQLite test engine, so a
        # synthetic asset id is fine for rows no worker will execute.
        input_asset_id="00000000-0000-0000-0000-000000000000",
        status=status,
        package_type="bottle",
        prompt=None,
        parameters_json=parameters_json if parameters_json is not None else {},
        progress=0,
        stage_message="inserted by test",
        created_at=created_at or datetime.now(timezone.utc),
    )
    db_session.add(gen)
    db_session.commit()
    db_session.refresh(gen)
    return gen


def poll_terminal(client: TestClient, generation_id: str, timeout: float = TERMINAL_POLL_TIMEOUT):
    """Poll GET /generations/{id} until a terminal state; returns last payload."""
    deadline = time.time() + timeout
    payload = None
    while time.time() < deadline:
        resp = client.get(f"/api/v1/generations/{generation_id}")
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        if payload["status"] in TERMINAL:
            return payload
        time.sleep(0.1)
    return payload


def upload_sketch(client: TestClient, valid_png_bytes: bytes) -> str:
    files = {"file": (f"sketch_{uuid.uuid4().hex[:6]}.png", valid_png_bytes, "image/png")}
    resp = client.post("/api/v1/uploads", files=files, data={"asset_type": "sketch"})
    assert resp.status_code == 201, resp.text
    return resp.json()["asset"]["id"]


def create_generation(client: TestClient, sketch_id: str) -> str:
    resp = client.post(
        "/api/v1/generations",
        json={"input_asset_id": sketch_id, "package_type": "bottle", "num_outputs": 1},
    )
    assert resp.status_code == 202, resp.text
    return resp.json()["id"]


# --------------------------------------------------------------------------- #
# Startup recovery (recover_stuck_jobs)
# --------------------------------------------------------------------------- #
def test_recover_stuck_jobs_marks_orphans_failed(db_session):
    orphan = make_generation(
        db_session,
        status="generating",
        created_at=datetime.now(timezone.utc) - timedelta(hours=2),
    )
    stale_queued = make_generation(
        db_session,
        status="queued",
        created_at=datetime.now(timezone.utc) - timedelta(hours=3),
    )
    fresh = make_generation(
        db_session,
        status="generating",
        created_at=datetime.now(timezone.utc),  # created during THIS process run
    )
    done = make_generation(db_session, status="completed")

    recovered = job_service.recover_stuck_jobs()
    assert recovered >= 2  # at least the two orphans inserted above

    db_session.expire_all()
    db_orphan = db_session.query(Generation).filter(Generation.id == orphan.id).first()
    assert db_orphan.status == "failed"
    assert db_orphan.error_code == "interrupted"
    assert "Server restarted during processing" in (db_orphan.error_message or "")
    assert db_orphan.completed_at is not None

    db_queued = db_session.query(Generation).filter(Generation.id == stale_queued.id).first()
    assert db_queued.status == "failed"
    assert db_queued.error_code == "interrupted"

    # Jobs created after process start and terminal jobs must be untouched.
    db_fresh = db_session.query(Generation).filter(Generation.id == fresh.id).first()
    assert db_fresh.status == "generating"
    db_done = db_session.query(Generation).filter(Generation.id == done.id).first()
    assert db_done.status == "completed"


# --------------------------------------------------------------------------- #
# Cancellation
# --------------------------------------------------------------------------- #
def test_cancel_active_job_reaches_cancelled_state(client, db_session, valid_png_bytes, monkeypatch):
    sketch_id = upload_sketch(client, valid_png_bytes)

    entered_generate = threading.Event()
    release = threading.Event()

    def blocked_generate(*args, **kwargs):
        entered_generate.set()
        release.wait(timeout=30)
        return [
            {
                "image": Image.new("RGB", (64, 64), (12, 34, 56)),
                "metrics": {"engine": "fake", "is_mock": True},
            }
        ]

    monkeypatch.setattr(
        inference_service, "generate", blocked_generate
    )

    gen_id = create_generation(client, sketch_id)

    # Wait until the worker is inside the (blocked) generation stage.
    assert entered_generate.wait(timeout=20), "worker never reached the generation stage"

    resp = client.post(f"/api/v1/generations/{gen_id}/cancel")
    assert resp.status_code == 200, resp.text
    assert resp.json()["id"] == gen_id
    assert resp.json()["status"] in ("generating", "preprocessing", "cancelled", "queued")

    release.set()
    final = poll_terminal(client, gen_id)
    assert final is not None and final["status"] == "cancelled", final
    assert final["error_code"] == "cancelled"
    # A cancelled job must never publish outputs.
    assert final["outputs"] == []


def test_cancel_terminal_job_returns_409(client, db_session):
    done = make_generation(db_session, status="completed")
    resp = client.post(f"/api/v1/generations/{done.id}/cancel")
    assert resp.status_code == 409
    assert "terminal" in resp.json()["detail"].lower()

    cancelled = make_generation(db_session, status="cancelled")
    resp = client.post(f"/api/v1/generations/{cancelled.id}/cancel")
    assert resp.status_code == 409

    resp = client.post("/api/v1/generations/00000000-0000-0000-0000-00000000000f/cancel")
    assert resp.status_code == 404


def test_delete_active_job_cancels_first(client, db_session, valid_png_bytes, monkeypatch):
    sketch_id = upload_sketch(client, valid_png_bytes)
    entered_generate = threading.Event()
    release = threading.Event()

    def blocked_generate(*args, **kwargs):
        entered_generate.set()
        release.wait(timeout=30)
        return [
            {
                "image": Image.new("RGB", (64, 64), (99, 99, 99)),
                "metrics": {"engine": "fake", "is_mock": True},
            }
        ]

    monkeypatch.setattr(
        inference_service, "generate", blocked_generate
    )

    gen_id = create_generation(client, sketch_id)
    assert entered_generate.wait(timeout=20)

    # DELETE must cancel-first and then remove the row without crashing the
    # still-running worker.
    del_resp = client.delete(f"/api/v1/generations/{gen_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["success"] is True

    release.set()
    time.sleep(0.5)  # give the worker a moment to observe the deletion
    assert client.get(f"/api/v1/generations/{gen_id}").status_code == 404


# --------------------------------------------------------------------------- #
# Retry cap
# --------------------------------------------------------------------------- #
def test_retry_cap_enforced(client, db_session, monkeypatch):
    # Avoid spawning real background work; we only exercise the cap logic.
    monkeypatch.setattr(job_service, "submit_generation_job", lambda gid: None)

    # Job that already used all 3 retries -> immediate 409.
    exhausted = make_generation(
        db_session, status="failed", parameters_json={"retry_count": 3}
    )
    resp = client.post(f"/api/v1/generations/{exhausted.id}/retry")
    assert resp.status_code == 409, resp.text
    assert "maximum" in resp.json()["detail"].lower()
    assert "3" in resp.json()["detail"]

    # Job one retry below the cap -> retry succeeds, count persists, next -> 409.
    near_cap = make_generation(
        db_session, status="failed", parameters_json={"retry_count": 2}
    )
    resp = client.post(f"/api/v1/generations/{near_cap.id}/retry")
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "queued"

    db_session.expire_all()
    refreshed = (
        db_session.query(Generation).filter(Generation.id == near_cap.id).first()
    )
    assert refreshed.parameters_json.get("retry_count") == 3

    resp = client.post(f"/api/v1/generations/{near_cap.id}/retry")
    assert resp.status_code == 409

    # A fresh failed job starts counting from 0.
    fresh = make_generation(db_session, status="failed")
    resp = client.post(f"/api/v1/generations/{fresh.id}/retry")
    assert resp.status_code == 200


def test_retry_missing_job_returns_404(client, db_session):
    resp = client.post("/api/v1/generations/00000000-0000-0000-0000-00000000000e/retry")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Concurrent jobs (registry + cancel + completion interleaving)
# --------------------------------------------------------------------------- #
def test_concurrent_jobs_cancel_one_complete_other(client, db_session, valid_png_bytes, monkeypatch):
    sketch_id = upload_sketch(client, valid_png_bytes)

    entered = threading.Event()
    release = threading.Event()
    entered_count = {"n": 0}
    lock = threading.Lock()

    def blocked_generate(*args, **kwargs):
        with lock:
            entered_count["n"] += 1
            if entered_count["n"] >= 2:
                entered.set()
        release.wait(timeout=30)
        return [
            {
                "image": Image.new("RGB", (48, 48), (200, 100, 50)),
                "metrics": {"engine": "fake", "is_mock": True},
            }
        ]

    monkeypatch.setattr(
        inference_service, "generate", blocked_generate
    )

    job_a = create_generation(client, sketch_id)
    job_b = create_generation(client, sketch_id)

    # Both jobs run concurrently inside the (blocked) generation stage.
    assert entered.wait(timeout=20), "both workers never entered the generation stage"

    cancel_resp = client.post(f"/api/v1/generations/{job_a}/cancel")
    assert cancel_resp.status_code == 200
    release.set()

    final_a = poll_terminal(client, job_a)
    final_b = poll_terminal(client, job_b)
    assert final_a["status"] == "cancelled", final_a
    assert final_b["status"] == "completed", final_b
    assert len(final_b["outputs"]) == 1


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #
def test_rate_limiter_sliding_window_unit():
    from backend.app.core.rate_limit import RateLimiter

    limiter = RateLimiter(max_requests=3, window_seconds=60.0)
    results = [limiter.check("key-a")[0] for _ in range(3)]
    assert results == [True, True, True]

    allowed, retry_after = limiter.check("key-a")
    assert not allowed
    assert 0 < retry_after <= 60.0

    # Per-key isolation: a different bucket is unaffected.
    allowed, _ = limiter.check("key-b")
    assert allowed

    # Disabled limiter always allows.
    limiter.enabled = False
    allowed, _ = limiter.check("key-a")
    assert allowed


def test_rate_limiter_window_expiry():
    from backend.app.core.rate_limit import RateLimiter

    limiter = RateLimiter(max_requests=2, window_seconds=10.0)
    clock = {"now": 1000.0}
    limiter._clock = lambda: clock["now"]

    assert limiter.check("ip")[0] is True
    assert limiter.check("ip")[0] is True
    allowed, retry_after = limiter.check("ip")
    assert not allowed
    assert 0 < retry_after <= 10.0

    # Slide the window forward past all hits -> allowed again.
    clock["now"] += 11.0
    allowed, _ = limiter.check("ip")
    assert allowed


def test_rate_limit_generations_create_http_429(client, db_session, monkeypatch):
    from backend.app.core import rate_limit

    limiter = rate_limit.generation_create_limiter
    monkeypatch.setattr(limiter, "enabled", True)
    monkeypatch.setattr(limiter, "max_requests", 1)
    limiter.reset()

    payload = {
        "input_asset_id": "00000000-0000-0000-0000-000000000000",
        "package_type": "bottle",
    }

    first = client.post("/api/v1/generations", json=payload)
    # First request passes the limiter (asset lookup then 404s).
    assert first.status_code in (404, 422)

    second = client.post("/api/v1/generations", json=payload)
    assert second.status_code == 429
    assert "retry-after" in {k.lower() for k in second.headers}
    assert int(second.headers["Retry-After"]) >= 1
    assert "rate limit" in second.json()["detail"].lower()

    third = client.post("/api/v1/generations", json=payload)
    assert third.status_code == 429
