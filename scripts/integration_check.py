"""Live integration check: boots nothing itself; expects backend already running.

Run:  .venv/Scripts/python.exe scripts/integration_check.py http://127.0.0.1:8010
Checks: health, taxonomy, upload, generation lifecycle w/ engine metadata,
cancel endpoint, rate limiting, history list, feedback, delete.
"""
import base64
import io
import json
import sys
import time
from pathlib import Path

import httpx
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8010"
ok = []


def check(name, cond, extra=""):
    ok.append((name, bool(cond)))
    print(f"{'PASS' if cond else 'FAIL'}  {name} {extra}")


def main():
    c = httpx.Client(base_url=BASE, timeout=60)

    r = c.get("/api/v1/health")
    check("health", r.status_code == 200, r.text[:60])

    r = c.get("/api/v1/taxonomy")
    tx = r.json()
    check("taxonomy", r.status_code == 200 and len(json.dumps(tx)) > 200)

    # upload sketch
    buf = io.BytesIO()
    Image.open(ROOT / "data/samples/carton_sketch.png").convert("RGB").save(buf, "PNG")
    buf.seek(0)
    r = c.post("/api/v1/uploads", files={"file": ("carton.png", buf, "image/png")})
    check("upload", r.status_code in (200, 201), str(r.status_code))
    upj = r.json()
    asset_id = (upj.get("asset", {}).get("id") or upj.get("id") or upj.get("asset_id") or "")

    # create generation (canonical taxonomy keys; JSON body per GenerationCreate)
    r = c.post("/api/v1/generations", json={
        "input_asset_id": asset_id,
        "package_type": "folding_carton",
        "material": "kraft_cardboard",
        "finish": "gold_foil",
        "viewpoint": "three_quarter_view",
        "background": "seamless_studio",
        "prompt": "premium organic tea box, warm studio light",
        "negative_prompt": "no watermark, no extra flaps",
    })
    check("create generation accepted", r.status_code in (200, 201, 202), f"{r.status_code} {r.text[:120]}")
    gid = (r.json().get("id") or r.json().get("job_id") or "")

    # poll to completion (local engine ~15s worst case)
    final = None
    for _ in range(40):
        time.sleep(1.5)
        g = c.get(f"/api/v1/generations/{gid}").json()
        st = g.get("status")
        if st in ("completed", "failed", "cancelled"):
            final = g
            break
    check("generation completed", final is not None and final.get("status") == "completed",
          (final or {}).get("status", "timeout"))
    outs = (final or {}).get("outputs") or []
    md = (outs[0].get("metrics") if outs else None) or (final or {}).get("metadata") or {}
    eng = md.get("engine_used") or (final or {}).get("engine_used")
    check("engine metadata present", bool(eng), str(eng))
    print("   engine:", eng, "| latency:", md.get("engine_latency_s"))

    # outputs downloadable (files endpoint takes asset_id, not the row id)
    if outs:
        first = outs[0]
        oid = first.get("asset_id") or first.get("id")
        r2 = c.get(f"/api/v1/files/{oid}")
        check("output downloadable", r2.status_code == 200 and r2.headers.get("content-type", "").startswith("image"),
              str(r2.status_code))
        r3 = c.get(f"/api/v1/files/{oid}", params={"thumb": "1"})
        check("thumbnail", r3.status_code == 200)

    # history list contains our job
    r = c.get("/api/v1/generations", params={"limit": 10})
    ids = [g.get("id") for g in r.json().get("items", r.json() if isinstance(r.json(), list) else [])] if r.status_code == 200 else []
    check("history list", r.status_code == 200 and gid in ids, f"{len(ids)} jobs")

    # feedback
    r = c.post(f"/api/v1/generations/{gid}/feedback", json={"rating": 4, "comment": "integration"})
    check("feedback", r.status_code in (200, 201), str(r.status_code))

    # cancel endpoint on a fresh job (immediate cancel may hit before processing)
    buf.seek(0)
    a2 = c.post("/api/v1/uploads", files={"file": ("c2.png", buf, "image/png")}).json()
    a2id = a2.get("asset", {}).get("id") or a2.get("id") or a2.get("asset_id")
    r = c.post("/api/v1/generations", json={"input_asset_id": a2id, "package_type": "folding_carton",
                                            "material": "kraft_cardboard", "finish": "matte"})
    gid2 = (r.json().get("id") or r.json().get("job_id"))
    rc = c.post(f"/api/v1/generations/{gid2}/cancel")
    check("cancel accepted", rc.status_code == 200, str(rc.status_code))

    # rate limit: hammer reads (120/min) — should NOT 429 within budget; then 12 rapid creates -> expect a 429
    codes = [c.get("/api/v1/taxonomy").status_code for _ in range(5)]
    check("reads not limited at 5", all(x == 200 for x in codes))
    seen_429 = False
    last_gid = gid2
    for i in range(12):
        buf.seek(0)
        a = c.post("/api/v1/uploads", files={"file": (f"r{i}.png", buf, "image/png")})
        if a.status_code == 429:
            seen_429 = True
            print("   upload rate limited at", i)
            break
        aid = a.json().get("asset", {}).get("id") or a.json().get("id") or a.json().get("asset_id")
        rr = c.post("/api/v1/generations", json={"input_asset_id": aid, "package_type": "folding_carton",
                                                 "material": "kraft_cardboard", "finish": "matte"})
        if rr.status_code == 429:
            seen_429 = True
            print("   create rate limited at", i)
            break
        last_gid = rr.json().get("id") or last_gid
    check("rate limiting active (429 after burst)", seen_429)
    c.delete(f"/api/v1/generations/{gid}")
    if last_gid and last_gid != gid:
        c.delete(f"/api/v1/generations/{last_gid}")

    fails = [n for n, v in ok if not v]
    print(f"RESULT: {len(ok) - len(fails)}/{len(ok)} passed" + (f"; FAILURES: {fails}" if fails else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
