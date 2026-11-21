"""Probe free HuggingFace ZeroGPU spaces for sketch-to-photoreal I2I.

Tries Flux Kontext spaces (no API key needed; ZeroGPU anonymous quota).
Usage: .venv/Scripts/python.exe scripts/probe_hf_spaces.py
"""
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKETCH = ROOT / "data/samples/carton_sketch.png"
OUTDIR = ROOT / "storage/outputs"

PROMPT = (
    "Convert this packaging sketch into a photorealistic studio product photograph. "
    "Keep the exact same box shape, silhouette, fold lines and panel layout. "
    "Material: natural kraft cardboard with visible paper grain. Gold foil stamped label border. "
    "Soft three-point studio lighting, seamless light grey background, sharp focus, commercial product photography."
)

SPACES = [
    "black-forest-labs/FLUX.1-Kontext-dev",
    "multimodalart/FLUX.1-Kontext-Dev",
    "Qwen/Qwen-Image-Edit",
]


def try_space(space: str) -> bool:
    from gradio_client import Client, handle_file

    print(f"--- {space} ---", flush=True)
    try:
        t0 = time.time()
        client = Client(space, verbose=False)
        print(f"connected in {time.time() - t0:.1f}s", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"connect failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
        return False

    # discover api endpoints
    try:
        apis = client.view_api(return_format="dict", print_info=False)
        named = apis.get("named_endpoints", {})
        print(f"endpoints: {list(named.keys())[:6]}", flush=True)
        ep, params = list(named.items())[0]
        pnames = [p.get("parameter_name") or p.get("label", "").lower().replace(" ", "_") for p in params.get("parameters", [])]
        print(f"using {ep} params={pnames}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"api introspect failed: {e}", flush=True)
        ep = None

    tmp = Path(shutil.copy(SKETCH, Path(OUTDIR) / f"probe_tmp_{abs(hash(space)) % 10000}.png"))
    try:
        t0 = time.time()
        if space.startswith("black-forest-labs/FLUX.1-Kontext"):
            result = client.predict(
                input_image=handle_file(str(tmp)),
                prompt=PROMPT,
                seed=0,
                randomize_seed=True,
                guidance_scale=2.5,
                steps=28,
                api_name="/infer",
            )
        else:
            # generic: pass first file param + prompt by best effort
            result = client.predict(handle_file(str(tmp)), PROMPT, api_name=ep or None)
        dt = time.time() - t0
        print(f"predict OK in {dt:.1f}s -> {result}", flush=True)
        outpath = None
        if isinstance(result, (list, tuple)):
            for item in result:
                if isinstance(item, str) and Path(item).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                    outpath = item
                    break
        elif isinstance(result, str):
            outpath = result
        if outpath and Path(outpath).exists():
            dest = OUTDIR / f"probe_hf_{space.replace('/', '_')}.png"
            shutil.copy(outpath, dest)
            print(f"SAVED {dest} bytes={dest.stat().st_size}", flush=True)
            return True
        print("no image path in result", flush=True)
        return False
    except Exception as e:  # noqa: BLE001
        print(f"predict failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
        return False
    finally:
        tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    ok = False
    for space in SPACES:
        ok = try_space(space) or ok
        if ok:
            break
    print("RESULT:", "SUCCESS" if ok else "ALL_FAILED")
    sys.exit(0 if ok else 1)
