"""Probe cloud providers + VLM connectivity using src.cloud_engine / src.vlm_client.

Prints the probe() availability table for all three image providers and runs one
tiny VLM text call to confirm VLM connectivity. No image generation by default;
pass --render to additionally render one mockup through the available provider.

Usage:
    .venv/Scripts/python.exe scripts/probe_cloud_providers.py
    .venv/Scripts/python.exe scripts/probe_cloud_providers.py --render [--provider gemini|openrouter|openai]
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cloud_engine import CloudEngineError, CloudRenderEngine  # noqa: E402
from src.vlm_client import ask_vlm, load_project_env, vlm_available  # noqa: E402

SKETCH = ROOT / "data" / "samples" / "carton_sketch.png"

RENDER_PROMPT = (
    "Photorealistic commercial studio photograph of this exact folding carton box, "
    "preserving silhouette and panel layout precisely. Material: kraft cardboard, "
    "matte finish, three-point softbox studio lighting, seamless neutral background."
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe cloud render providers and VLM connectivity.")
    parser.add_argument("--render", action="store_true", help="also perform one real cloud render")
    parser.add_argument("--provider", default="auto", choices=["auto", "gemini", "openrouter", "openai"])
    args = parser.parse_args()

    load_project_env()
    engine = CloudRenderEngine(provider=args.provider)

    print("== Cloud image provider availability ==")
    print(f"{'provider':<12}{'status':<12}{'configured':<12}last_error")
    for name, info in engine.probe().items():
        print(f"{name:<12}{info['status']:<12}{str(info['configured']):<12}{info['last_error'] or '-'}")

    print()
    print("== VLM connectivity (one tiny text call, no image generation) ==")
    if not vlm_available():
        print("VLM: unavailable - no provider API key configured (GEMINI/OPENROUTER/OPENAI)")
    else:
        t0 = time.time()
        reply = ask_vlm("Reply with exactly the single word: PACKVISION")
        elapsed = time.time() - t0
        preview = (reply or "<empty>").strip()[:120]
        print(f"VLM reply in {elapsed:.1f}s: {preview!r}")
        print("VLM: OK" if reply else "VLM: FAILED (empty reply)")

    if args.render:
        print()
        print("== Cloud render test ==")
        if not SKETCH.exists():
            print(f"missing sample sketch: {SKETCH}")
            return 1
        from PIL import Image  # noqa: E402 - deferred, only needed for --render

        try:
            result = engine.render(
                Image.open(SKETCH).convert("RGB"),
                RENDER_PROMPT,
                negative_prompt="blurry, low quality, warped text",
                size=1024,
            )
        except CloudEngineError as exc:
            print(f"RENDER FAILED: {exc}")
            return 1
        outdir = ROOT / "storage" / "outputs"
        outdir.mkdir(parents=True, exist_ok=True)
        out = outdir / f"probe_cloud_{result.provider}.png"
        result.image.save(out, format="PNG")
        print(
            f"provider={result.provider} model={result.model} latency={result.latency_s:.1f}s "
            f"cost={result.cost_note}"
        )
        print(f"SAVED {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
