# GitHub & Cloud-API Research — PackVision AI (2026-11-11)

Findings from the research agent. Full report preserved here; key facts distilled below.

## Capability matrix (verified live by orchestrator)
| Capability | Gemini (free key) | OpenRouter (free key) | OpenAI (key) |
|---|---|---|---|
| Image gen I2I | 429 quota=0 for image models | 402 needs ≥$1.00 balance | 429 insufficient_quota |
| Text/VLM | OK: `gemini-3.8-flash` (free tier) | OK: 13 `:free` models (incl. `nvidia/nemotron-3.5-super-vl:free`) | No credits |
| Deprecated/notes | `gemini-2.5-flash` 404 for new users | `:free` image models do not exist | — |

## Best cloud primitives for sketch→photoreal (ranked)
1. **Stability AI `stable-image/control/sketch`** — purpose-built sketch→photoreal, `control_strength` param, $0.05/img. Bearer key, multipart REST. (Not yet implemented — add as 4th provider in cloud engine.)
2. fal.ai `fal-ai/flux-lora-canny` ($0.035/MP) — canny conditioning + LoRA slots for material control.
3. Replicate `black-forest-labs/flux-canny-dev` ($0.025–0.035/img), `flux-schnell` $0.003/img.
4. HF Inference Providers — no arbitrary ControlNet via router; use direct fal/Replicate instead.

## Best local-CPU neural path
- **pix2pix-turbo** (MIT, github.com/GaParmar/img2img-turbo) via ONNX→OpenVINO: ~1.7 s / 512×512 on desktop CPU (fastsdcpu numbers). Official Intel OpenVINO notebook exists for sketch-to-image. Replacement path for PackVisionNet-v2 realism.
- Full SDXL on CPU: 2–5 min/img — route to cloud instead.
- CPU training beyond ~10M params impractical (kohya #679: 6–11 days/GTX-1070-class).

## Top borrowable repos
- lllyasviel/ControlNet (34k) — lineart training recipe: mix auto-extracted + hand sketches; conditioning augmentation.
- TencentARC/T2I-Adapter (3.8k) — 77M-param sketch adapter shape (CPU/borderline trainable).
- DLR-RM/BlenderProc (GPL-3.0) — procedural dataset generation w/ HDRI/material/camera randomization (210 → 5–50k pairs).
- Mikubill/sd-webui-controlnet (17.8k) — preprocessor zoo (Informative Drawings/HED/PiDi/AnimeLineArt, ONNX) for multi-extractor sketch synthesis.
- tencent-ailab/IP-Adapter (6.7k) — reference-image identity lock (cloud) for label/logo preservation.
- VSAnimator/Sketch-a-Sketch — incremental/partial-sketch progressive re-render UX.
- JIA-Lab-research/ControlNeXt (Apache-2.0) — thin control branch, fewer params.
- rupeshs/fastsdcpu (MIT) — OpenVINO serving path for 1-step CPU preview.

## Dataset strategy
- Multi-preprocessor sketch extraction (3–4 per render, random pick) + jitter augmentation (thickness, inversion, shift/scale/rotate, masking) = ControlNet-lineart convention.
- Packaging data on HF is thin: P4rz1val/SyntheticBeverages (~1.6k procedural bottles), contralabs/creative-ad-design-dataset, projectsim/bottles-and-containers. No public paired sketch→packaging-mockup set — self-generation is the moat.
- VLM auto-captioning of renders for prompt-vocabulary alignment (works free via Gemini today).

## Ranked improvement actions (post-current-phase)
1. Add Stability `control/sketch` provider to cloud engine (~1 day, needs $ credits).
2. Grow dataset 210 → 3–10k via BlenderProc domain randomization (2–4 days, CPU-only).
3. Multi-preprocessor extraction + jitter augmentation on all pairs (~1 day).
4. OpenVINO pix2pix-turbo local preview tier (1–2 days).
5. Material prompt-fragment library + cloud material LoRA slots (2–3 days).
6. T2I-Adapter-style 77M adapter training on enlarged set (needs rented GPU $1–2/h).
7. IP-Adapter artwork identity lock on cloud path (2–3 days).
8. Upscale/super-resolution post-stage + edge-IoU regression harness (~1 day).

## License flags
- FLUX.1-dev/tools/Kontext = non-commercial; FLUX-schnell = Apache-2.0; SDXL = OpenRAIL++; SDXL-Lightning = Apache-2.0; ControlNet = Apache-2.0; T2I-Adapter: no OSS license file — verify before redistribution; BlenderProc = GPL-3.0 (keep it as an external dataset tool, don't link into repo code).
