# Changelog

All notable changes to PackVision AI are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] - 2026-11-11

### Added
- **Hybrid cloud engine** (`src/cloud_engine.py`): auto-detects Gemini /
  OpenRouter / OpenAI image APIs, selected via `RENDER_ENGINE`
  (`cloud` | `local` | `auto`) and `CLOUD_IMAGE_PROVIDER`; falls back
  gracefully to the local neural checkpoint or procedural PBR renderer.
  Per-provider image model overrides: `OPENROUTER_IMAGE_MODEL`,
  `OPENAI_IMAGE_MODEL`, `GEMINI_IMAGE_MODEL`.
- **Shared packaging taxonomy** (`src/taxonomy.py`): single source of truth
  for categories, materials, finishes, lighting, viewpoints and backgrounds,
  consumed by the prompt engine, dataset generator, and backend `/taxonomy`.
- **VLM-based evaluation provenance**: evaluation metrics can be enriched with
  Gemini VLM judgments (`GEMINI_VLM_MODEL`); provenance of each metric is
  recorded alongside the value.
- **Job cancellation & crash recovery** (backend): cooperative cancellation
  at stage boundaries via `POST /api/v1/generations/{id}/cancel`, recovery of
  jobs orphaned by a server restart, retry cap enforcement on
  `POST /generations/{id}/retry`.
- **Rate limiting** (backend): dependency-free per-IP sliding-window limiter —
  10 req/min for uploads and generation creates, 120 req/min for reads;
  configurable via `RATE_LIMIT_*` env vars; `429` responses carry
  `Retry-After`.
- **UI history persistence & engine badges**: render history survives page
  reloads and displays which engine produced each result.
- `data/repair_manifests.py`: manifest repair tool that reconciles the
  on-disk train/val triplet directories with `metadata.jsonl` and the split
  manifests (dry-run by default, `--write` to apply).
- CLI flags for `data/dataset_generator.py`: `--samples`, `--val-samples`,
  `--categories`, `--seed`, `--output-root` (previous defaults preserved).

### Changed
- **Dataset manifests repaired** (`data/repair_manifests.py --write`):
  train_manifest.jsonl 175 → 197 records, val_manifest.jsonl 35 → 56 records,
  metadata.jsonl 210 → 253 records. The 43 cross-split duplicate triplets
  (stale copies from an earlier generator run, 22 in train/ and 21 in val/)
  are now included under disambiguated `{sample_id}_stale` ids with
  `stale_duplicate_of` provenance and `seed: null`; no sample_id appears in
  both splits.
- `scripts/test_render_engine.py` and `scripts/test_facet_seg.py` now import
  `FacetAware3DSegmenter` / `FacetAware3DShader` /
  `ProceduralPackagingRenderer` from `src.pipeline` instead of duplicating
  them inline, so they exercise production code; stage timings and facet
  classification counts are printed.
- `scripts/train_custom_model.py`: a missing split manifest is now a hard
  error with remediation instructions instead of silently training on
  `metadata.jsonl` for both splits (data-leakage fix).
- `train_custom_model.bat`: no longer regenerates the dataset on every run
  (the old `--samples 80 --output-dir ...` call passed flags the generator
  never accepted); it now trains on the existing manifests and only offers
  regeneration when they are missing.
- `Dockerfile.backend`: the frontend workbench is copied into the image so
  `/` no longer 404s in the container; the container now runs as a
  non-root `appuser`.
- `docker-compose.yml`: passes through `RENDER_ENGINE` and cloud provider
  API keys / model overrides so the hybrid engine works under compose.
- `.gitignore`: root-level test artifacts (`/temp_render_*.png`,
  `/test_3d_*.png`, `/or_models_temp.json`) and `*.log` are ignored.
- Docs: `CUSTOM_MODEL_TRAINING_REPORT.md` metrics corrected to match
  `models/checkpoints/training_history.json` (1 epoch, val SSIM 0.5796,
  PSNR 8.62 dB, edge IoU 0.2177, 17,354,444 parameters) with a new
  Limitations section; `API_REFERENCE.md` rewritten against the actual
  routers/schemas (list + pagination, retry, cancel, delete, file serving,
  429 behavior); `README.md` gains the hybrid cloud engine, job control
  features, and a Configuration reference with a cloud-cost note.

### Removed
- `configs/inference_config.yaml` and `configs/train_lora_sdxl.yaml` (dead
  configuration files never loaded by any code; verified by repo-wide grep).
