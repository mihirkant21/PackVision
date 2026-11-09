# PackVision AI: Sketch-to-Photoreal Packaging Renderer

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/Diffusers-SDXL%20%2B%20ControlNet-yellow.svg)](https://huggingface.co/docs/diffusers)

PackVision AI is a visual prototyping system that bridges the gap between early packaging ideation sketches and client-ready 3D mockups. Using conditional image-to-image translation combined with structural edge conditioning and substrate material synthesis, PackVision AI transforms 2D packaging concepts (folding cartons, flexible pouches, cosmetic bottles, cans) into studio-grade photorealistic renders in seconds.

---

## Key Features

- **Structural Geometric Preservation:** Locks package dielines, folds, seams, and perspective geometry using ControlNet line-art and Canny edge extraction.
- **Physical Packaging Substrate Simulation:** Accurately renders kraft paperboard, E-flute corrugated cardboard, coated SBS paper, flexible mylar, frosted glass, and brushed aluminum.
- **Specialized Print Finish Rendering:** Simulates optical properties of gold/silver hot-foil stamping, spot UV gloss varnish, 3D embossing, debossing, and iridescent holographic films.
- **Two-Stage Brand Logo Compositor:** Prevents garbled AI text by perspective-warping flat vector artwork via 4-point homography and blending it with underlying surface luminance and micro-shadows.
- **Asynchronous Job Worker & REST API:** Complete FastAPI backend with SQLite/PostgreSQL persistence, job polling (`queued` ➔ `preprocessing` ➔ `generating` ➔ `postprocessing` ➔ `completed`), and thumbnail caching.
- **Interactive Dual Workbench:**
  - **Web Studio UI:** Dark-mode luxury CPG interface with real-time before/after comparison slider, canvas brush, sample sketch loader, and multi-criteria feedback dialog.
  - **Gradio Studio:** Rapid ML experimentation workbench with parameter sliders and metric readouts.
- **Built-in Dual-Engine Execution:** Runs full SDXL + ControlNet on CUDA GPUs, and includes an intelligent procedural PBR packaging renderer fallback for instant execution on standard CPUs.

---

## Architecture Overview

```
+-------------------------------------------------------------------------+
|                           USER INPUT LAYER                              |
|  1. Raw Packaging Sketch (Hand-drawn / iPad Procreate / Illustrator)    |
|  2. Material & Finish Spec (e.g., "Matte black carton, gold foil")      |
|  3. Optional Brand Logo / Vector Label Swatch                           |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                       PRE-PROCESSING PIPELINE                           |
|  - Background Inversion & 1024x1024 Aspect Normalization                |
|  - Structural Lineart & Adaptive Canny Edge Extraction                  |
|  - Pseudo-Depth Map Estimation from 2D Contours                         |
|  - Prompt Enrichment Engine (Maps specs to PBR optical prompts)         |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                    CORE IMAGE-TO-IMAGE AI ENGINE                        |
|  - Production: Stable Diffusion XL + Multi-ControlNet + Packaging LoRA  |
|  - CPU Fallback: Procedural PBR Substrate & Raytraced Lighting Engine   |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                  STAGE 2: LOGO COMPOSITING & POST-PROCESSING            |
|  - 4-Point Homography Perspective Warping                               |
|  - CIELAB Luminance & Ambient Shadow Blending                           |
|  - High-Frequency Detail Sharpening & Edge SSIM Metric Validation       |
+-------------------------------------------------------------------------+
```

---

## Directory Structure

```text
PackVision-AI/
├── ai/ / src/                       # Core AI/ML Engine
│   ├── preprocessor.py              # Sketch normalization & edge extraction
│   ├── prompt_engine.py             # Packaging taxonomy & prompt synthesis
│   ├── pipeline.py                  # SDXL + ControlNet + Procedural renderer
│   ├── postprocessor.py             # Stage-2 Homography & CIELAB Logo compositor
│   └── evaluation.py                # Edge SSIM, CLIP Score, FID, LPIPS metrics
├── backend/app/                     # FastAPI Backend Server
│   ├── main.py                      # Server entry point & CORS
│   ├── core/config.py               # Pydantic Settings & environment variables
│   ├── db/                          # SQLAlchemy database session & models
│   ├── schemas/                     # Pydantic request/response schemas
│   ├── services/                    # AssetService, JobService, InferenceService
│   └── api/v1/                      # Endpoints (/uploads, /generations, /files, etc.)
├── frontend/                        # Web Studio Workbench (HTML5/Tailwind/JS)
├── ui/                              # Standalone Gradio ML Studio
│   └── app_gradio.py
├── data/                            # Datasets & Samples
│   ├── samples/                     # Sample packaging sketches & logos
│   └── build_dataset.py             # Synthetic paired dataset generator
├── configs/                         # Model training and inference configs
├── docs/                            # Architecture & API documentation
├── storage/                         # Local asset storage (inputs, outputs, thumbs)
├── Dockerfile.backend               # Container definition
├── docker-compose.yml               # Multi-container orchestration
└── requirements.txt                 # Python dependencies
```

---

## Getting Started

### 1. Installation
Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/packvision-ai.git
cd packvision-ai

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment
Copy the example environment configuration:
```bash
cp .env.example .env
```

To run on GPU with full Stable Diffusion XL weights:
```ini
USE_MOCK_INFERENCE=false
MODEL_DEVICE=cuda
```

To run instantly on CPU with procedural PBR rendering:
```ini
USE_MOCK_INFERENCE=true
MODEL_DEVICE=cpu
```

### 3. Launching the Backend API
Start the FastAPI server:
```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```
- API Docs: `http://localhost:8000/docs`
- Web Studio UI: `http://localhost:8000/`

### 4. Launching the Gradio ML Workbench
Start the Gradio interactive studio:
```bash
python ui/app_gradio.py
```
- Gradio Studio: `http://localhost:7860`

---

## API Quick Tour

### Upload a Packaging Sketch
```bash
curl -X POST "http://localhost:8000/api/v1/uploads?asset_type=sketch" \
  -F "file=@data/samples/carton_sketch.png"
```

### Queue a Photoreal Render Job
```bash
curl -X POST "http://localhost:8000/api/v1/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "input_asset_id": "<asset-id-from-upload>",
    "package_type": "folding_carton",
    "material": "kraft_cardboard",
    "finish": "gold_foil",
    "viewpoint": "three_quarter",
    "prompt": "Luxury herbal tea packaging with botanical accents"
  }'
```

### Poll Job Progress & Retrieve Render
```bash
curl "http://localhost:8000/api/v1/generations/<job-id>"
```

---

## Evaluation Benchmark Targets

| Metric | Target | Purpose |
| :--- | :--- | :--- |
| **Edge SSIM / IoU** | $\ge 0.82$ | Structural line preservation vs original sketch |
| **CLIP Score** | $\ge 0.31$ | Semantic alignment to requested materials & finishes |
| **LPIPS** | $\le 0.28$ | Perceptual realism distance |
| **FID** | $\le 25.0$ | Closeness to studio product photography distribution |

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
