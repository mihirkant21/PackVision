# PackVision AI — Sketch-to-Photorealistic Packaging Renderer

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.3+-61DAFB.svg)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/Vite-6.0+-646CFF.svg)](https://vitejs.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/Diffusers-SD%201.5%20%2B%20ControlNet-yellow.svg)](https://huggingface.co/docs/diffusers)

**PackVision AI** is a professional, local-only packaging design studio that transforms rough 2D packaging sketches, vector dielines, and concept drawings into studio-grade commercial product photographs using Generative AI, Stable Diffusion 1.5, and ControlNet Canny edge guidance.

---

## 1. Project Overview & Capabilities

### Supported Packaging Categories
- **Product Boxes & Folding Cartons**: Rigid cardboard, folding tuck boxes, gift cartons, mailer boxes.
- **Bottles & Jars**: Optical clear glass, amber tinted apothecary bottles, dropper vials, cosmetic cream jars.
- **Pouches & Packets**: Stand-up coffee pouches, flexible kraft barrier packets, resealable mylar sachets.
- **Cans & Aerosols**: Brushed aluminum soda cans, matte coated beverage tins.
- **Dispenser Tubes & Sleeves**: Cosmetic squeeze tubes, outer cardboard display sleeves.

### Preservation vs Synthesis
| Original Sketch Preserved | AI Enhanced & Synthesized |
| :--- | :--- |
| Package silhouette & geometric proportions | Physical substrate textures (coated SBS, kraft, glass SSS, aluminum) |
| Structural dielines, fold lines, and seams | Professional studio lighting rigs (dual softbox, rim light, golden hour) |
| Perspective camera angle & horizon placement | Authentic contact shadows & ambient ground occlusion |
| Brand logo placement & typography | Surface print embellishments (gold foil, spot UV varnish, embossing) |

---

## 2. Local-Only & Free-Only Architecture

PackVision AI runs **100% locally** on your computer. No cloud deployments (Render, Vercel, Railway), no paid SaaS APIs (OpenAI, Stability AI, Midjourney), and no cloud database dependencies are required.

```
+-----------------------------------------------------------------------------------+
|                        LOCAL FRONTEND (React 18 + Vite)                           |
|       Luxury Dark Studio UI • 3-Panel Workspace • Real-Time Before/After Slider   |
|                 URL: http://localhost:5173 (Port 5173)                            |
+------------------------------------------+----------------------------------------+
                                           |  REST API / CORS
                                           v
+-----------------------------------------------------------------------------------+
|                        LOCAL BACKEND (FastAPI + Uvicorn)                          |
|         Asynchronous Worker Queue • SQLite Persistence • Storage Management      |
|                 URL: http://127.0.0.1:8000 • Docs: /docs                          |
+-------------------+----------------------+--------------------+-------------------+
                    |                      |                    |
                    v                      v                    v
      +-------------------------+  +--------------+  +----------------------+
      |   INFERENCE STRATEGY    |  | LOCAL STORAGE|  |   LOCAL DATABASE     |
      | 1. Local GPU (SD 1.5)   |  | ./storage/   |  |   ./packvision.db    |
      | 2. Colab GPU Bridge     |  |  - inputs/   |  |   SQLite 3 engine    |
      | 3. Procedural Fallback  |  |  - outputs/  |  |   Projects, History  |
      +-------------------------+  +--------------+  +----------------------+
```

### Hardware & Inference Strategy
- **Laptop GPU (e.g. NVIDIA GTX 1650 Ti with 4GB VRAM)**:
  - Stable Diffusion 1.5 (`runwayml/stable-diffusion-v1-5`) + ControlNet Canny (`lllyasviel/control_v11p_sd15_canny`).
  - FP16 precision, attention slicing (`pipe.enable_attention_slicing()`), and sequential CPU offloading (`pipe.enable_model_cpu_offload()`).
  - Lightweight 512x512 resolution designed specifically for 4GB VRAM safety.
- **Optional Free Google Colab GPU Bridge**:
  - Connects to an external free Google Colab T4 GPU instance when the user configures `COLAB_INFERENCE_URL` in Settings.
  - Complete, self-contained worker notebook provided in `colab/PackVision_Colab_GPU.ipynb`.
  - All UI, database, uploads, and outputs remain strictly local.
- **Procedural Preview Fallback**:
  - Automatically activates when neither local GPU weights nor Colab bridge is configured.
  - Generates instant PBR packaging previews with ambient shadows and material tints.
  - Honest labeling: clearly marks outputs as procedural preview fallback.

---

## 3. Directory Structure

```text
PackVision-AI/
├── backend/                         # FastAPI Backend Application
│   ├── app/
│   │   ├── main.py                  # Entrypoint, CORS, static mounts, lifespan
│   │   ├── core/config.py           # Pydantic settings & hardware configuration
│   │   ├── db/                      # SQLAlchemy models & SQLite session
│   │   │   ├── models.py            # User, Project, Asset, Generation, Feedback
│   │   │   └── session.py           # SQLite connection & table auto-migration
│   │   ├── schemas/                 # Pydantic schemas (Project, Generation, Asset)
│   │   ├── services/
│   │   │   ├── asset_service.py     # File validation, thumbnailing, disk storage
│   │   │   ├── inference_service.py # Adapter bridging jobs with inference engines
│   │   │   └── job_service.py       # Asynchronous ThreadPool queue & cancellation
│   │   └── api/v1/
│   │       ├── router.py            # Aggregated API router
│   │       ├── health.py            # Hardware diagnostics & Colab health check
│   │       ├── projects.py          # Complete Projects CRUD
│   │       ├── preprocess.py        # Canny edge, line-art, & depth map previews
│   │       ├── generations.py       # Render jobs create, poll, list, retry, delete
│   │       ├── composite.py         # Deterministic logo positioning & ink blend
│   │       ├── evaluation.py        # Structural Edge SSIM & contour IoU metrics
│   │       ├── uploads.py           # Image upload & dimension validation
│   │       ├── files.py             # File and thumbnail serving
│   │       └── taxonomy.py          # Canonical packaging materials & finishes
│   ├── tests/                       # Backend test suite (49 tests)
│   └── requirements.txt             # Backend dependencies
├── frontend/                        # React 18 + Vite Web Studio
│   ├── src/
│   │   ├── main.jsx                 # React root
│   │   ├── App.jsx                  # Main view router & toast notifications
│   │   ├── context/AppContext.jsx   # Global state, projects, and active studio
│   │   ├── services/api.js          # Typed API client for all backend endpoints
│   │   ├── components/
│   │   │   ├── BeforeAfterSlider.jsx# Draggable before/after comparison slider
│   │   │   ├── LogoEditor.jsx       # Interactive logo position/scale/rotation/blend
│   │   │   └── EvaluationModal.jsx  # SSIM & structural consistency modal
│   │   ├── layouts/Sidebar.jsx      # Navigation sidebar with hardware status
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx        # Stats, quick-start templates, recent renders
│   │   │   ├── Studio.jsx           # 3-Panel design studio workspace
│   │   │   ├── Projects.jsx         # Project management grid & creation modal
│   │   │   ├── History.jsx          # Generations log gallery with filters
│   │   │   ├── Settings.jsx         # Colab GPU URL setup & hardware diagnostics
│   │   │   └── About.jsx            # Model architecture & license documentation
│   │   └── styles/index.css         # Tailwind directives & luxury dark theme
│   ├── package.json                 # React, Vite, Lucide icons, Tailwind
│   └── vite.config.js               # Port 5173 server & API proxy
├── src/                             # Core AI & Image Processing Engine
│   ├── controlnet_pipeline.py       # SD 1.5 + ControlNet Canny + Colab remote bridge
│   ├── preprocessor.py              # OpenCV normalization, Canny edges, depth map
│   ├── prompt_engine.py             # Packaging prompt synthesis from taxonomy
│   ├── postprocessor.py             # 4-point homography & CIELAB luminance blend
│   ├── evaluation.py                # Edge SSIM, contour IoU, structural metrics
│   └── taxonomy.py                  # Canonical materials, finishes, and forms
├── colab/                           # Optional Google Colab GPU Bridge
│   ├── PackVision_Colab_GPU.ipynb   # 1-Click notebook for free T4 GPU inference
│   └── packvision_colab_worker.py   # Standalone Colab FastAPI worker server
├── storage/                         # Local storage (Gitignored)
│   ├── inputs/                      # Uploaded sketches and moodboards
│   ├── outputs/                     # Generated photorealistic renders
│   ├── composites/                  # Logo-composited outputs
│   └── thumbnails/                  # 256x256 UI thumbnails
├── start_backend.bat                # 1-Click launcher for backend (Port 8000)
├── start_frontend.bat               # 1-Click launcher for frontend (Port 5173)
├── start_studio.bat                 # Launches both backend and frontend together
├── run_tests.bat                    # Runs complete pytest suite (271 tests)
└── README.md                        # Documentation
```

---

## 4. Local Installation & Startup Guide

### Prerequisites
- Windows 10/11 (or Linux/macOS)
- Python 3.10 to 3.13
- Node.js v18+ and npm

### Quick 1-Click Launch (Windows)
Double-click **`start_studio.bat`** in the project root directory. It automatically spins up the FastAPI backend on `http://127.0.0.1:8000` and launches the Vite React frontend at `http://localhost:5173`.

Alternatively, run each service independently:
- **`start_backend.bat`**: Starts FastAPI backend with automatic virtual environment & `PYTHONPATH` resolution.
- **`start_frontend.bat`**: Starts Vite development server at `http://localhost:5173`.

---

### Manual Terminal Setup (Cross-Platform)

#### Terminal 1: Backend Setup
```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate
# (On Linux/macOS: source venv/bin/activate)

# Install Python dependencies
pip install -r requirements.txt

# Start FastAPI backend server
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
The backend will be live at:
- Backend URL: `http://127.0.0.1:8000`
- Interactive Swagger API Documentation: `http://127.0.0.1:8000/docs`

### Terminal 2: Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install

# Start Vite React development server
npm run dev
```
Open your browser and navigate to:
- **Frontend Studio UI**: `http://localhost:5173`

---

## 5. Google Colab GPU Setup (Optional Free Acceleration)

If you prefer to run heavy diffusion passes on a 16GB T4 GPU instead of your laptop GPU:
1. Open Google Colab: [colab.research.google.com](https://colab.research.google.com).
2. Set Runtime accelerator: **Runtime → Change runtime type → T4 GPU**.
3. Upload and open `colab/PackVision_Colab_GPU.ipynb`.
4. Run the notebook cells to start the Colab worker and generate a public URL (via ngrok or localtunnel).
5. Open your local studio at `http://localhost:5173`, go to **Settings → Google Colab Remote GPU Bridge**, paste the URL, and click **Test Connection**.
6. PackVision AI will now route diffusion generation to the Colab GPU while saving all images and project data locally!

---

## 6. Evaluation & Verification

To run the automated validation test suite across all modules:
```bash
# Run all tests
python -m pytest backend/tests tests -v
```
**Test Results**: **271 Passed (100% Pass Rate)**
- `backend/tests/`: 49 passed (`test_api.py`, `test_features.py`, `test_job_lifecycle.py`, `test_preprocessor_unit.py`)
- `tests/`: 222 passed (`test_cloud_engine.py`, `test_taxonomy_eval.py`, `test_config_classifier.py`, `test_photo_mode.py`)

---

## 7. Model Licenses & Attribution

- **Stable Diffusion 1.5**: [CreativeML Open RAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license)
- **ControlNet Canny**: [Apache 2.0](https://github.com/lllyasviel/ControlNet/blob/main/LICENSE)
- **OpenCV**: [Apache 2.0](https://opencv.org/license/)
- **FastAPI & Uvicorn**: [MIT License](https://github.com/tiangolo/fastapi/blob/master/LICENSE)
- **React & Vite**: [MIT License](https://github.com/facebook/react/blob/main/LICENSE)
