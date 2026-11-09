# PackVision AI: Product Specification & Requirements Document

## 1. Project Overview & Domain Context
- **Title:** PackVision AI: Sketch-to-Photoreal Packaging Renderer
- **Technique:** Conditional Image-to-Image (I2I) Translation / Diffusion Structural Conditioning
- **Domain:** Consumer Packaged Goods (CPG), Brand Identity, Industrial Packaging Design
- **Core Mission:** Accelerate packaging ideation by translating rough 2D hand or digital sketches directly into photorealistic, studio-lit 3D mockups with precise substrate textures and print finishes.

---

## 2. Problem Statement & Value Proposition
Traditional packaging prototyping requires:
1. CAD/3D modeling (Blender, Cinema 4D, KeyShot)
2. Manual UV mapping of dielines
3. Complex PBR material and shader configuration
4. Studio lighting setup and ray-traced rendering

This iteration cycle typically consumes between 4 to 24 hours per concept. PackVision AI cuts this turnaround to seconds, allowing creative directors and brand managers to test substrates, foil stamping, embossing, and lighting variations interactively in early design meetings.

---

## 3. Detailed Functional Requirements

### FR-1: Image Ingestion & Input Validation
- Accepts PNG, JPG/JPEG, and WEBP formats up to 25MB.
- Validates file signatures (magic bytes) to prevent malicious upload tampering.
- Bounds check: ensures resolutions fall between 128x128 and 4096x4096px.
- EXIF tag orientation auto-correction.
- Retains pristine original upload and generates a 256x256 thumbnail.

### FR-2: Structural Preprocessing
- Aspect ratio normalization to 1024x1024 via smart letterboxing.
- Inversion handling (swaps between dark-on-light sketch lines and light-on-dark edge conditioning).
- Adaptive Canny thresholding with morphological dilation ($2\times 2$ kernel) to heal broken pencil strokes.
- Pseudo-depth map estimation from structural contours.

### FR-3: Packaging Configuration Taxonomy
- **Categories:** Folding Carton, Rigid Setup Box, Stand-up Pouch, Cosmetic Bottle, Dropper Jar, Beverage Can, Sleeve.
- **Substrates:** Unbleached Kraft Paperboard, E-Flute Corrugated Cardboard, Matte Coated Folding Carton, SBS Bleached Board, Glossy Mylar Barrier Film, Frosted Luxury Glass, Brushed Aluminum, Molded Fiber Pulp.
- **Print Finishes:** Metallic Gold Hot-Foil Stamping, Silver Foil, Spot UV Gloss Varnish, Deep 3D Embossed Relief, Recessed Debossed Indentation, Iridescent Holographic Film, Soft-Touch Matte Lamination.
- **Lighting:** Three-Point Studio Softbox, Dramatic Luxury Rim Light, Commercial High-Key Clean, Natural Window Daylight.
- **Camera Views:** Front-Facing Hero, Three-Quarter Isometric Perspective, Low-Angle Dynamic, Top-Down Flat Lay.
- **Backgrounds:** Seamless White/Neutral Studio, Warm Beige Pedestal, Minimalist Dark Luxury Stone, Transparent/Isolated.

### FR-4: Asynchronous Inference Execution
- Immediate `202 Accepted` job return with unique UUID.
- Discrete state engine: `queued` ➔ `preprocessing` ➔ `generating` ➔ `postprocessing` ➔ `completed` (or `failed`).
- Real-time progress percentage (0 - 100%) and human-readable stage descriptions.
- Job cancellation and single-click retry support.

### FR-5: Two-Stage Hybrid Logo Compositing
- Solves generative AI's tendency to warp or corrupt small typography and logos.
- Stage 1 generates the photoreal blank container with full lighting, reflections, and substrate textures.
- Stage 2 warps the flat vector artwork onto the container using 4-point homography and modulates it with the CIELAB luminance of the underlying surface so texture and shadows show through.

### FR-6: Designer Review Workbench & Comparison
- Interactive Split Comparison Slider (reveals sketch vs. render).
- Edge SSIM structural fidelity score readout.
- Full-resolution PNG download with generation metadata embedded.
- Multi-criteria feedback dialog (1-5 stars; shape, material, finish, lighting).

---

## 4. Hardware Profiles & Deployment Modes

### GPU Production Mode
- Target GPU: NVIDIA A100 (40GB), RTX 4090 (24GB), or L4 (24GB).
- Backbone: Stable Diffusion XL (`stabilityai/stable-diffusion-xl-base-1.0`).
- Structural Control: SDXL ControlNet Lineart/Canny (`diffusers/controlnet-canny-sdxl-1.0`).
- VAE: FP16 fixed VAE (`madebyollin/sdxl-vae-fp16-fix`).
- Scheduler: UniPCMultistepScheduler (20-25 steps).

### CPU / Edge Fallback Mode (`USE_MOCK_INFERENCE=true`)
- Immediate response on standard CPUs without GPU or multi-gigabyte weight downloads.
- Procedural PBR engine synthesizes physical substrate textures (kraft pulp grain, corrugation fluting, glass subsurface scattering, brushed metallic anisotropic highlights) and studio lighting shaders directly conditioned on input sketch contours.
