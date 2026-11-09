# PackVision AI: System Architecture & Technical Specification

## 1. Executive Architecture Summary

PackVision AI transforms 2D packaging sketches (hand-drawn or digital line-art) into photorealistic 3D mockups using a conditional generative pipeline. To meet the stringent requirements of packaging designers and brand teams, the system utilizes a **Two-Stage Hybrid Architecture**:

1. **Stage 1: Conditional Structural Diffusion & Material Synthesis**
   - Structural conditioning preserves the exact folds, dielines, seams, and perspective geometry of the original sketch.
   - Textual and semantic material conditioning generates realistic PBR substrates (kraft cardboard, corrugated fiber, frosted glass, brushed aluminum, molded pulp) and surface finishes (foil stamping, spot UV gloss, embossing/debossing).
2. **Stage 2: Homography-Warped Vector Compositing**
   - Brand logos and high-frequency typography are projected via 4-point homography and blended into the rendered surface using CIELAB luminance blending with specular shading preservation.

---

## 2. High-Level System Architecture Diagram

```
+-----------------------------------------------------------------------------------+
|                               USER INTERFACE LAYER                                |
|  - Web Studio Workbench (Dark luxury UI, canvas sketching, comparison slider)    |
|  - Standalone Gradio Workbench (ML experimentation & rapid parameter tuning)      |
|  - REST API Client (/api/v1)                                                      |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                               FASTAPI BACKEND TIER                                |
|  - /api/v1/uploads: Magic byte & dimension validation, EXIF correction, storage   |
|  - /api/v1/generations: Asynchronous job creation, status polling, cancellation   |
|  - /api/v1/taxonomy: Packaging categories, substrates, finishes, lighting         |
|  - /api/v1/files: Secure asset retrieval with thumbnail caching                   |
|  - /api/v1/feedback: Multi-criteria rating & quality evaluation logging           |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                            ASYNC WORKER & JOB QUEUE                               |
|  - State Engine: queued -> preprocessing -> generating -> postprocessing -> done  |
|  - Progress Tracking (0 - 100%) & Stage Messaging                                 |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                                 CORE AI/ML ENGINE                                 |
|                                                                                   |
|  [1. Preprocessing]                                                               |
|    - Aspect ratio normalization & letterboxing (1024x1024)                        |
|    - Canny / Adaptive Edge extraction with morphological dilation                  |
|    - Pseudo-depth map estimation                                                  |
|                                                                                   |
|  [2. Prompt Enrichment Engine]                                                    |
|    - Maps design parameters into physical optical prompts                         |
|    - Adds packaging geometry protection negative prompts                          |
|                                                                                   |
|  [3. Image-to-Image Generation Engine]                                            |
|    - Production: Stable Diffusion XL + Multi-ControlNet + Packaging LoRA          |
|    - CPU / Fallback: Procedural PBR Packaging Renderer with raytraced lighting     |
|                                                                                   |
|  [4. Stage-2 Postprocessing & Logo Compositing]                                   |
|    - 4-point homography perspective warping                                       |
|    - CIELAB luminance-matched blending (Multiply / Hard Light / Overlay)          |
|    - High-frequency edge preservation & sharpening                                |
|                                                                                   |
|  [5. Quantitative Evaluation Engine]                                              |
|    - Edge SSIM & IoU (structural fidelity against sketch)                         |
|    - CLIP Score (material & finish prompt alignment)                              |
|    - LPIPS & FID benchmark interfaces                                             |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                             STORAGE & PERSISTENCE                                 |
|  - Database: SQLite / PostgreSQL with SQLAlchemy ORM (Assets, Jobs, Feedback)     |
|  - Local File Storage: storage/inputs, storage/outputs, storage/thumbnails        |
+-----------------------------------------------------------------------------------+
```

---

## 3. Mathematical Foundations

### 3.1 Structural Conditioning Objective
Given a source sketch $c_s$, material text prompt $c_t$, and optional image reference $c_{img}$, the latent diffusion model is trained using the ControlNet zero-convolution formulation:

$$\mathcal{L}_{PackVision} = \mathbb{E}_{z_0, t, c_t, c_s, c_{img}, \epsilon \sim \mathcal{N}(0, I)} \left[ \left\| \epsilon - \epsilon_\theta(z_t, t, c_t, c_s, c_{img}) \right\|_2^2 \right]$$

### 3.2 Two-Stage Luminance-Preserving Logo Compositing
To prevent brand artwork from appearing pasted on top of the 3D render like a flat sticker, the warped logo $I_{warped}$ is combined with the underlying surface luminance:

$$I_{final}(x, y) = I_{warped\_label}(x, y) \odot \left( \frac{L_{render}(x, y)}{\mu_L} \right)^\gamma$$

where:
- $L_{render}(x,y)$ is the CIELAB lightness channel of the base render
- $\mu_L$ is the local mean surface luminance
- $\gamma$ is the shadow transfer coefficient ($\approx 1.2$)

This allows paper fibers, corrugation ridges, glass highlights, and foil sheen to physically modulate the printed ink.

---

## 4. Packaging Material & Print Finish Taxonomy

| Substrate Key | Material Description | Optical Simulation Properties |
| :--- | :--- | :--- |
| `kraft_cardboard` | Unbleached natural brown kraft paperboard | Visible wood fibers, high roughness, diffuse warm reflection |
| `corrugated_box` | Sturdy E-flute corrugated cardboard | Horizontal fluted edge texture, directional acoustic shadows |
| `matte_carton` | Smooth matte-laminated folding carton | Soft-touch velvet finish, wide specular lobe |
| `glossy_pouch` | Flexible stand-up mylar barrier pouch | High-gloss laminated film, sharp directional reflections |
| `frosted_glass` | Heavy luxury cosmetic bottle | Subsurface scattering, translucent blurred transmission |
| `brushed_aluminum` | Anodized cylindrical container | Anisotropic metallic sheen, linear microscopic grain |

| Finish Key | Print Treatment | Blending & Shader Behavior |
| :--- | :--- | :--- |
| `gold_foil` | Metallic gold hot-foil stamping | Specular metallic reflection, micro-embossed edge bevel |
| `spot_uv` | Clear gloss Spot UV varnish | Localized high-specularity coating contrasting matte substrate |
| `embossed` | 3D raised relief typography | High-light top edge, cast shadow bottom edge |
| `debossed` | Recessed indentation | Cast shadow top edge, reflected ambient light bottom edge |
| `holographic` | Iridescent rainbow film | Thin-film interference color shift varying across surface normal |
