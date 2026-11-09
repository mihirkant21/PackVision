# PackVision AI: Quantitative & Qualitative Evaluation Protocol

## 1. Objectives

The goal of the PackVision AI evaluation protocol is to evaluate both:
1. **Realism:** Does the output look like a commercial studio product photograph?
2. **Structural Preservation:** Does the output preserve the designer's original silhouette, panels, folds, and perspective without deformation?
3. **Semantic Alignment:** Does the generated render accurately exhibit the requested substrate (e.g., kraft fibers, frosted glass) and print finish (e.g., gold hot foil, spot UV)?

---

## 2. Quantitative Metric Benchmarks

| Metric | Target Benchmark | Mathematical Formulation | Interpretation in PackVision AI |
| :--- | :--- | :--- | :--- |
| **Edge SSIM / IoU** | $\ge 0.82$ | $\text{SSIM}(E_{sketch}, E_{render}) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$ | Evaluates edge overlap between input sketch outlines and Canny contours of the generated packaging. |
| **CLIP Alignment Score** | $\ge 0.31$ | $\text{sim}(I_{render}, T_{prompt}) = \frac{\mathbf{v}_{img} \cdot \mathbf{v}_{text}}{\|\mathbf{v}_{img}\| \|\mathbf{v}_{text}\|}$ | Measures semantic agreement between output render embeddings and the requested packaging prompt. |
| **LPIPS** | $\le 0.28$ | $d(x, x_0) = \sum_l \frac{1}{H_l W_l} \sum_{h, w} \| w_l \odot (\hat{y}^l_{hw} - \hat{y}_{0,hw}^l) \|_2^2$ | Deep perceptual feature distance computed across AlexNet / VGG layers for paired test sets. |
| **FID (Fréchet Inception Distance)** | $\le 25.0$ | $\|\mu_r - \mu_g\|_2^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$ | Measures population-level distribution distance between generated packaging renders and real commercial product photography. |

---

## 3. Human Qualitative Evaluation Rubric (1 - 5 Scale)

Reviewers (packaging designers and brand managers) rate each generated mockup across 5 core dimensions:

### 3.1 Shape & Geometry Preservation
- **1 (Unusable):** Major silhouette deformation; extra flaps added; perspective warped.
- **3 (Acceptable):** General package shape preserved, but slight curvature drift or softening of corner creases.
- **5 (Exemplary):** Crisp 1:1 structural fidelity matching the designer's input line-art.

### 3.2 Substrate & Material Realism
- **1 (Unusable):** Generic plastic look; kraft paper lacks texture or appears like flat paint.
- **3 (Acceptable):** Plausible color and rough surface, but lacking micro-fibers or natural light absorption.
- **5 (Exemplary):** Authentic tactile texture: visible kraft pulp fibers, corrugation fluting, or frosted glass subsurface scattering.

### 3.3 Print Finish Accuracy
- **1 (Unusable):** Foil appears as flat yellow paint; spot UV or embossing is completely absent.
- **3 (Acceptable):** Finish is identifiable with basic shine, but lacks micro-bevels or specular highlights.
- **5 (Exemplary):** Convincing anisotropic metallic reflections for foil; crisp depth shadows for embossing.

### 3.4 Lighting & Studio Environment
- **1 (Unusable):** Unnatural ambient lighting; conflicting cast shadows; blown-out highlights.
- **3 (Acceptable):** Standard studio illumination; plausible contact shadows.
- **5 (Exemplary):** Three-point softbox studio lighting with realistic rim highlights and subtle ambient occlusion.

### 3.5 Brand Logo & Typography Fidelity (Stage 2)
- **1 (Unusable):** Warped or AI-hallucinated characters; jagged pixelation.
- **3 (Acceptable):** Flat logo pasted on surface without surface texture interaction.
- **5 (Exemplary):** Razor-sharp vector typography with physically plausible luminance blending and perspective warping.
