# PackVision AI: Custom Dataset & Deep Learning Model Training Report

> **Revision note (2026-11-11):** An earlier version of this report quoted
> metrics (6 training epochs, validation SSIM 0.9460, 16,495,875 parameters)
> that do not match the artifacts in this repository. All figures below are
> now taken directly from
> [`models/checkpoints/training_history.json`](file:///c:/Users/Lenovo/Downloads/GenAI%20Project/models/checkpoints/training_history.json)
> and the checkpoint itself.

## 1. Executive Summary

To provide native domain adaptation and reduce external dependency constraints on CPU environments, we synthesized a paired packaging dataset and trained **PackVisionNet-v2**, a proprietary conditional packaging image-to-image neural network.

- **Model Architecture:** [`src/models/packvision_net.py`](file:///c:/Users/Lenovo/Downloads/GenAI%20Project/src/models/packvision_net.py) (`PackVisionNet`)
- **Parameters:** 17,354,444 trainable parameters (17.35M)
- **Trained Checkpoint:** [`models/checkpoints/packvision_custom_v1.pt`](file:///c:/Users/Lenovo/Downloads/GenAI%20Project/models/checkpoints/packvision_custom_v1.pt) (stores epoch, metrics, and model config metadata)
- **Training Trajectory Log:** [`models/checkpoints/training_history.json`](file:///c:/Users/Lenovo/Downloads/GenAI%20Project/models/checkpoints/training_history.json)
- **Training Run:** **1 epoch** (395.97 s total on CPU; learning rate 2e-4)
- **Validation Metrics (epoch 1, procedural ground truth):** SSIM **0.5796**, PSNR **8.62 dB**, Edge IoU **0.2177**, Val Loss **0.63219**
- **Working Resolution:** 256x256 (model config `image_size: 256`); the output is subsequently up-composited at 1024x1024 with the facet-aware shader stages in [`src/pipeline.py`](file:///c:/Users/Lenovo/Downloads/GenAI%20Project/src/pipeline.py)

These metrics are from a **single-epoch run and reflect a partially converged model**, not a production-quality result. See [Section 7 — Limitations](#7-limitations).

---

## 2. Custom Packaging Dataset Engineering

The dataset was generated using [`data/dataset_generator.py`](file:///c:/Users/Lenovo/Downloads/GenAI%20Project/data/dataset_generator.py) across 7 packaging categories and 7 material substrates.

### 2.1 Dataset Statistics

Per-generator defaults (30 samples per category, 5 of each reserved for validation): **210 samples (175 train / 35 val)** with distinct ids `pv_pkg_0001`–`pv_pkg_0210`.

Current on-disk state after the manifest repair run of `data/repair_manifests.py` (2026-11-11):

- **Triplets on disk:** 253 (197 in `data/dataset/train/`, 56 in `data/dataset/val/`), each `{sample_id}_sketch.png` / `{sample_id}_render.png` / `{sample_id}_depth.png`
- **`train_manifest.jsonl`:** 197 records — 175 authoritative train samples plus 22 stale duplicates (older-generation renders of val ids, included as `{sample_id}_stale` with a `stale_duplicate_of` provenance field)
- **`val_manifest.jsonl`:** 56 records — 35 authoritative val samples plus 21 stale duplicates
- **`metadata.jsonl`:** 253 records (the union of both manifests; every on-disk triplet accounted for, no sample_id in both splits)

Manifest repair is reproducible: `python data/repair_manifests.py` (dry run) / `--write`.

### 2.2 Data Modalities per Sample
1. **Designer Sketch:** Synthetic hand-drawn-style lineart with graphite noise (`_draw_geometric_sketch`).
2. **Ground Truth Render:** Procedurally rendered 512x512 packshot with per-facet lighting, substrate texture, and finish (`_render_ground_truth`).
3. **Pseudo-Depth Map:** Structural depth relief (`_generate_depth_map`).
4. **Structured Record:** Category / material / finish / lighting / background attributes plus generated positive & negative prompts in the manifest.

### 2.3 Coverage Matrix

Attributes rotate deterministically across categories (see `PackagingDatasetGenerator.build_dataset`):

| Attribute | Values (from the generator taxonomy) |
| :--- | :--- |
| **Categories (7)** | carton, box, pouch, bottle, jar, can, tube |
| **Materials (7)** | kraft_cardboard, corrugated_box, matte_carton, coated_sbs, flexible_pouch, frosted_glass, brushed_aluminum |
| **Finishes (7)** | gold_foil, silver_foil, spot_uv, embossed, debossed, holographic, matte |
| **Lighting (4)** | studio_softbox, dramatic_rim, high_key_commercial, natural_daylight |
| **Backgrounds (3)** | seamless_studio, warm_pedestal, dark_luxury |

---

## 3. Neural Architecture: PackVisionNet-v2

```
                    +-----------------------------+
                    |  Input Sketch Tensor (1ch)  |
                    +--------------+--------------+
                                   |
                                   v
+-----------------------+   +--------------+
| Categorical Selectors |-->| Conditioning | (Category, Material, Finish)
| (Embeddings: 3 x 64d) |   |     MLP      |  -> concatenated 192-dim vector
+-----------------------+   +-------+------+
                                    |
          +-------------------------+-------------------------+
          |                         | (FiLM Modulation)       |
          v                         v                         v
+-------------------+     +-------------------+     +-------------------+
| Encoder Stage 1-4 | --> | Spatial Attention | --> | Decoder Stage 1-4 |
| (ResNet + FiLM +  |     |    Bottleneck     |     | (Skip Concat +    |
|  Edge Attention)  |     |                   |     |  Attention Gates) |
+-------------------+     +-------------------+     +-------------------+
```

### 3.1 Architectural Highlights
1. **FiLM (Feature-wise Linear Modulation):** Substrate materials and print finishes modulate feature maps affinely:
   $$\mathbf{y} = \gamma(\mathbf{c}) \odot \mathbf{x} + \beta(\mathbf{c})$$
   where $\gamma, \beta$ are generated from the 192-dimensional conditioning vector $\mathbf{c}$ (three 64-dim embeddings for category, material, finish; `cond_dim = embedding_dim * 3`).
2. **Structural Edge Attention:** Edge maps attend over encoder/decoder features to preserve fold creases and silhouettes.
3. **Attention Gates & Multi-Scale Skips:** Preserve fine geometry from early encoder layers into the upsampling decoder.
4. **Model config (from checkpoint):** in 1ch / out 3ch, base channels 32, channel multipliers (1, 2, 4, 8), 16-way category/material/finish vocabularies, `image_size` 256, version "2.0".

---

## 4. Multi-Objective Training Loss

$$\mathcal{L}_{total} = \lambda_{l1} \mathcal{L}_{L1} + \lambda_{ssim} (1 - \text{SSIM}) + \lambda_{edge} \mathcal{L}_{edge} + \lambda_{color} \mathcal{L}_{color}$$

with $\lambda_{l1} = 1.0$, $\lambda_{ssim} = 0.5$, $\lambda_{edge} = 0.4$, $\lambda_{color} = 0.3$ (checkpoint `loss_weights`; `scripts/train_custom_model.py` defaults).

- **$\mathcal{L}_{L1}$ (Pixel Reconstruction):** Enforces base colors, lighting falloff, and shadow gradients.
- **$(1-\text{SSIM})$ (Structural Contrast):** Aligns local luminance/contrast structure with the ground-truth render.
- **$\mathcal{L}_{edge}$ (Sobel Structural Loss):** Compares spatial edge gradient magnitudes to keep folds and seams sharp.
- **$\mathcal{L}_{color}$ (Color Consistency):** Penalizes color shifts against the ground-truth render.

---

## 5. Training Trajectory

Single run, single epoch, CPU-only (batch size 4, gradient accumulation 2, AdamW with 2-epoch linear warmup configured — the run stopped after epoch 1):

| Epoch | Learning Rate | Train Loss | Train L1 | Train SSIM | Train Edge | Train Color | Val Loss | Val L1 | Val SSIM | Val PSNR (dB) | Val Edge IoU | Val Color | Epoch Time (s) | Checkpoint Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **01** | $2 \times 10^{-4}$ | 0.6716 | 0.32264 | 0.5798 | 0.1081 | 0.31876 | **0.63219** | 0.29429 | **0.5796** | **8.62** | **0.2177** | 0.28789 | 395.54 | Saved (Best) |

Total training time: 395.97 s. The checkpoint additionally stores `val_loss`, `val_ssim`, `val_psnr`, and `val_edge_iou` as top-level metadata, which `src/pipeline.py` surfaces as render metadata (`metrics_source: "checkpoint_training_history"`).

---

## 6. Verification & Deployment

The trained checkpoint [`models/checkpoints/packvision_custom_v1.pt`](file:///c:/Users/Lenovo/Downloads/GenAI%20Project/models/checkpoints/packvision_custom_v1.pt) is integrated into:

- [`src/pipeline.py`](file:///c:/Users/Lenovo/Downloads/GenAI%20Project/src/pipeline.py): `_load_custom_model()` lazily loads the checkpoint and `_render_with_custom_model()` executes inference, then composites the 256x256 neural output into the 1024x1024 facet-lit scene (contact shadow, 3D lighting, finish, tone mapping).
- [`ui/app_gradio.py`](file:///c:/Users/Lenovo/Downloads/GenAI%20Project/ui/app_gradio.py): selectable in the engine dropdown as **`PackVisionNet-v2 (local)`**.
- Test inference output from a previous session: [`storage/outputs/custom_model_carton_render.png`](file:///c:/Users/Lenovo/Downloads/GenAI%20Project/storage/outputs/custom_model_carton_render.png).

Inference latency was **not re-benchmarked** in this revision; no timing claim is made here.

---

## 7. Limitations

These limitations should be read alongside every metric in this report:

1. **Ground-truth circularity.** The training targets are produced by the procedural renderer in `data/dataset_generator.py` (`FacetPBRRenderer`), which is a close sibling of the production CPU fallback in `src/pipeline.py`. The model is therefore trained to imitate the procedural engine's own output style. Validation SSIM/PSNR measure agreement with these synthetic targets, **not** with real product photography.
2. **Dataset scale.** 210 distinct samples (253 on-disk triplets including 43 stale duplicate renders) across 7 categories is small; the model has seen each category roughly 30 times, and 43 of the training records are older-generation re-renders of val ids (`*_stale` records in `train_manifest.jsonl`).
3. **Single-epoch, CPU-only training.** The run completed 1 of the planned 10 epochs in ~6.6 minutes on CPU and was not resumed. Metrics (val SSIM 0.5796, PSNR 8.62 dB, edge IoU 0.2177) indicate a substantially under-converged model; no learning-rate or capacity search was performed.
4. **Attribute-only conditioning.** The model conditions on category/material/finish indices only — it does not consume the text prompts or brand/color fields present in the manifest.
5. **Resolution gap.** The network operates at 256x256; fine label typography and micro-textures in the 1024x1024 composite come from the procedural post-pass, not from the network.
