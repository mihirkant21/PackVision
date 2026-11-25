# PackVision AI — Complete Testing Kit & Prompt Recipes

Welcome to the **PackVision AI Test Kit**! This folder contains everything you need to test the end-to-end packaging synthesis pipeline—from rough structural sketches to tuned taxonomy prompts and transparent brand logos.

---

## Folder Contents

```text
sample_test_kit/
├── sketches/                                # 5 Ready-to-use packaging sketches (1024x1024)
│   ├── 01_folding_carton_box.png           # 3D folding tuck carton with perspective panels
│   ├── 02_cosmetic_dropper_bottle.png      # Cosmetic dropper bottle with pipette & bulb
│   ├── 03_standup_coffee_pouch.png         # Stand-up flexible pouch (doypack) with heat seal
│   ├── 04_aluminum_beverage_can.png        # 330ml beverage can with pull tab & rim bevel
│   └── 05_luxury_perfume_flask.png         # Heavy crystal glass perfume flacon with stopper
├── logos/                                   # 3 High-resolution transparent PNG brand logos
│   ├── logo_aurora_botanicals_gold.png     # Gold/charcoal luxury botanical emblem & seal
│   ├── logo_minimalist_noir.png            # Bold Swiss minimalist monogram & typography
│   └── logo_vibrant_beverage.png           # Electric blue & orange circular craft soda badge
└── TEST_RECIPES_AND_PROMPTS.md             # This guide with tested copy-paste prompt recipes
```

---

## 5 Tested Prompt Recipes for Maximum Quality

### Recipe 1: Luxury Organic Tea Folding Carton Box
- **Sketch File**: `sample_test_kit/sketches/01_folding_carton_box.png`
- **UI Settings**:
  - **Category**: `Folding Carton / Box`
  - **Substrate Material**: `Unbleached Kraft Paper` (or `Coated SBS Paperboard`)
  - **Finish**: `Gold Foil Stamping`
  - **Viewpoint**: `Three-Quarter View`
  - **Lighting**: `Commercial Studio Softbox`
- **Copy-Paste Prompt**:
  ```text
  Photorealistic commercial packaging product photograph of a luxury organic herbal tea folding carton box, authentic natural fibrous kraft paperboard texture, elegant hot gold foil stamped accents along the edges and front label, sharp creased fold seams, illuminated with soft diffused studio softbox lighting, clean studio background, authentic contact drop shadow, 8k resolution CPG mockup
  ```
- **Diffusion Settings**: Steps: `30` | CFG: `7.5` | Edge Strength: `0.85` | Resolution: `512px`
- **Logo Compositor (Tab 3)**:
  - Select `sample_test_kit/logos/logo_aurora_botanicals_gold.png`
  - Position: `X: 49%` | `Y: 56%` | `Scale: 28%` | `Rotation: 0°` | `Ink Blend: Multiply`

---

### Recipe 2: Emerald Glass Cosmetic Dropper Bottle
- **Sketch File**: `sample_test_kit/sketches/02_cosmetic_dropper_bottle.png`
- **UI Settings**:
  - **Category**: `Bottle / Jar`
  - **Substrate Material**: `Optical Clear Glass`
  - **Finish**: `Spot UV Gloss Varnish`
  - **Viewpoint**: `Front Eye-Level`
  - **Lighting**: `Dramatic Rim Light`
- **Copy-Paste Prompt**:
  ```text
  High-end commercial cosmetics photograph of an optical clear emerald tinted glass dropper bottle filled with golden botanical face serum, authentic internal glass pipette tube, black matte rubber suction bulb collar with gold metallic rim, sharp optical caustics, soft reflection on glossy surface, dark luxury studio backdrop, 8k resolution product hero shot
  ```
- **Diffusion Settings**: Steps: `30` | CFG: `7.5` | Edge Strength: `0.85` | Resolution: `512px`
- **Logo Compositor (Tab 3)**:
  - Select `sample_test_kit/logos/logo_minimalist_noir.png`
  - Position: `X: 50%` | `Y: 60%` | `Scale: 24%` | `Ink Blend: Normal`

---

### Recipe 3: Specialty Coffee Stand-Up Flexible Pouch
- **Sketch File**: `sample_test_kit/sketches/03_standup_coffee_pouch.png`
- **UI Settings**:
  - **Category**: `Flexible Pouch`
  - **Substrate Material**: `Matte Laminated Board`
  - **Finish**: `Matte Lamination`
  - **Viewpoint**: `Front Eye-Level`
  - **Lighting**: `Natural Daylight`
- **Copy-Paste Prompt**:
  ```text
  Commercial product photograph of a modern matte black stand-up flexible coffee bean pouch, authentic sealed side crimp ribs, top heat seal band with tear notch, organic kraft paper texture, subtle satin surface sheen, standing upright on a minimalist warm wood pedestal table, studio softbox illumination, crisp product photograph
  ```
- **Diffusion Settings**: Steps: `28` | CFG: `7.0` | Edge Strength: `0.82` | Resolution: `512px`
- **Logo Compositor (Tab 3)**:
  - Select `sample_test_kit/logos/logo_aurora_botanicals_gold.png`
  - Position: `X: 50%` | `Y: 52%` | `Scale: 32%` | `Ink Blend: Multiply`

---

### Recipe 4: Electric Craft Beverage Aluminum Soda Can
- **Sketch File**: `sample_test_kit/sketches/04_aluminum_beverage_can.png`
- **UI Settings**:
  - **Category**: `Can / Tin`
  - **Substrate Material**: `Brushed Aluminum`
  - **Finish**: `Spot UV Gloss Varnish`
  - **Viewpoint**: `Front Eye-Level`
  - **Lighting**: `High-Key Commercial`
- **Copy-Paste Prompt**:
  ```text
  Commercial beverage advertisement photograph of a sleek 330ml aluminum soda can, brushed metallic silver aluminum finish with condensation micro-droplets on the surface, crisp top rim and pull tab, studio rim lighting, vibrant clean reflections, bright commercial studio lighting, authentic contact shadow, ultra-sharp detail
  ```
- **Diffusion Settings**: Steps: `30` | CFG: `7.5` | Edge Strength: `0.85` | Resolution: `512px`
- **Logo Compositor (Tab 3)**:
  - Select `sample_test_kit/logos/logo_vibrant_beverage.png`
  - Position: `X: 50%` | `Y: 55%` | `Scale: 30%` | `Ink Blend: Multiply`

---

### Recipe 5: Ultra-Luxury Crystal Flacon Perfume Bottle
- **Sketch File**: `sample_test_kit/sketches/05_luxury_perfume_flask.png`
- **UI Settings**:
  - **Category**: `Bottle / Jar`
  - **Substrate Material**: `Optical Clear Glass`
  - **Finish**: `Gold Foil Stamping`
  - **Viewpoint**: `Front Eye-Level`
  - **Lighting**: `Commercial Studio Softbox`
- **Copy-Paste Prompt**:
  ```text
  Luxury perfume bottle photograph, heavy rectangular crystal glass flacon with thick base and beveled corners, filled with translucent luminous fragrance liquid, internal glass dip tube, polished gold atomizer collar, spherical crystal stopper cap, seated on a reflective glass surface with soft caustics, softbox illumination, 8k resolution
  ```
- **Diffusion Settings**: Steps: `30` | CFG: `8.0` | Edge Strength: `0.85` | Resolution: `512px`
- **Logo Compositor (Tab 3)**:
  - Select `sample_test_kit/logos/logo_minimalist_noir.png`
  - Position: `X: 50%` | `Y: 58%` | `Scale: 22%` | `Ink Blend: Multiply`

---

## Step-by-Step Testing Workflow in PackVision AI

1. **Launch Studio**:
   Double click `start_studio.bat` (or open `http://localhost:5173`).
2. **Upload a Sketch**:
   In the left panel under **Input Packaging Sketch**, click to upload any sketch from `sample_test_kit/sketches/` (e.g., `01_folding_carton_box.png`).
3. **Inspect Edge & Depth Maps**:
   Click the **Edge & Depth Maps** tab at the top of the canvas to verify the Canny edge contour and pseudo-depth estimation.
4. **Set Directives & Prompt**:
   - Choose the Category, Substrate, and Finish according to the recipe above.
   - Paste the recipe prompt into the **Positive Prompt Directive** accordion (or let the auto-synthesizer build it).
5. **Generate Packaging Render**:
   Click the purple **Generate Packaging Render** button on the right panel.
6. **Compare Before & After**:
   Once generated (~15 seconds on Colab GPU or instant on local shader), drag the center slider to inspect how the structural lines locked the 3D render.
7. **Apply Vector Brand Logo**:
   - Click the **Logo Compositor** tab.
   - Upload any logo from `sample_test_kit/logos/`.
   - Adjust the sliders for X/Y position, scale, and blend mode.
   - Click **Apply & Save Logo Composite** to produce the final client-ready mockup!
