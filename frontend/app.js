/**
 * PackVision AI - Luxury Packaging Studio Workbench
 * =================================================
 * Frontend client orchestrating sketch ingestion, interactive canvas touch-up,
 * packaging taxonomy configuration, logo overlay, async job polling,
 * interactive split comparison slider, and designer feedback.
 */

"use strict";

(function () {
  // Application State
  const state = {
    backendHealthy: false,
    currentSketchBlob: null,
    currentSketchUrl: null,
    currentLogoBlob: null,
    currentLogoUrl: null,
    activeJobId: null,
    pollInterval: null,
    currentGeneration: null,
    history: [],
    toolMode: "brush",
    brushSize: 3,
    isDrawing: false,
    lastX: 0,
    lastY: 0,
    sliderActive: false,
    viewMode: "split", // 'split', 'side', 'render'
    feedbackRating: 5,
    taxonomyReady: null, // promise resolving when /api/v1/taxonomy has been applied
    taxonomyLabels: {}, // category -> { id -> label }
    materialFinishMap: {}, // material id -> { default_finish, compatible_finishes }
    backendHistoryAvailable: true,
  };

  // DOM Elements Cache
  const elements = {
    backendStatusBadge: document.getElementById("backend-status-badge"),
    backendStatusText: document.getElementById("backend-status-text"),
    sketchFileInput: document.getElementById("sketch-file-input"),
    dropZone: document.getElementById("drop-zone"),
    uploadEmptyState: document.getElementById("upload-empty-state"),
    sketchCanvas: document.getElementById("sketch-canvas"),
    canvasOverlayBar: document.getElementById("canvas-overlay-bar"),
    canvasTools: document.getElementById("canvas-tools"),
    btnInvertLines: document.getElementById("btn-invert-lines"),
    btnClearCanvas: document.getElementById("btn-clear-canvas"),
    brushSizeInput: document.getElementById("brush-size"),
    brushSizeVal: document.getElementById("brush-size-val"),
    categorySelect: document.getElementById("category-select"),
    inputModeSelect: document.getElementById("input-mode-select"),
    btnRedetect: document.getElementById("btn-redetect"),
    detectStatus: document.getElementById("detect-status"),
    materialSelect: document.getElementById("material-select"),
    finishSelect: document.getElementById("finish-select"),
    lightingSelect: document.getElementById("lighting-select"),
    viewpointSelect: document.getElementById("viewpoint-select"),
    backgroundSelect: document.getElementById("background-select"),
    userPrompt: document.getElementById("user-prompt"),
    negativePrompt: document.getElementById("negative-prompt"),
    btnEnrichPrompt: document.getElementById("btn-enrich-prompt"),
    toggleLogoPanel: document.getElementById("toggle-logo-panel"),
    logoPanelContent: document.getElementById("logo-panel-content"),
    logoPanelIcon: document.getElementById("logo-panel-icon"),
    logoFileInput: document.getElementById("logo-file-input"),
    logoPreviewThumb: document.getElementById("logo-preview-thumb"),
    logoEmptyIcon: document.getElementById("logo-empty-icon"),
    logoFilename: document.getElementById("logo-filename"),
    btnLoadSampleLogo: document.getElementById("btn-load-sample-logo"),
    logoPosX: document.getElementById("logo-pos-x"),
    logoPosY: document.getElementById("logo-pos-y"),
    logoScale: document.getElementById("logo-scale"),
    logoXVal: document.getElementById("logo-x-val"),
    logoYVal: document.getElementById("logo-y-val"),
    logoScaleVal: document.getElementById("logo-scale-val"),
    logoBlendMode: document.getElementById("logo-blend-mode"),
    toggleAdvancedPanel: document.getElementById("toggle-advanced-panel"),
    advancedPanelContent: document.getElementById("advanced-panel-content"),
    advancedPanelIcon: document.getElementById("advanced-panel-icon"),
    inputSteps: document.getElementById("input-steps"),
    inputGuidance: document.getElementById("input-guidance"),
    inputControlnetScale: document.getElementById("input-controlnet-scale"),
    inputSeed: document.getElementById("input-seed"),
    btnGenerate: document.getElementById("btn-generate"),
    btnGenerateText: document.getElementById("btn-generate-text"),
    themeToggleBtn: document.getElementById("theme-toggle-btn"),
    themeIconMoon: document.getElementById("theme-icon-moon"),
    themeIconSun: document.getElementById("theme-icon-sun"),
    jobStatusPill: document.getElementById("job-status-pill"),
    renderEmptyState: document.getElementById("render-empty-state"),
    renderProgressCard: document.getElementById("render-progress-card"),
    progressStageTitle: document.getElementById("progress-stage-title"),
    progressPctBadge: document.getElementById("progress-pct-badge"),
    progressBarFill: document.getElementById("progress-bar-fill"),
    progressStageMsg: document.getElementById("progress-stage-msg"),
    stepQueued: document.getElementById("step-queued"),
    stepPre: document.getElementById("step-pre"),
    stepGen: document.getElementById("step-gen"),
    stepPost: document.getElementById("step-post"),
    sliderContainer: document.getElementById("slider-container"),
    sketchPreviewImg: document.getElementById("sketch-preview-img"),
    renderClipLayer: document.getElementById("render-clip-layer"),
    renderResultImg: document.getElementById("render-result-img"),
    sliderHandle: document.getElementById("slider-handle"),
    sideBySideContainer: document.getElementById("side-by-side-container"),
    sideSketchImg: document.getElementById("side-sketch-img"),
    sideRenderImg: document.getElementById("side-render-img"),
    renderActionsPanel: document.getElementById("render-actions-panel"),
    engineBanner: document.getElementById("engine-banner"),
    engineBadgeText: document.getElementById("engine-badge-text"),
    engineNoteText: document.getElementById("engine-note-text"),
    metricSsimBadge: document.getElementById("metric-ssim-badge"),
    metricClipBadge: document.getElementById("metric-clip-badge"),
    metricIouBadge: document.getElementById("metric-iou-badge"),
    metricLatencyBadge: document.getElementById("metric-latency-badge"),
    btnDownloadRender: document.getElementById("btn-download-render"),
    btnDownloadMetadata: document.getElementById("btn-download-metadata"),
    btnOpenFeedback: document.getElementById("btn-open-feedback"),
    feedbackModal: document.getElementById("feedback-modal"),
    btnCloseFeedback: document.getElementById("btn-close-feedback"),
    btnCancelFeedback: document.getElementById("btn-cancel-feedback"),
    btnSubmitFeedback: document.getElementById("btn-submit-feedback"),
    starBtns: document.querySelectorAll(".star-btn"),
    starRatingLabel: document.getElementById("star-rating-label"),
    feedbackComment: document.getElementById("feedback-comment"),
    historyDrawer: document.getElementById("history-drawer"),
    toggleHistoryBtn: document.getElementById("toggle-history-btn"),
    btnCloseDrawer: document.getElementById("btn-close-drawer"),
    historyItemsList: document.getElementById("history-items-list"),
    historyBadgeCount: document.getElementById("history-badge-count"),
    toastContainer: document.getElementById("toast-container"),
    viewModeSplit: document.getElementById("view-mode-split"),
    viewModeSide: document.getElementById("view-mode-side"),
    viewModeRender: document.getElementById("view-mode-render"),
  };

  const canvasCtx = elements.sketchCanvas.getContext("2d");

  // =========================================================================
  // TOAST NOTIFICATIONS
  // =========================================================================
  function showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = `toast-msg pointer-events-auto flex items-center gap-2.5 rounded-xl bg-white px-4 py-3 text-[13px] font-medium shadow-sm transition text-neutral-800 dark:bg-neutral-900 dark:text-neutral-200 border-l-2 ${
      type === "success"
        ? "border-l-green-600"
        : type === "error"
        ? "border-l-red-600"
        : "border-l-neutral-300 dark:border-l-neutral-500"
    }`;
    toast.innerHTML = `<span>${message}</span>`;
    elements.toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateY(10px)";
      setTimeout(() => toast.remove(), 250);
    }, 3500);
  }

  // =========================================================================
  // BACKEND HEALTH & TAXONOMY INITIALIZATION
  // =========================================================================
  async function checkBackendHealth() {
    try {
      const res = await fetch("/api/v1/health");
      if (res.ok) {
        const data = await res.json();
        state.backendHealthy = true;
        // Health payload nests device info under hardware{}; guard against
        // missing fields so the badge never crashes into the offline branch.
        const deviceLabel = String(
          data.device || data.hardware?.device_name || data.hardware?.configured_device || "CPU"
        ).toUpperCase();
        elements.backendStatusText.textContent = `Online • ${deviceLabel}`;
        elements.backendStatusBadge.className =
          "flex items-center gap-2 text-xs font-medium text-green-600 dark:text-green-400";
      } else {
        throw new Error("Bad status");
      }
    } catch {
      state.backendHealthy = false;
      elements.backendStatusText.textContent = "Backend offline";
      elements.backendStatusBadge.className =
        "flex items-center gap-2 text-xs font-medium text-red-600 dark:text-red-400";
    }
  }

  // =========================================================================
  // TAXONOMY (LIVE FROM GET /api/v1/taxonomy, HARDCODED HTML AS FALLBACK)
  // =========================================================================
  const TAXONOMY_SOURCES = [
    // [select cache key, accepted response field names in priority order]
    ["categorySelect", ["package_types", "packageTypes"]],
    ["materialSelect", ["materials"]],
    ["finishSelect", ["finishes"]],
    ["lightingSelect", ["lighting_presets", "lightingPresets", "lighting"]],
    ["viewpointSelect", ["viewpoints", "camera_angles", "cameraAngles"]],
    ["backgroundSelect", ["backgrounds"]],
  ];

  function setSelectValueIfPresent(selectEl, value) {
    if (!selectEl || value === null || value === undefined) return false;
    const exists = Array.from(selectEl.options).some((opt) => opt.value === value);
    if (exists) {
      selectEl.value = value;
      applyFinishCompatibility();
      return true;
    }
    return false;
  }

  function resolveCanonicalValue(selectEl) {
    // Prefer the canonical taxonomy key attached to the selected option so
    // generation requests always send canonical ids, even when the visible
    // option is a legacy alias kept for backwards compatibility.
    if (!selectEl || selectEl.selectedIndex < 0) return "";
    const opt = selectEl.options[selectEl.selectedIndex];
    return (opt && opt.dataset && opt.dataset.canonical) || selectEl.value;
  }

  function taxonomyLabelFor(selectEl, value) {
    const source = value || (selectEl ? selectEl.value : "");
    const labels = state.taxonomyLabels[selectEl ? selectEl.id : ""] || {};
    return labels[source] || source;
  }

  function populateSelectFromItems(selectEl, items) {
    if (!selectEl || !Array.isArray(items) || items.length === 0) return false;

    const previous = selectEl.value;
    selectEl.innerHTML = "";
    const labels = {};

    items.forEach((item) => {
      if (!item || !item.id) return;
      const option = document.createElement("option");
      option.value = item.id;
      option.textContent = item.label || item.id;
      if (item.description) option.title = item.description;
      const meta = item.metadata || {};
      // Legacy alias entries point at their canonical key so requests still
      // send canonical ids; canonical entries carry their own id.
      option.dataset.canonical = meta.canonical_key || (meta.legacy ? "" : item.id) || item.id;
      selectEl.appendChild(option);
      labels[item.id] = item.label || item.id;
    });

    state.taxonomyLabels[selectEl.id] = labels;

    if (previous && !Array.from(selectEl.options).some((opt) => opt.value === previous)) {
      // previous selection no longer exists; leave the first option selected
    } else if (previous) {
      selectEl.value = previous;
    }
    return true;
  }

  function applyFinishCompatibility() {
    const materialSelect = elements.materialSelect;
    if (!materialSelect || !elements.finishSelect) return;
    const canonicalId = resolveCanonicalValue(materialSelect);
    const rawId = materialSelect.value;
    const compat = state.materialFinishMap[canonicalId] || state.materialFinishMap[rawId];
    const finishSelect = elements.finishSelect;
    if (!finishSelect || !compat || !Array.isArray(compat.compatible_finishes) || compat.compatible_finishes.length === 0) {
      // No compatibility info: show every finish option
      Array.from(finishSelect.options).forEach((opt) => (opt.hidden = false));
      return;
    }
    const allowed = new Set(compat.compatible_finishes);
    let currentAllowed = false;
    Array.from(finishSelect.options).forEach((opt) => {
      const canonical = opt.dataset.canonical || opt.value;
      const isAllowed = allowed.has(opt.value) || allowed.has(canonical);
      opt.hidden = !isAllowed;
      if (opt.value === finishSelect.value && isAllowed) currentAllowed = true;
    });
    if (!currentAllowed) {
      const defaultKey = compat.default_finish;
      const fallback =
        Array.from(finishSelect.options).find((opt) => (opt.dataset.canonical || opt.value) === defaultKey) ||
        Array.from(finishSelect.options).find((opt) => !opt.hidden);
      if (fallback) finishSelect.value = fallback.value;
    }
  }

  async function fetchTaxonomy() {
    state.taxonomyReady = (async () => {
      try {
        const res = await fetch("/api/v1/taxonomy");
        if (!res.ok) throw new Error(`taxonomy HTTP ${res.status}`);
        const data = await res.json();

        let populated = 0;
        TAXONOMY_SOURCES.forEach(([selectKey, fieldNames]) => {
          for (const fieldName of fieldNames) {
            const items = data[fieldName];
            if (Array.isArray(items) && items.length > 0) {
              if (populateSelectFromItems(elements[selectKey], items)) populated += 1;
              break;
            }
          }
        });

        // Material compatibility metadata (drives finish filtering)
        const materialItems = data.materials;
        if (Array.isArray(materialItems)) {
          materialItems.forEach((m) => {
            if (m && m.id) {
              state.materialFinishMap[m.id] = {
                default_finish: m.default_finish,
                compatible_finishes: m.compatible_finishes,
              };
            }
          });
        }

        if (populated > 0) {
          applyFinishCompatibility();
          console.info(`[PackVision] Taxonomy loaded from API (${populated}/6 selectors).`);
        } else {
          console.warn("[PackVision] Taxonomy response had no usable lists; keeping hardcoded fallbacks.");
        }
      } catch (err) {
        console.warn("[PackVision] Taxonomy fetch failed, using hardcoded fallbacks:", err.message);
      }
    })();
    return state.taxonomyReady;
  }

  function waitForTaxonomyReady() {
    const timeout = new Promise((resolve) => setTimeout(resolve, 3000));
    return Promise.race([state.taxonomyReady || Promise.resolve(), timeout]).catch(() => {});
  }

  // =========================================================================
  // CANVAS DRAWING & IMAGE MANAGEMENT
  // =========================================================================
  function renderImageToCanvas(img) {
    const canvas = elements.sketchCanvas;
    canvas.width = 1024;
    canvas.height = 1024;

    canvasCtx.fillStyle = "#ffffff";
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

    // Letterbox draw
    const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
    const nw = img.width * scale;
    const nh = img.height * scale;
    const ox = (canvas.width - nw) / 2;
    const oy = (canvas.height - nh) / 2;

    canvasCtx.drawImage(img, ox, oy, nw, nh);

    elements.uploadEmptyState.classList.add("hidden");
    canvas.classList.remove("hidden");
    elements.canvasOverlayBar.classList.remove("hidden");

    canvas.toBlob((blob) => {
      state.currentSketchBlob = blob;
      state.currentSketchUrl = URL.createObjectURL(blob);
    }, "image/png");
  }

  function loadSketchFile(file) {
    if (!file || !file.type.startsWith("image/")) {
      showToast("Please select a valid image file (PNG, JPG, WEBP).", "error");
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        renderImageToCanvas(img);
        showToast(`Loaded sketch: ${file.name}`, "success");
        // Auto-detect the packaging configuration from the uploaded picture.
        autoDetectConfig(file);
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  // -------------------------------------------------------------------------
  // AUTO-DETECTION: inspect the uploaded picture and auto-fill the config
  // selectors (input type, form, material, finish, lighting, viewpoint,
  // backdrop) using the backend vision model. A quiet #detect-status line in
  // the left rail mirrors progress; the manual selectors always stay
  // authoritative and failures never block the workflow.
  // -------------------------------------------------------------------------
  let autoDetectSeq = 0;
  let detectFailTimer = null;
  const DETECT_FAIL_CLEAR_MS = 4500;

  // Status line states: --detecting (animated dots), --ok (summary), --fail
  // (brief note that auto-clears). Empty text collapses the line entirely.
  function setDetectStatus(kind, text) {
    const el = elements.detectStatus;
    if (!el) return;
    if (detectFailTimer) {
      clearTimeout(detectFailTimer);
      detectFailTimer = null;
    }
    el.classList.remove("detect-status--detecting", "detect-status--ok", "detect-status--fail");
    if (!kind || !text) {
      el.textContent = "";
      return;
    }
    el.classList.add(`detect-status--${kind}`);
    el.textContent = text;
    if (kind === "fail") {
      detectFailTimer = setTimeout(() => {
        el.textContent = "";
        el.classList.remove("detect-status--fail");
        detectFailTimer = null;
      }, DETECT_FAIL_CLEAR_MS);
    }
  }

  function setDetectBusy(busy) {
    // The Re-detect control is disabled for the whole detection round,
    // regardless of whether it was triggered by an upload or by the button.
    if (elements.btnRedetect) elements.btnRedetect.disabled = busy;
  }

  // Backend sends detected_summary as lowercase canonical keys joined by
  // " · " (e.g. "stand up pouch · flexible mylar · matte"); fall back to the
  // raw canonical fields, then to the input mode alone.
  function buildDetectedSummary(cfg) {
    const parts = [];
    const push = (value) => {
      const t = String(value || "").trim().replace(/_/g, " ").toLowerCase();
      if (t && !parts.includes(t)) parts.push(t);
    };
    String(cfg.detected_summary || "")
      .split("·")
      .forEach((part) => push(part));
    if (parts.length === 0) {
      push(cfg.package_type);
      push(cfg.material);
      push(cfg.finish);
    }
    if (parts.length === 0) push(cfg.input_mode);
    return parts.join(" · ");
  }

  async function autoDetectConfig(file) {
    const seq = ++autoDetectSeq;
    setDetectBusy(true);
    setDetectStatus("detecting", "Detecting configuration");
    try {
      // ---- Phase 1: instant local classifier (<200ms, no network) ----
      const fd = new FormData();
      fd.append("file", file, file.name || "upload.png");
      fd.append("enrich", "false");
      const res = await fetch("/api/v1/uploads/analyze", { method: "POST", body: fd });
      if (!res.ok) {
        if (seq === autoDetectSeq) setDetectStatus("fail", "Auto-detect unavailable — settings kept");
        return;
      }
      const cfg = await res.json();
      if (seq !== autoDetectSeq) return; // a newer upload superseded this one

      const setSelect = (el, value) => {
        if (!el || !value) return false;
        const wanted = String(value);
        const hit = [...el.options].some((o) => o.value === wanted);
        if (!hit) return false;
        el.value = wanted;
        el.dispatchEvent(new Event("change", { bubbles: true }));
        return true;
      };

      const fillFromCfg = (data) => {
        let count = 0;
        if (setSelect(elements.inputModeSelect, data.input_mode)) count++;
        if (setSelect(elements.categorySelect, data.package_type)) count++;
        if (setSelect(elements.materialSelect, data.material)) count++;
        if (setSelect(elements.finishSelect, data.finish)) count++;
        if (setSelect(elements.lightingSelect, data.lighting)) count++;
        if (setSelect(elements.viewpointSelect, data.viewpoint)) count++;
        if (setSelect(elements.backgroundSelect, data.background)) count++;
        return count;
      };

      // Phase 1 results: selectors auto-fill IMMEDIATELY.
      let filled = fillFromCfg(cfg);
      if (filled > 0) {
        const summary = buildDetectedSummary(cfg);
        setDetectStatus("ok", summary ? `Detected: ${summary}` : `Auto-filled ${filled} settings`);
        showToast(`Detected ${cfg.input_mode === "photo" ? "a product photo" : "a sketch"} — config auto-filled`, "success");
        // Prompt auto-fill: only when the user hasn't typed their own text
        // (empty field, sample-default prompt, or a previous auto-fill).
        const suggested = String(cfg.suggested_prompt || "");
        if (suggested && elements.userPrompt) {
          const current = (elements.userPrompt.value || "").trim();
          const isAuto = !current || current === state.lastAutoPrompt || /luxury herbal botanical tea carton/i.test(current);
          if (isAuto) {
            elements.userPrompt.value = suggested;
            elements.userPrompt.dispatchEvent(new Event("input", { bubbles: true }));
            state.lastAutoPrompt = suggested;
          }
        }
      } else {
        setDetectStatus("fail", "Auto-detect unavailable — settings kept");
        setDetectBusy(false);
        return;
      }

      // ---- Phase 2 (background): VLM enrichment adds lighting/viewpoint/
      // backdrop + upgrades sketch material/finish when free providers have
      // capacity. Never blocks: phase-1 results are already applied.
      const enrichFd = new FormData();
      enrichFd.append("file", file, file.name || "upload.png");
      enrichFd.append("enrich", "true");
      const enrichRes = await fetch("/api/v1/uploads/analyze", { method: "POST", body: enrichFd });
      if (seq !== autoDetectSeq) return;
      if (enrichRes.ok) {
        const enriched = await enrichRes.json();
        const upgraded = fillFromCfg(enriched);
        const summary = buildDetectedSummary(enriched);
        if (summary) setDetectStatus("ok", `Detected: ${summary}`);
        if (enriched.suggested_prompt) {
          state.lastAutoPrompt = enriched.suggested_prompt;
          if (!(elements.userPrompt.value || "").trim()) {
            elements.userPrompt.value = enriched.suggested_prompt;
            elements.userPrompt.dispatchEvent(new Event("input", { bubbles: true }));
          }
        }
        if (upgraded > 0 || enriched.lighting || enriched.viewpoint || enriched.background) {
          showToast("Configuration enriched with lighting & scene details", "info");
        }
      }
      // Enrichment failures stay silent: phase-1 config already applied.
    } catch (err) {
      // Detection is best-effort; never block the workflow.
      if (seq === autoDetectSeq) setDetectStatus("fail", "Auto-detect unavailable — settings kept");
    } finally {
      if (seq === autoDetectSeq) setDetectBusy(false);
    }
  }

  // Re-detect runs the same analysis against the CURRENT canvas content
  // (brush/eraser edits included): canvas -> PNG blob -> File-like object,
  // then the regular FormData upload path in autoDetectConfig.
  function redetectFromCanvas() {
    const canvas = elements.sketchCanvas;
    if (canvas.classList.contains("hidden") || !state.currentSketchBlob) {
      showToast("Load a sketch first — nothing to re-detect yet.", "info");
      return;
    }
    canvas.toBlob((blob) => {
      if (!blob) return;
      autoDetectConfig(new File([blob], "canvas.png", { type: "image/png" }));
    }, "image/png");
  }

  async function loadSampleSketch(sampleId) {
    // Canonical taxonomy keys (legacy aliases remain selectable via /api/v1/taxonomy)
    const sampleEndpoints = {
      carton: {
        file: "carton_sketch.png",
        category: "folding_carton",
        material: "kraft_cardboard",
        finish: "gold_foil",
        lighting: "studio_softbox",
        camera: "three_quarter_view",
        prompt: "Luxury herbal botanical tea carton, gold hot-foil debossed logo, minimalist clean dielines",
      },
      pouch: {
        file: "pouch_sketch.png",
        category: "stand_up_pouch",
        material: "flexible_mylar",
        finish: "matte",
        lighting: "studio_softbox",
        camera: "front_view",
        prompt: "Artisanal organic ground coffee stand-up pouch, earth tones, matte velvet finish, clean seal",
      },
      bottle: {
        file: "bottle_sketch.png",
        category: "bottle",
        material: "frosted_glass",
        finish: "spot_uv",
        lighting: "dramatic_rim",
        camera: "front_view",
        prompt: "Luxury cosmetic dropper bottle, frosted amber glass, gold metallic collar, studio lighting",
      },
    };

    const config = sampleEndpoints[sampleId];
    if (!config) return;

    try {
      const res = await fetch(`/api/v1/samples/${config.file}`);
      if (!res.ok) throw new Error("Sample file not found");
      const blob = await res.blob();
      const img = new Image();
      img.onload = async () => {
        renderImageToCanvas(img);
        await waitForTaxonomyReady();
        setSelectValueIfPresent(elements.categorySelect, config.category);
        setSelectValueIfPresent(elements.materialSelect, config.material);
        setSelectValueIfPresent(elements.finishSelect, config.finish);
        setSelectValueIfPresent(elements.lightingSelect, config.lighting);
        setSelectValueIfPresent(elements.viewpointSelect, config.camera);
        elements.userPrompt.value = config.prompt;
        showToast(`Sample loaded: ${(config.category || "sample").replace(/_/g, " ")}`, "success");
      };
      img.src = URL.createObjectURL(blob);
    } catch (err) {
      showToast(`Could not load sample: ${err.message}`, "error");
    }
  }

  function invertCanvasLines() {
    const canvas = elements.sketchCanvas;
    const imgData = canvasCtx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imgData.data;

    for (let i = 0; i < data.length; i += 4) {
      data[i] = 255 - data[i]; // R
      data[i + 1] = 255 - data[i + 1]; // G
      data[i + 2] = 255 - data[i + 2]; // B
    }
    canvasCtx.putImageData(imgData, 0, 0);

    canvas.toBlob((blob) => {
      state.currentSketchBlob = blob;
      state.currentSketchUrl = URL.createObjectURL(blob);
      showToast("Sketch inverted.", "info");
    }, "image/png");
  }

  function clearCanvas() {
    const canvas = elements.sketchCanvas;
    canvasCtx.fillStyle = "#ffffff";
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
    canvas.classList.add("hidden");
    elements.uploadEmptyState.classList.remove("hidden");
    elements.canvasOverlayBar.classList.add("hidden");
    state.currentSketchBlob = null;
    state.currentSketchUrl = null;
    elements.sketchFileInput.value = "";
    showToast("Canvas cleared.", "info");
  }

  // Canvas Mouse & Touch Drawing Handlers
  function getCanvasCoords(e) {
    const rect = elements.sketchCanvas.getBoundingClientRect();
    const scaleX = elements.sketchCanvas.width / rect.width;
    const scaleY = elements.sketchCanvas.height / rect.height;

    let clientX = e.clientX;
    let clientY = e.clientY;
    if (e.touches && e.touches.length > 0) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    }
    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY,
    };
  }

  function startDrawing(e) {
    state.isDrawing = true;
    const coords = getCanvasCoords(e);
    state.lastX = coords.x;
    state.lastY = coords.y;
  }

  function draw(e) {
    if (!state.isDrawing) return;
    e.preventDefault();
    const coords = getCanvasCoords(e);

    canvasCtx.beginPath();
    canvasCtx.moveTo(state.lastX, state.lastY);
    canvasCtx.lineTo(coords.x, coords.y);

    if (state.toolMode === "eraser") {
      canvasCtx.strokeStyle = "#ffffff";
      canvasCtx.lineWidth = state.brushSize * 4;
    } else {
      canvasCtx.strokeStyle = "#1e293b";
      canvasCtx.lineWidth = state.brushSize * 1.5;
    }
    canvasCtx.lineCap = "round";
    canvasCtx.lineJoin = "round";
    canvasCtx.stroke();

    state.lastX = coords.x;
    state.lastY = coords.y;
  }

  function stopDrawing() {
    if (!state.isDrawing) return;
    state.isDrawing = false;
    elements.sketchCanvas.toBlob((blob) => {
      state.currentSketchBlob = blob;
      state.currentSketchUrl = URL.createObjectURL(blob);
    }, "image/png");
  }

  // =========================================================================
  // LOGO OVERLAY MANAGEMENT
  // =========================================================================
  function loadLogoFile(file) {
    if (!file || !file.type.startsWith("image/")) {
      showToast("Please upload a PNG logo.", "error");
      return;
    }
    state.currentLogoBlob = file;
    state.currentLogoUrl = URL.createObjectURL(file);

    elements.logoPreviewThumb.src = state.currentLogoUrl;
    elements.logoPreviewThumb.classList.remove("hidden");
    elements.logoEmptyIcon.classList.add("hidden");
    elements.logoFilename.textContent = file.name;
    showToast(`Logo loaded: ${file.name}`, "success");
  }

  async function loadSampleBrandLogo() {
    try {
      const res = await fetch("/api/v1/samples/sample_logo.png");
      if (!res.ok) throw new Error("Sample logo not found");
      const blob = await res.blob();
      state.currentLogoBlob = blob;
      state.currentLogoUrl = URL.createObjectURL(blob);

      elements.logoPreviewThumb.src = state.currentLogoUrl;
      elements.logoPreviewThumb.classList.remove("hidden");
      elements.logoEmptyIcon.classList.add("hidden");
      elements.logoFilename.textContent = "Aura Botanica Emblem";
      showToast("Sample logo loaded.", "success");
    } catch (err) {
      showToast(`Could not load sample logo: ${err.message}`, "error");
    }
  }

  // =========================================================================
  // PROMPT ENRICHMENT
  // =========================================================================
  function enrichPrompt() {
    const catText = elements.categorySelect.options[elements.categorySelect.selectedIndex].text;
    const matText = elements.materialSelect.options[elements.materialSelect.selectedIndex].text;
    const finText = elements.finishSelect.options[elements.finishSelect.selectedIndex].text;
    const litText = elements.lightingSelect.options[elements.lightingSelect.selectedIndex].text;
    const camText = elements.viewpointSelect.options[elements.viewpointSelect.selectedIndex].text;
    const bgText = elements.backgroundSelect.options[elements.backgroundSelect.selectedIndex].text;
    const current = elements.userPrompt.value.trim();

    const parts = [
      `High-end commercial ${catText} packaging design`,
      `crafted from authentic ${matText}`,
      `highlighting ${finText}`,
      `professional ${litText} with realistic specular highlights and ambient occlusion`,
      `${camText} composition`,
      `positioned in ${bgText}`,
    ];

    if (current) {
      parts.splice(1, 0, current);
    }

    elements.userPrompt.value = parts.join(", ") + ", 8k resolution, photoreal master render.";
    showToast("Prompt enriched.", "info");
  }

  // =========================================================================
  // GENERATION LIFECYCLE & ASYNCHRONOUS POLLING
  // =========================================================================
  async function submitGeneration() {
    if (!state.currentSketchBlob) {
      showToast("Please upload or draw a packaging sketch first.", "error");
      return;
    }

    elements.btnGenerate.disabled = true;
    elements.btnGenerateText.textContent = "Uploading sketch...";
    elements.renderEmptyState.classList.add("hidden");
    elements.sliderContainer.classList.add("hidden");
    elements.sideBySideContainer.classList.add("hidden");
    elements.renderActionsPanel.classList.add("hidden");
    elements.renderProgressCard.classList.remove("hidden");
    updateProgressUI("queued", 5, "Uploading sketch...");
    hideEngineBanner();

    try {
      // 1. Upload sketch asset
      const sketchFormData = new FormData();
      sketchFormData.append("file", state.currentSketchBlob, "packaging_sketch.png");
      sketchFormData.append("asset_type", "sketch");

      const sketchUploadRes = await fetch("/api/v1/uploads", {
        method: "POST",
        body: sketchFormData,
      });

      if (!sketchUploadRes.ok) {
        throw new Error("Failed to upload sketch asset.");
      }
      const sketchUploadData = await sketchUploadRes.json();
      const inputAssetId = sketchUploadData.asset.id;

      // 2. Upload logo asset (if present)
      let logoAssetId = null;
      if (state.currentLogoBlob) {
        updateProgressUI("preprocessing", 15, "Uploading logo...");
        const logoFormData = new FormData();
        logoFormData.append("file", state.currentLogoBlob, "brand_logo.png");
        logoFormData.append("asset_type", "logo");

        const logoUploadRes = await fetch("/api/v1/uploads", {
          method: "POST",
          body: logoFormData,
        });
        if (logoUploadRes.ok) {
          const logoUploadData = await logoUploadRes.json();
          logoAssetId = logoUploadData.asset.id;
        }
      }

      // 3. Submit generation job
      updateProgressUI("queued", 20, "Queuing render...");

      const generationPayload = {
        input_asset_id: inputAssetId,
        logo_asset_id: logoAssetId,
        package_type: resolveCanonicalValue(elements.categorySelect),
        material: resolveCanonicalValue(elements.materialSelect),
        finish: resolveCanonicalValue(elements.finishSelect),
        viewpoint: resolveCanonicalValue(elements.viewpointSelect),
        background: resolveCanonicalValue(elements.backgroundSelect),
        lighting: resolveCanonicalValue(elements.lightingSelect),
        prompt: elements.userPrompt.value.trim() || undefined,
        negative_prompt: elements.negativePrompt.value.trim() || undefined,
        // Input preprocessing hint: auto | sketch | photo (see "Input Type" select)
        input_mode: document.getElementById("input-mode-select")?.value || "auto",
        steps: parseInt(elements.inputSteps.value) || 30,
        guidance_scale: parseFloat(elements.inputGuidance.value) || 7.5,
        controlnet_scale: parseFloat(elements.inputControlnetScale.value) || 0.85,
        seed: parseInt(elements.inputSeed.value) === -1 ? undefined : parseInt(elements.inputSeed.value),
        logo_placement: logoAssetId
          ? {
              x: parseFloat(elements.logoPosX.value) / 100.0,
              y: parseFloat(elements.logoPosY.value) / 100.0,
              scale: parseFloat(elements.logoScale.value) / 100.0,
              blend_mode: elements.logoBlendMode.value,
            }
          : undefined,
      };

      const genRes = await fetch("/api/v1/generations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(generationPayload),
      });

      if (!genRes.ok) {
        const err = await genRes.json();
        throw new Error(err.detail || "Failed to start generation job.");
      }

      const genData = await genRes.json();
      state.activeJobId = genData.id;
      showToast(`Job queued: ${genData.id.slice(0, 8)}`, "info");

      // 4. Start polling job status
      startJobPolling(genData.id);
    } catch (err) {
      showToast(`Failed — ${err.message}`, "error");
      elements.btnGenerate.disabled = false;
      elements.btnGenerateText.textContent = "Generate mockup";
      elements.renderProgressCard.classList.add("hidden");
      elements.renderEmptyState.classList.remove("hidden");
    }
  }

  function startJobPolling(jobId) {
    if (state.pollInterval) clearInterval(state.pollInterval);

    state.pollInterval = setInterval(async () => {
      try {
        const res = await fetch(`/api/v1/generations/${jobId}`);
        if (!res.ok) return;

        const job = await res.json();
        updateProgressUI(job.status, job.progress, job.stage_message);

        if (job.status === "completed") {
          clearInterval(state.pollInterval);
          state.pollInterval = null;
          handleGenerationCompleted(job);
        } else if (job.status === "failed" || job.status === "cancelled") {
          clearInterval(state.pollInterval);
          state.pollInterval = null;
          handleGenerationFailed(job);
        }
      } catch (err) {
        console.error("Polling error:", err);
      }
    }, 1200); // 50 req/min — stay under the 120/min read rate limit
  }

  function updateProgressUI(status, progress, stageMessage) {
    elements.progressPctBadge.textContent = `${Math.min(100, Math.max(0, progress))}%`;
    elements.progressBarFill.style.width = `${Math.min(100, Math.max(0, progress))}%`;
    elements.progressStageMsg.textContent = stageMessage || "Processing...";

    // Stepper: plain text states, no pills
    const resetPills = () => {
      [elements.stepQueued, elements.stepPre, elements.stepGen, elements.stepPost].forEach((el) => {
        el.className = "text-neutral-400";
      });
    };

    resetPills();
    if (status === "queued") {
      elements.stepQueued.className = "text-neutral-900 dark:text-neutral-100 font-medium";
      elements.progressStageTitle.textContent = "Queued";
    } else if (status === "preprocessing") {
      elements.stepQueued.className = "text-green-600 dark:text-green-400";
      elements.stepPre.className = "text-neutral-900 dark:text-neutral-100 font-medium";
      elements.progressStageTitle.textContent = "Preprocessing";
    } else if (status === "generating") {
      elements.stepQueued.className = "text-green-600 dark:text-green-400";
      elements.stepPre.className = "text-green-600 dark:text-green-400";
      elements.stepGen.className = "text-neutral-900 dark:text-neutral-100 font-medium";
      elements.progressStageTitle.textContent = "Rendering";
    } else if (status === "postprocessing") {
      elements.stepQueued.className = "text-green-600 dark:text-green-400";
      elements.stepPre.className = "text-green-600 dark:text-green-400";
      elements.stepGen.className = "text-green-600 dark:text-green-400";
      elements.stepPost.className = "text-neutral-900 dark:text-neutral-100 font-medium";
      elements.progressStageTitle.textContent = "Compositing logo";
    } else if (status === "completed") {
      [elements.stepQueued, elements.stepPre, elements.stepGen, elements.stepPost].forEach((el) => {
        el.className = "text-green-600 dark:text-green-400";
      });
      elements.progressStageTitle.textContent = "Render complete";
    }
  }

  // =========================================================================
  // ENGINE ATTRIBUTION & METRIC PROVENANCE (DISPLAY-ONLY, NEVER REQUIRED)
  // =========================================================================
  function humanizeEngine(raw) {
    const value = String(raw || "").trim();
    if (!value) return null;
    const v = value.toLowerCase();
    if (v.startsWith("cloud")) {
      const provider = value.replace(/^cloud[-_]?/i, "").replace(/[-_]+/g, " ").trim();
      return `Cloud · ${provider ? provider.replace(/\b\w/g, (c) => c.toUpperCase()) : "Provider"}`;
    }
    if (v.includes("packvisionnet")) return "Local · PackVisionNet-v2";
    if (v.includes("procedural")) return "Local · Procedural PBR";
    if (v.includes("sdxl") || v.includes("controlnet")) return "GPU · SDXL + ControlNet";
    if (v.includes("gemini")) return "Cloud · Gemini";
    // Unknown engine token: present it tidily instead of hiding it
    return value.replace(/[-_]+/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  }

  function extractEngineUsed(job) {
    const output = (job.outputs && job.outputs[0]) || {};
    const metrics = output.metrics || {};
    const candidates = [
      metrics.engine_used,
      metrics.engine,
      job.engine_used,
      job.metadata && job.metadata.engine_used,
      job.parameters_json && job.parameters_json.engine_used,
      job.model_name,
    ];
    return candidates.find((c) => c !== undefined && c !== null && String(c).trim() !== "");
  }

  function showEngineBanner(job) {
    if (!elements.engineBanner || !elements.engineBadgeText) return;
    const humanized = humanizeEngine(extractEngineUsed(job));
    if (!humanized) {
      hideEngineBanner();
      return;
    }
    elements.engineBadgeText.textContent = `Rendered by: ${humanized}`;

    // Optional fallback / provenance notes from the pipeline metadata
    const output = (job.outputs && job.outputs[0]) || {};
    const metrics = output.metrics || {};
    const note =
      metrics.engine_note ||
      metrics.fallback_note ||
      (job.metadata && (job.metadata.engine_note || job.metadata.fallback_note)) ||
      "";
    if (elements.engineNoteText) {
      if (note) {
        elements.engineNoteText.textContent = note;
        elements.engineNoteText.classList.remove("hidden");
      } else {
        elements.engineNoteText.classList.add("hidden");
      }
    }
    elements.engineBanner.classList.remove("hidden");
  }

  function hideEngineBanner() {
    if (elements.engineBanner) elements.engineBanner.classList.add("hidden");
    if (elements.engineNoteText) elements.engineNoteText.classList.add("hidden");
  }

  const ESTIMATE_SOURCES = new Set(["heuristic", "vlm", "estimate", "estimated", "proxy", "pseudo"]);

  function metricIsEstimate(metrics, metricKey) {
    if (!metrics) return false;
    const perMetric =
      (metrics.metric_sources && metrics.metric_sources[metricKey]) ||
      (metrics.provenance && metrics.provenance[metricKey]);
    const globalSource = metrics.metrics_source || metrics.provenance_source;
    const source = perMetric || globalSource || (metricKey === "clip" ? metrics.clip_source : null);
    if (!source) return false;
    return ESTIMATE_SOURCES.has(String(source).toLowerCase());
  }

  function metricChipText(value, isEstimate, suffix) {
    if (value === undefined || value === null || Number.isNaN(Number(value))) return "--";
    const base = `${Number(value).toFixed(3)}${suffix ? ` (${suffix})` : ""}`;
    return isEstimate ? `${base} · estimate` : base;
  }

  function setMetricBadge(badgeEl, value, isEstimate, suffix) {
    if (!badgeEl) return;
    badgeEl.textContent = metricChipText(value, isEstimate, suffix);
    if (isEstimate) badgeEl.title = "Heuristic estimate — not the nominal metric";
    else badgeEl.removeAttribute("title");
  }

  function handleGenerationCompleted(job) {
    state.currentGeneration = job;
    elements.btnGenerate.disabled = false;
    elements.btnGenerateText.textContent = "Generate mockup";

    elements.renderProgressCard.classList.add("hidden");
    elements.renderEmptyState.classList.add("hidden");
    elements.sliderContainer.classList.remove("hidden");
    elements.renderActionsPanel.classList.remove("hidden");

    // Populate split slider images
    const sketchUrl = job.input_asset ? job.input_asset.url : state.currentSketchUrl;
    if (sketchUrl) {
      elements.sketchPreviewImg.src = sketchUrl;
      elements.sideSketchImg.src = sketchUrl;
    }

    if (job.outputs && job.outputs.length > 0) {
      const output = job.outputs[0];
      const renderUrl = output.asset ? output.asset.url : `/api/v1/files/${output.asset_id}`;
      elements.renderResultImg.src = renderUrl;
      elements.sideRenderImg.src = renderUrl;

      // Quality Scorecards — show "--" when a metric is absent, tag estimates
      const metrics = output.metrics || {};
      setMetricBadge(elements.metricSsimBadge, metrics.edge_ssim, metricIsEstimate(metrics, "ssim"), "");
      setMetricBadge(elements.metricClipBadge, metrics.clip_score, metricIsEstimate(metrics, "clip"), "");
      setMetricBadge(elements.metricIouBadge, metrics.edge_iou, metricIsEstimate(metrics, "iou"), "");
      // If the photoreal cloud engine was unavailable and we silently fell back
      // to the local preview model, say so loudly — never pass a local preview
      // off as a photoreal render. Then AUTO-RETRY once after the free-tier
      // cooldown (~75s): the rolling quota usually frees up within a couple of
      // minutes, so the user gets the photoreal render without babysitting.
      const fallbackNote = String(metrics.engine_note || "");
      if (fallbackNote.toLowerCase().includes("cloud unavailable")) {
        showToast(
          `Photoreal engine busy (${fallbackNote.replace("cloud unavailable (", "").replace("); fell back to local engine", "")}). ` +
            "Local preview shown — auto-retrying for photoreal quality…",
          "error"
        );
        const jobId = job.id;
        state.autoPhotorealRetries = state.autoPhotorealRetries || {};
        if (!state.autoPhotorealRetries[jobId]) {
          state.autoPhotorealRetries[jobId] = true;
          setTimeout(async () => {
            try {
              const res = await fetch(`/api/v1/generations/${jobId}/retry`, { method: "POST" });
              if (res.ok) {
                showToast("Photoreal engine re-queued — rendering again…", "info");
                elements.btnGenerate.disabled = true;
                elements.renderEmptyState.classList.add("hidden");
                elements.renderProgressCard.classList.remove("hidden");
                startJobPolling(jobId);
              } else if (res.status === 429) {
                showToast("Free-tier quota is exhausted for now — try again in a few minutes.", "error");
              }
            } catch (e) {
              /* silent: local preview already available */
            }
          }, 90000);
        }
      }
      const latencyMs = metrics.inference_time_ms ?? metrics.inference_time;
      const latency = latencyMs !== undefined && latencyMs !== null
        ? `${(Number(latencyMs) / 1000).toFixed(2)}s`
        : "--";
      elements.metricLatencyBadge.textContent = latency;
      if (latencyMs === undefined || latencyMs === null) {
        elements.metricLatencyBadge.removeAttribute("title");
      }
    } else {
      ["metricSsimBadge", "metricClipBadge", "metricIouBadge", "metricLatencyBadge"].forEach((key) => {
        if (elements[key]) elements[key].textContent = "--";
      });
    }

    // Engine attribution badge (only when the record carries engine info)
    showEngineBanner(job);

    // Set slider handle to 50%
    setSliderPosition(50);

    // Add to history
    addToHistory(job);

    showToast("Render complete", "success");
  }

  function handleGenerationFailed(job) {
    elements.btnGenerate.disabled = false;
    elements.btnGenerateText.textContent = "Generate mockup";
    elements.renderProgressCard.classList.add("hidden");
    elements.sliderContainer.classList.add("hidden");
    elements.renderActionsPanel.classList.add("hidden");
    hideEngineBanner();
    elements.renderEmptyState.classList.remove("hidden");
    showToast(`Failed — ${job.error_message || "see note"}`, "error");
  }

  // =========================================================================
  // INTERACTIVE SPLIT COMPARISON SLIDER
  // =========================================================================
  function setSliderPosition(percentage) {
    const clamped = Math.max(0, Math.min(100, percentage));
    elements.sliderHandle.style.left = `${clamped}%`;
    elements.renderClipLayer.style.clipPath = `polygon(${clamped}% 0, 100% 0, 100% 100%, ${clamped}% 100%)`;
  }

  function onSliderMove(e) {
    if (!state.sliderActive) return;
    const rect = elements.sliderContainer.getBoundingClientRect();
    let clientX = e.clientX;
    if (e.touches && e.touches.length > 0) {
      clientX = e.touches[0].clientX;
    }
    const offset = clientX - rect.left;
    const pct = (offset / rect.width) * 100;
    setSliderPosition(pct);
  }

  // =========================================================================
  // VIEW MODE SWITCHER
  // =========================================================================
  function switchViewMode(mode) {
    state.viewMode = mode;
    const inactive = "px-3 py-1 rounded-full text-sm text-neutral-400 transition";
    const active = "px-3 py-1 rounded-full text-sm font-semibold text-neutral-900 dark:text-white transition";
    elements.viewModeSplit.className = inactive;
    elements.viewModeSide.className = inactive;
    elements.viewModeRender.className = inactive;

    if (mode === "split") {
      elements.viewModeSplit.className = active;
      elements.sliderContainer.classList.remove("hidden");
      elements.sideBySideContainer.classList.add("hidden");
      setSliderPosition(50);
    } else if (mode === "side") {
      elements.viewModeSide.className = active;
      elements.sliderContainer.classList.add("hidden");
      elements.sideBySideContainer.classList.remove("hidden");
    } else if (mode === "render") {
      elements.viewModeRender.className = active;
      elements.sliderContainer.classList.remove("hidden");
      elements.sideBySideContainer.classList.add("hidden");
      setSliderPosition(0); // Reveal full render
    }
  }

  // =========================================================================
  // DOWNLOAD ACTIONS
  // =========================================================================
  function downloadRender() {
    if (!state.currentGeneration || !state.currentGeneration.outputs || !state.currentGeneration.outputs.length) {
      showToast("No active render to download.", "error");
      return;
    }
    const output = state.currentGeneration.outputs[0];
    const url = output.asset ? output.asset.url : `/api/v1/files/${output.asset_id}`;
    const a = document.createElement("a");
    a.href = url;
    a.download = `packvision_${state.currentGeneration.package_type}_${state.currentGeneration.id.slice(0, 8)}.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    showToast("Download started.", "success");
  }

  function downloadMetadata() {
    if (!state.currentGeneration) {
      showToast("No generation metadata available.", "error");
      return;
    }
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(state.currentGeneration, null, 2));
    const a = document.createElement("a");
    a.href = dataStr;
    a.download = `packvision_metadata_${state.currentGeneration.id.slice(0, 8)}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    showToast("Metadata downloaded.", "success");
  }

  // =========================================================================
  // DESIGN REVIEW & FEEDBACK MODAL
  // =========================================================================
  function openFeedbackModal() {
    if (!state.currentGeneration) {
      showToast("Please generate a mockup before reviewing.", "error");
      return;
    }
    elements.feedbackModal.classList.remove("hidden");
  }

  function closeFeedbackModal() {
    elements.feedbackModal.classList.add("hidden");
  }

  function setStarRating(rating) {
    state.feedbackRating = rating;
    const labels = {
      1: "1.0 / 5.0 (Poor)",
      2: "2.0 / 5.0 (Fair)",
      3: "3.0 / 5.0 (Good)",
      4: "4.0 / 5.0 (Very Good)",
      5: "5.0 / 5.0 (Outstanding)",
    };
    elements.starRatingLabel.textContent = labels[rating] || `${rating}.0 / 5.0`;

    elements.starBtns.forEach((btn) => {
      const val = parseInt(btn.dataset.val);
      if (val <= rating) {
        btn.classList.add("text-red-600");
        btn.classList.remove("text-neutral-300", "dark:text-neutral-600");
      } else {
        btn.classList.remove("text-red-600");
        btn.classList.add("text-neutral-300", "dark:text-neutral-600");
      }
    });
  }

  async function submitFeedback() {
    if (!state.currentGeneration) return;

    const payload = {
      rating: state.feedbackRating,
      issue_categories: {
        shape_preserved: document.getElementById("crit-shape").checked,
        material_realistic: document.getElementById("crit-material").checked,
        finish_accurate: document.getElementById("crit-finish").checked,
        lighting_convincing: document.getElementById("crit-lighting").checked,
        logo_crisp: document.getElementById("crit-logo").checked,
      },
      comment: elements.feedbackComment.value.trim() || undefined,
    };

    try {
      const res = await fetch(`/api/v1/generations/${state.currentGeneration.id}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error("Could not record feedback");

      closeFeedbackModal();
      showToast("Feedback recorded.", "success");
    } catch (err) {
      showToast(`Failed — ${err.message}`, "error");
    }
  }

  // =========================================================================
  // GENERATION HISTORY DRAWER (PERSISTENT VIA GET /api/v1/generations)
  // =========================================================================
  function historyStatusInfo(job) {
    switch (job.status) {
      case "completed":
        return { label: "Done", cls: "history-status-done" };
      case "failed":
        return { label: "Failed", cls: "history-status-failed" };
      case "cancelled":
        return { label: "Cancelled", cls: "history-status-queued" };
      case "queued":
        return { label: "Queued", cls: "history-status-queued" };
      default:
        return { label: "Running", cls: "history-status-running" };
    }
  }

  function historyThumbUrl(job) {
    const output = job.outputs && job.outputs[0];
    if (output) {
      if (output.asset && output.asset.thumbnail_url) return output.asset.thumbnail_url;
      if (output.asset_id) return `/api/v1/files/${output.asset_id}?thumb=1`;
    }
    if (job.input_asset && job.input_asset.thumbnail_url) return job.input_asset.thumbnail_url;
    if (job.input_asset_id) return `/api/v1/files/${job.input_asset_id}?thumb=1`;
    return "/api/v1/samples/carton_sketch.png";
  }

  function historyEntryTitle(job) {
    const type = taxonomyLabelFor(elements.categorySelect, job.package_type) || job.package_type || "render";
    const material = job.material
      ? taxonomyLabelFor(elements.materialSelect, job.material)
      : "";
    return `${String(type).replace(/_/g, " ")}${material ? " • " + String(material).replace(/_/g, " ") : ""}`;
  }

  function formatHistoryDate(job) {
    if (!job.created_at) return "";
    const date = new Date(job.created_at);
    if (Number.isNaN(date.getTime())) return "";
    return date.toLocaleString([], {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = String(text);
    return div.innerHTML;
  }

  function mergeHistory(items) {
    const byId = new Map();
    // Local session jobs first (fresher metadata), then backend records.
    state.history.forEach((job) => byId.set(job.id, job));
    (items || []).forEach((job) => {
      if (job && job.id && !byId.has(job.id)) byId.set(job.id, job);
    });
    state.history = Array.from(byId.values()).sort((a, b) => {
      const da = new Date(a.created_at || 0).getTime();
      const db = new Date(b.created_at || 0).getTime();
      return db - da; // newest first
    });
    elements.historyBadgeCount.textContent = String(state.history.length);
  }

  function addToHistory(job) {
    const byId = state.history.filter((j) => j.id !== job.id);
    byId.unshift(job);
    state.history = byId.sort((a, b) => {
      const da = new Date(a.created_at || 0).getTime();
      const db = new Date(b.created_at || 0).getTime();
      return db - da;
    });
    elements.historyBadgeCount.textContent = String(state.history.length);
    renderHistoryDrawer();
  }

  async function fetchBackendHistory() {
    try {
      const res = await fetch("/api/v1/generations?limit=50&offset=0");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const items = Array.isArray(data) ? data : Array.isArray(data.items) ? data.items : [];
      state.backendHistoryAvailable = true;
      mergeHistory(items);
      renderHistoryDrawer();
    } catch (err) {
      state.backendHistoryAvailable = false;
      console.warn("[PackVision] Generation history unavailable:", err.message);
      renderHistoryDrawer();
    }
  }

  function renderHistoryDrawer() {
    const list = elements.historyItemsList;
    if (!list) return;
    list.innerHTML = "";

    if (state.history.length === 0) {
      const msg = state.backendHistoryAvailable
        ? "No renders yet.<br>Generate a mockup to see it here."
        : "History unavailable — backend offline.<br>Start the backend and reopen History.";
      list.innerHTML = `<p class="text-xs text-neutral-400 dark:text-neutral-500 text-center py-8">${msg}</p>`;
      return;
    }

    state.history.forEach((job) => {
      const status = historyStatusInfo(job);
      const item = document.createElement("div");
      item.className = "history-item rounded-xl overflow-hidden p-2 bg-white dark:bg-neutral-900 cursor-pointer flex items-center gap-3 hover:shadow-sm transition";
      item.title = `Open generation ${job.id.slice(0, 8)} (${job.status})`;

      item.innerHTML = `
        <img src="${escapeHtml(historyThumbUrl(job))}" class="w-14 h-14 rounded-lg object-cover bg-neutral-100 dark:bg-neutral-800 shrink-0" alt="Thumbnail"
             onerror="this.src='/api/v1/samples/carton_sketch.png'">
        <div class="flex-1 min-w-0">
          <div class="flex items-center justify-between gap-2">
            <span class="text-xs font-medium text-neutral-900 dark:text-neutral-100 truncate" style="text-transform: capitalize;">${escapeHtml(historyEntryTitle(job))}</span>
            <span class="text-[10px] ${status.cls}">${status.label}</span>
          </div>
          <p class="text-[11px] text-neutral-500 dark:text-neutral-400 truncate mt-0.5">${escapeHtml(job.prompt || "No prompt")}</p>
          <span class="text-[10px] text-neutral-400 dark:text-neutral-500">${escapeHtml(formatHistoryDate(job))}</span>
        </div>
      `;

      item.addEventListener("click", () => openHistoryJob(job));

      list.appendChild(item);
    });
  }

  function restoreSettingsFromJob(job) {
    if (!job) return;
    setSelectValueIfPresent(elements.categorySelect, job.package_type);
    setSelectValueIfPresent(elements.materialSelect, job.material);
    setSelectValueIfPresent(elements.finishSelect, job.finish);
    setSelectValueIfPresent(elements.viewpointSelect, job.viewpoint);
    setSelectValueIfPresent(elements.backgroundSelect, job.background);
    elements.userPrompt.value = job.prompt || "";
    elements.negativePrompt.value = job.negative_prompt || "";
    if (job.seed !== undefined && job.seed !== null) elements.inputSeed.value = job.seed;
    const params = job.parameters_json || {};
    if (params.steps) elements.inputSteps.value = params.steps;
    if (params.guidance_scale) elements.inputGuidance.value = params.guidance_scale;
    if (params.controlnet_scale) elements.inputControlnetScale.value = params.controlnet_scale;
  }

  async function openHistoryJob(job) {
    elements.historyDrawer.classList.add("translate-x-full");
    await waitForTaxonomyReady();
    restoreSettingsFromJob(job);

    if (job.status === "completed") {
      handleGenerationCompleted(job);
    } else if (job.status === "failed" || job.status === "cancelled") {
      state.currentGeneration = job;
      elements.renderProgressCard.classList.add("hidden");
      elements.sliderContainer.classList.add("hidden");
      elements.renderActionsPanel.classList.add("hidden");
      hideEngineBanner();
      elements.renderEmptyState.classList.remove("hidden");
      showToast(
        `Generation ${job.id.slice(0, 8)} ${job.status}: ${job.error_message || "no error detail recorded"}`,
        "error"
      );
    } else {
      // queued / processing record from a previous session
      state.currentGeneration = job;
      elements.renderEmptyState.classList.add("hidden");
      elements.renderProgressCard.classList.remove("hidden");
      updateProgressUI(job.status, job.progress, job.stage_message);
      showToast(`Generation ${job.id.slice(0, 8)} is ${job.status} — resuming status polling.`, "info");
      startJobPolling(job.id);
    }
  }

  // =========================================================================
  // EVENT BINDINGS
  // =========================================================================
  function bindEvents() {
    // Sketch file input & drop zone
    elements.sketchFileInput.addEventListener("change", (e) => {
      if (e.target.files.length) loadSketchFile(e.target.files[0]);
    });

    ["dragenter", "dragover"].forEach((eventName) => {
      elements.dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        elements.dropZone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      elements.dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        elements.dropZone.classList.remove("dragover");
      });
    });

    elements.dropZone.addEventListener("drop", (e) => {
      if (e.dataTransfer.files.length) {
        loadSketchFile(e.dataTransfer.files[0]);
      }
    });

    // Sample buttons
    document.querySelectorAll(".sample-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        loadSampleSketch(btn.dataset.sample);
      });
    });

    // Canvas tools
    elements.btnInvertLines.addEventListener("click", invertCanvasLines);
    elements.btnClearCanvas.addEventListener("click", clearCanvas);

    // Re-detect: re-run auto-detection on the current canvas content
    elements.btnRedetect.addEventListener("click", redetectFromCanvas);

    document.querySelectorAll("input[name='tool-mode']").forEach((radio) => {
      radio.addEventListener("change", (e) => {
        state.toolMode = e.target.value;
      });
    });

    elements.brushSizeInput.addEventListener("input", (e) => {
      state.brushSize = parseInt(e.target.value);
      elements.brushSizeVal.textContent = `${state.brushSize}px`;
    });

    // Canvas drawing events
    elements.sketchCanvas.addEventListener("mousedown", startDrawing);
    elements.sketchCanvas.addEventListener("mousemove", draw);
    window.addEventListener("mouseup", stopDrawing);

    elements.sketchCanvas.addEventListener("touchstart", startDrawing, { passive: false });
    elements.sketchCanvas.addEventListener("touchmove", draw, { passive: false });
    window.addEventListener("touchend", stopDrawing);

    // Prompt enrich
    elements.btnEnrichPrompt.addEventListener("click", enrichPrompt);

    // Material choice constrains compatible finishes (taxonomy metadata)
    elements.materialSelect.addEventListener("change", applyFinishCompatibility);

    // Logo panel toggle & sliders
    elements.toggleLogoPanel.addEventListener("click", () => {
      const isHidden = elements.logoPanelContent.classList.contains("hidden");
      elements.logoPanelContent.classList.toggle("hidden");
      elements.logoPanelIcon.style.transform = isHidden ? "rotate(180deg)" : "rotate(0deg)";
    });

    elements.logoFileInput.addEventListener("change", (e) => {
      if (e.target.files.length) loadLogoFile(e.target.files[0]);
    });

    elements.btnLoadSampleLogo.addEventListener("click", loadSampleBrandLogo);

    elements.logoPosX.addEventListener("input", (e) => {
      elements.logoXVal.textContent = `${e.target.value}%`;
    });
    elements.logoPosY.addEventListener("input", (e) => {
      elements.logoYVal.textContent = `${e.target.value}%`;
    });
    elements.logoScale.addEventListener("input", (e) => {
      elements.logoScaleVal.textContent = `${e.target.value}%`;
    });

    // Advanced panel toggle
    elements.toggleAdvancedPanel.addEventListener("click", () => {
      const isHidden = elements.advancedPanelContent.classList.contains("hidden");
      elements.advancedPanelContent.classList.toggle("hidden");
      elements.advancedPanelIcon.style.transform = isHidden ? "rotate(180deg)" : "rotate(0deg)";
    });

    // Generate CTA (main pill + compact header pill)
    elements.btnGenerate.addEventListener("click", submitGeneration);
    // Dark / light theme toggle (persisted)
    // NOTE: the vendored Tailwind build boots with darkMode:"media" and ignores
    // post-load strategy changes unless the config is re-assigned AND the DOM
    // mutates (its recompile trigger). Without this shim the toggle is a no-op
    // for anyone whose OS prefers dark colors.
    const nudgeTailwind = () => {
      try {
        document.body.dataset.twNudge = String(Date.now());
        document.body.appendChild(document.createComment("tw-refresh"));
      } catch (e) { /* non-fatal */ }
    };
    try {
      if (window.tailwind && window.tailwind.config) {
        window.tailwind.config = Object.assign({}, window.tailwind.config, { darkMode: "class" });
        nudgeTailwind();
      }
    } catch (e) { /* non-fatal */ }
    const applyTheme = (dark) => {
      document.documentElement.classList.toggle("dark", dark);
      if (elements.themeIconMoon) elements.themeIconMoon.classList.toggle("hidden", dark);
      if (elements.themeIconSun) elements.themeIconSun.classList.toggle("hidden", !dark);
      nudgeTailwind();
      try { localStorage.setItem("packvision-theme", dark ? "dark" : "light"); } catch (e) { /* private mode */ }
    };
    let savedTheme = "light";
    try { savedTheme = localStorage.getItem("packvision-theme") || "light"; } catch (e) { /* private mode */ }
    applyTheme(savedTheme === "dark");
    if (elements.themeToggleBtn) {
      elements.themeToggleBtn.addEventListener("click", () => {
        applyTheme(!document.documentElement.classList.contains("dark"));
      });
    }

    // Slider interaction
    elements.sliderContainer.addEventListener("mousedown", (e) => {
      state.sliderActive = true;
      onSliderMove(e);
    });
    window.addEventListener("mouseup", () => {
      state.sliderActive = false;
    });
    window.addEventListener("mousemove", onSliderMove);

    elements.sliderContainer.addEventListener("touchstart", (e) => {
      state.sliderActive = true;
      onSliderMove(e);
    });
    window.addEventListener("touchend", () => {
      state.sliderActive = false;
    });
    window.addEventListener("touchmove", onSliderMove);

    // View mode switches
    elements.viewModeSplit.addEventListener("click", () => switchViewMode("split"));
    elements.viewModeSide.addEventListener("click", () => switchViewMode("side"));
    elements.viewModeRender.addEventListener("click", () => switchViewMode("render"));

    // Downloads
    elements.btnDownloadRender.addEventListener("click", downloadRender);
    elements.btnDownloadMetadata.addEventListener("click", downloadMetadata);

    // Feedback modal
    elements.btnOpenFeedback.addEventListener("click", openFeedbackModal);
    elements.btnCloseFeedback.addEventListener("click", closeFeedbackModal);
    elements.btnCancelFeedback.addEventListener("click", closeFeedbackModal);
    elements.btnSubmitFeedback.addEventListener("click", submitFeedback);

    elements.starBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        setStarRating(parseInt(btn.dataset.val));
      });
    });

    // History drawer (re-syncs with backend each time it opens)
    elements.toggleHistoryBtn.addEventListener("click", () => {
      elements.historyDrawer.classList.toggle("translate-x-full");
      if (!elements.historyDrawer.classList.contains("translate-x-full")) {
        fetchBackendHistory();
      }
    });
    elements.btnCloseDrawer.addEventListener("click", () => {
      elements.historyDrawer.classList.add("translate-x-full");
    });
  }

  // =========================================================================
  // BOOTSTRAP INITIALIZATION
  // =========================================================================
  document.addEventListener("DOMContentLoaded", () => {
    bindEvents();
    checkBackendHealth();
    fetchTaxonomy();
    fetchBackendHistory();
    // Pre-populate with Carton template sketch for instant visual delight
    setTimeout(() => {
      loadSampleSketch("carton");
    }, 200);
  });
})();
