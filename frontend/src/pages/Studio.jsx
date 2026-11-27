import React, { useState, useEffect, useRef } from 'react';
import {
  Upload,
  Sparkles,
  Sliders,
  Image as ImageIcon,
  Layers,
  Split,
  Download,
  RotateCw,
  X,
  Check,
  AlertCircle,
  Play,
  StopCircle,
  Award,
  RefreshCw,
  FolderPlus,
  Bookmark,
  PanelLeft,
  PanelRight,
  Maximize2,
  Minimize2,
  ChevronDown,
  ChevronUp,
  SlidersHorizontal,
  Sun,
  Eye,
  FileText,
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import { api } from '../services/api';
import BeforeAfterSlider from '../components/BeforeAfterSlider';
import LogoEditor from '../components/LogoEditor';

export default function Studio() {
  const {
    studioInitialState,
    setStudioInitialState,
    taxonomy,
    showNotification,
    openEvaluationModal,
    activeProject,
  } = useApp();

  // Layout & Collapsible Panels State
  const [isLeftOpen, setIsLeftOpen] = useState(true);
  const [isRightOpen, setIsRightOpen] = useState(true);
  const [isFocusMode, setIsFocusMode] = useState(false);

  // Accordion Section Toggles (Clean UI)
  const [showLightingSection, setShowLightingSection] = useState(false);
  const [showPromptsSection, setShowPromptsSection] = useState(false);
  const [showAdvancedDiffusion, setShowAdvancedDiffusion] = useState(false);

  // Input & Customization State
  const [sketchFile, setSketchFile] = useState(null);
  const [sketchAssetId, setSketchAssetId] = useState(null);
  const [sketchPreviewUrl, setSketchPreviewUrl] = useState(null);

  const [packageType, setPackageType] = useState('carton');
  const [material, setMaterial] = useState('coated_sbs');
  const [finish, setFinish] = useState('matte');
  const [viewpoint, setViewpoint] = useState('three_quarter');
  const [background, setBackground] = useState('seamless_studio');
  const [lighting, setLighting] = useState('studio_softbox');
  const [prompt, setPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState(
    'distorted packaging, warped geometry, deformed edges, blurry text, extra objects, unrealistic shadows'
  );

  // Settings State
  const [steps, setSteps] = useState(30);
  const [guidanceScale, setGuidanceScale] = useState(7.5);
  const [controlnetScale, setControlnetScale] = useState(0.85);
  const [seed, setSeed] = useState(null);
  const [resolution, setResolution] = useState(512);

  // Preprocessing State
  const [cannyLow, setCannyLow] = useState(50);
  const [cannyHigh, setCannyHigh] = useState(150);
  const [preprocessedMaps, setPreprocessedMaps] = useState(null);
  const [isPreprocessing, setIsPreprocessing] = useState(false);

  // Active Center Workspace View Tab: 'render' | 'maps' | 'logo'
  const [activeCenterTab, setActiveCenterTab] = useState('render');

  // Generation Lifecycle State
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentJobId, setCurrentJobId] = useState(null);
  const [jobProgress, setJobProgress] = useState(0);
  const [jobStageMessage, setJobStageMessage] = useState('');
  const [activeRenderResult, setActiveRenderResult] = useState(null);
  const [activeRenderAssetId, setActiveRenderAssetId] = useState(null);
  const [rawRenderResult, setRawRenderResult] = useState(null);
  const [rawRenderAssetId, setRawRenderAssetId] = useState(null);
  const [generationMeta, setGenerationMeta] = useState(null);

  // Logo asset state
  const [logoAssetId, setLogoAssetId] = useState(null);
  const [logoPreviewUrl, setLogoPreviewUrl] = useState(null);

  // Project selector
  const [projectsList, setProjectsList] = useState([]);
  const [selectedProjectId, setSelectedProjectId] = useState(activeProject?.id || '');

  const pollingRef = useRef(null);
  const fileInputRef = useRef(null);
  const [isDragOverLeft, setIsDragOverLeft] = useState(false);

  // Load projects list
  useEffect(() => {
    api.listProjects({ limit: 50 }).then((res) => {
      setProjectsList(res.items || []);
    }).catch((e) => console.error(e));
  }, []);

  // Hydrate from studioInitialState if passed from Dashboard / Projects / History
  useEffect(() => {
    if (studioInitialState) {
      if (studioInitialState.packageType) setPackageType(studioInitialState.packageType);
      if (studioInitialState.material) setMaterial(studioInitialState.material);
      if (studioInitialState.finish) setFinish(studioInitialState.finish);
      if (studioInitialState.viewpoint) setViewpoint(studioInitialState.viewpoint);
      if (studioInitialState.background) setBackground(studioInitialState.background);
      if (studioInitialState.prompt) setPrompt(studioInitialState.prompt);
      if (studioInitialState.negativePrompt) setNegativePrompt(studioInitialState.negativePrompt);
      if (studioInitialState.projectId) setSelectedProjectId(studioInitialState.projectId);

      const inId = studioInitialState.inputAssetId || studioInitialState.inputAsset?.id;
      const inUrl = studioInitialState.inputAssetUrl || studioInitialState.inputAsset?.url || (inId ? `/api/v1/files/${inId}` : null);
      if (inUrl) {
        setSketchAssetId(inId);
        setSketchPreviewUrl(inUrl);
      }

      const outId = studioInitialState.completedOutputId || studioInitialState.completedOutput?.id;
      const outUrl = studioInitialState.completedOutputUrl || studioInitialState.completedOutput?.url || (outId ? `/api/v1/files/${outId}` : null);
      if (outUrl) {
        setActiveRenderResult(outUrl);
        setActiveRenderAssetId(outId);
        setRawRenderResult(outUrl);
        setRawRenderAssetId(outId);
      }

      if (studioInitialState.completedGen) {
        setGenerationMeta(studioInitialState.completedGen);
      }
      setStudioInitialState(null);
    }
  }, [studioInitialState]);

  // Sync activeRenderResult whenever generationMeta is present
  useEffect(() => {
    if (generationMeta) {
      const firstOut = generationMeta.outputs && generationMeta.outputs[0];
      const outputAsset = firstOut?.asset || (generationMeta.assets?.find((a) => a.asset_type === 'output'));
      const outputAssetId = firstOut?.asset_id || outputAsset?.id || generationMeta.output_asset_id;
      const outputUrl = outputAsset?.url || (outputAssetId ? `/api/v1/files/${outputAssetId}` : null);

      if (outputUrl && !activeRenderResult) {
        setActiveRenderResult(outputUrl);
        setActiveRenderAssetId(outputAssetId);
        setRawRenderResult(outputUrl);
        setRawRenderAssetId(outputAssetId);
      }

      const inAsset = generationMeta.input_asset || (generationMeta.assets?.find((a) => a.asset_type === 'sketch'));
      const inAssetId = generationMeta.input_asset_id || inAsset?.id;
      const inUrl = inAsset?.url || (inAssetId ? `/api/v1/files/${inAssetId}` : null);
      if (inUrl && !sketchPreviewUrl) {
        setSketchAssetId(inAssetId);
        setSketchPreviewUrl(inUrl);
      }
    }
  }, [generationMeta, activeRenderResult, sketchPreviewUrl]);

  // Auto-restore latest generation on initial mount if workspace is empty
  useEffect(() => {
    let isMounted = true;
    const restoreLatest = async () => {
      if (!sketchPreviewUrl && !activeRenderResult && !studioInitialState) {
        try {
          const res = await api.listGenerations({ limit: 1 });
          if (isMounted && res.items && res.items.length > 0) {
            const latest = res.items[0];
            setGenerationMeta(latest);

            const inAsset = latest.input_asset || (latest.assets?.find((a) => a.asset_type === 'sketch'));
            const inAssetId = latest.input_asset_id || inAsset?.id;
            const inUrl = inAsset?.url || (inAssetId ? `/api/v1/files/${inAssetId}` : null);
            if (inUrl) {
              setSketchAssetId(inAssetId);
              setSketchPreviewUrl(inUrl);
            }

            const firstOut = latest.outputs && latest.outputs[0];
            const outputAsset = firstOut?.asset || (latest.assets?.find((a) => a.asset_type === 'output'));
            const outputAssetId = firstOut?.asset_id || outputAsset?.id || latest.output_asset_id;
            const outputUrl = outputAsset?.url || (outputAssetId ? `/api/v1/files/${outputAssetId}` : null);
            if (outputUrl) {
              setActiveRenderResult(outputUrl);
              setActiveRenderAssetId(outputAssetId);
              setRawRenderResult(outputUrl);
              setRawRenderAssetId(outputAssetId);
            }

            if (latest.package_type) setPackageType(latest.package_type);
            if (latest.material) setMaterial(latest.material);
            if (latest.finish) setFinish(latest.finish);
            if (latest.viewpoint) setViewpoint(latest.viewpoint);
            if (latest.background) setBackground(latest.background);
            if (latest.prompt) setPrompt(latest.prompt);
          }
        } catch (e) {
          // ignore silent fail on background restore
        }
      }
    };
    restoreLatest();
    return () => {
      isMounted = false;
    };
  }, []);

  // Clean polling on unmount
  useEffect(() => {
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, []);

  // Auto Prompt Synthesis Generator
  const generatePromptFromSpecs = () => {
    const pkgLabel = packageType.replace(/_/g, ' ');
    const matLabel = material.replace(/_/g, ' ');
    const finLabel = finish.replace(/_/g, ' ');
    const lightLabel = lighting.replace(/_/g, ' ');
    const bgLabel = background.replace(/_/g, ' ');

    const synthesized = `Photorealistic commercial packaging product photograph of a premium ${pkgLabel}, made of authentic ${matLabel} with ${finLabel} surface finish, illuminated with professional ${lightLabel}, positioned on a clean ${bgLabel}, crisp structural edges, authentic contact shadows, high material detail, 8k resolution studio mockup.`;
    setPrompt(synthesized);
    showNotification('Prompt synthesized from taxonomy choices!', 'info');
  };

  // Preprocessing Trigger
  const triggerPreprocessing = async (assetId = sketchAssetId, file = sketchFile) => {
    if (!assetId && !file) return;

    try {
      setIsPreprocessing(true);
      const res = await api.preprocessSketch({
        assetId,
        file,
        lowThreshold: cannyLow,
        highThreshold: cannyHigh,
        targetSize: resolution,
      });

      if (res.success) {
        setPreprocessedMaps(res);
        showNotification('Structural edge maps computed successfully.', 'success');
      }
    } catch (err) {
      showNotification(`Preprocessing failed: ${err.message}`, 'error');
    } finally {
      setIsPreprocessing(false);
    }
  };

  // Dedicated image processor for sketch upload (shared between file picker and drag-drop)
  const processSketchFile = async (file) => {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      showNotification('Please select a valid image file (PNG, JPG, JPEG, WebP).', 'error');
      return;
    }

    try {
      showNotification('Uploading sketch...', 'info');
      setSketchFile(file);
      setSketchPreviewUrl(URL.createObjectURL(file));

      const uploadRes = await api.uploadAsset(file, 'sketch');
      if (uploadRes.success && uploadRes.asset) {
        setSketchAssetId(uploadRes.asset.id);
        showNotification('Sketch uploaded. Computing structural edge maps...', 'success');
        await triggerPreprocessing(uploadRes.asset.id, null);
      }
    } catch (err) {
      showNotification(`Upload failed: ${err.message}`, 'error');
    }
  };

  // Upload handler for sketch file input
  const handleSketchUpload = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      processSketchFile(file);
    }
    // Reset input value so selecting the same file triggers onChange
    e.target.value = '';
  };

  // Reset sketch inputs
  const handleClearSketch = () => {
    setSketchFile(null);
    setSketchAssetId(null);
    setSketchPreviewUrl(null);
    setPreprocessedMaps(null);
    showNotification('Sketch cleared.', 'info');
  };

  // Sample quick loader
  const handleLoadSample = async (sampleKey) => {
    try {
      showNotification(`Loading sample ${sampleKey} sketch...`, 'info');
      const samplesRes = await api.getSamples();
      const samples = samplesRes.samples || [];
      const chosen = samples.find((s) => s.id?.includes(sampleKey)) || samples[0];

      if (chosen) {
        setSketchPreviewUrl(chosen.url);
        setSketchAssetId(chosen.id);
        setPackageType(chosen.package_type || 'carton');
        if (chosen.prompt) setPrompt(chosen.prompt);

        const uploadRes = await fetch(chosen.url);
        const blob = await uploadRes.blob();
        const file = new File([blob], chosen.filename, { type: 'image/png' });
        const postRes = await api.uploadAsset(file, 'sketch');
        if (postRes.success && postRes.asset) {
          setSketchAssetId(postRes.asset.id);
          await triggerPreprocessing(postRes.asset.id, null);
        }
        showNotification(`Sample ${sampleKey} loaded.`, 'success');
      }
    } catch (err) {
      showNotification(`Failed to load sample: ${err.message}`, 'error');
    }
  };

  // Polling Job Status
  const pollJobStatus = (jobId) => {
    if (pollingRef.current) clearInterval(pollingRef.current);

    pollingRef.current = setInterval(async () => {
      try {
        const job = await api.getGeneration(jobId);
        setCurrentJobId(jobId);
        setJobProgress(job.progress || 0);
        setJobStageMessage(job.stage_message || 'Rendering in progress...');

        if (job.status === 'completed') {
          clearInterval(pollingRef.current);
          setIsGenerating(false);
          setJobProgress(100);
          setGenerationMeta(job);

          const firstOut = job.outputs && job.outputs[0];
          const outputAsset = firstOut?.asset || (job.assets?.find((a) => a.asset_type === 'output'));
          const outputAssetId = firstOut?.asset_id || outputAsset?.id || job.output_asset_id;
          const outputUrl = outputAsset?.url || (outputAssetId ? `/api/v1/files/${outputAssetId}` : null);

          if (outputUrl) {
            setActiveRenderResult(outputUrl);
            setActiveRenderAssetId(outputAssetId);
            setRawRenderResult(outputUrl);
            setRawRenderAssetId(outputAssetId);
          }

          if (job.input_asset) {
            setSketchAssetId((prev) => prev || job.input_asset.id);
            setSketchPreviewUrl((prev) => prev || job.input_asset.url);
          }
          showNotification('Packaging render generated successfully!', 'success');
        } else if (job.status === 'failed') {
          clearInterval(pollingRef.current);
          setIsGenerating(false);
          showNotification(`Render generation failed: ${job.error_message || 'Unknown error'}`, 'error');
        }
      } catch (err) {
        console.error('Job polling error:', err);
      }
    }, 1200);
  };

  // Trigger Generation
  const handleGenerate = async () => {
    if (!sketchAssetId) {
      showNotification('Please upload or select a packaging sketch first.', 'error');
      return;
    }

    try {
      setIsGenerating(true);
      setJobProgress(5);
      setJobStageMessage('Submitting synthesis job to inference worker...');

      let finalPrompt = prompt.trim();
      if (!finalPrompt) {
        const pkgLabel = packageType.replace(/_/g, ' ');
        const matLabel = material.replace(/_/g, ' ');
        const finLabel = finish.replace(/_/g, ' ');
        finalPrompt = `Photorealistic commercial packaging product photograph of a premium ${pkgLabel}, authentic ${matLabel}, ${finLabel} finish, studio softbox illumination, clean background, sharp edges, 8k resolution.`;
        setPrompt(finalPrompt);
      }

      const payload = {
        input_asset_id: sketchAssetId,
        project_id: selectedProjectId || null,
        prompt: finalPrompt,
        negative_prompt: negativePrompt,
        package_type: packageType,
        material: material,
        finish: finish,
        lighting: lighting,
        viewpoint: viewpoint,
        background: background,
        steps: steps,
        guidance_scale: guidanceScale,
        controlnet_scale: controlnetScale,
        seed: seed || null,
        width: resolution,
        height: resolution,
      };

      const res = await api.createGeneration(payload);
      if (res.id) {
        setCurrentJobId(res.id);
        showNotification('Inference pipeline dispatched.', 'info');
        pollJobStatus(res.id);
      }
    } catch (err) {
      setIsGenerating(false);
      showNotification(`Generation dispatch failed: ${err.message}`, 'error');
    }
  };

  // Cancel generation
  const handleCancelGeneration = async () => {
    if (!currentJobId) return;
    try {
      await api.cancelGeneration(currentJobId);
      if (pollingRef.current) clearInterval(pollingRef.current);
      setIsGenerating(false);
      showNotification('Generation cancelled by user.', 'info');
    } catch (err) {
      showNotification(`Failed to cancel: ${err.message}`, 'error');
    }
  };

  const toggleFocusMode = () => {
    if (isFocusMode) {
      setIsLeftOpen(true);
      setIsRightOpen(true);
      setIsFocusMode(false);
    } else {
      setIsLeftOpen(false);
      setIsRightOpen(false);
      setIsFocusMode(true);
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden bg-[#0B0D12]">
      {/* =========================================================================
          TOP ACTION TOOLBAR: Navigation tabs, panel toggles & export buttons
         ========================================================================= */}
      <div className="h-14 border-b border-[#1E2330] bg-[#12151D] px-4 flex items-center justify-between shrink-0 gap-3">
        {/* Left Side: Left Panel Toggle & Breadcrumb */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsLeftOpen(!isLeftOpen)}
            className={`px-2.5 py-1.5 rounded-lg border text-xs font-medium flex items-center gap-1.5 transition-all ${
              isLeftOpen
                ? 'bg-[#171B25] border-[#242A38] text-white hover:border-[#8B5CF6]/50'
                : 'bg-[#8B5CF6]/20 border-[#8B5CF6]/40 text-[#A78BFA]'
            }`}
            title={isLeftOpen ? 'Collapse Sketch & Directives Panel' : 'Expand Sketch & Directives Panel'}
          >
            <PanelLeft className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">{isLeftOpen ? 'Hide Inputs' : 'Show Inputs'}</span>
          </button>

          <span className="text-xs font-semibold text-white font-mono hidden md:flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-[#8B5CF6]" />
            Studio Workspace
          </span>
        </div>

        {/* Center: Mode Tabs */}
        <div className="flex items-center bg-[#171B25] p-1 rounded-xl border border-[#242A38]">
          <button
            onClick={() => setActiveCenterTab('render')}
            className={`px-3 py-1 rounded-lg text-xs font-medium transition-all flex items-center gap-1.5 ${
              activeCenterTab === 'render'
                ? 'bg-[#8B5CF6] text-white shadow-sm'
                : 'text-[#94A3B8] hover:text-white'
            }`}
          >
            <Split className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Before / After</span> Render
          </button>
          <button
            onClick={() => setActiveCenterTab('maps')}
            className={`px-3 py-1 rounded-lg text-xs font-medium transition-all flex items-center gap-1.5 ${
              activeCenterTab === 'maps'
                ? 'bg-[#8B5CF6] text-white shadow-sm'
                : 'text-[#94A3B8] hover:text-white'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            Edge & Depth Maps
          </button>
          <button
            onClick={() => setActiveCenterTab('logo')}
            className={`px-3 py-1 rounded-lg text-xs font-medium transition-all flex items-center gap-1.5 ${
              activeCenterTab === 'logo'
                ? 'bg-[#8B5CF6] text-white shadow-sm'
                : 'text-[#94A3B8] hover:text-white'
            }`}
          >
            <Sliders className="w-3.5 h-3.5" />
            Logo Compositor
          </button>
        </div>

        {/* Right Action Tools */}
        <div className="flex items-center gap-2">
          {activeRenderAssetId && sketchAssetId && (
            <button
              onClick={() => openEvaluationModal(sketchAssetId, activeRenderAssetId, prompt)}
              className="px-2.5 py-1.5 rounded-lg text-xs font-medium bg-[#171B25] hover:bg-[#242A38] text-white border border-[#242A38] flex items-center gap-1.5 transition-all"
              title="Evaluate Edge SSIM & Structural Preservation"
            >
              <Award className="w-3.5 h-3.5 text-[#8B5CF6]" />
              <span className="hidden lg:inline">Evaluate SSIM</span>
            </button>
          )}

          {activeRenderResult && (
            <a
              href={activeRenderResult}
              download={`packvision_${packageType}.png`}
              className="px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-[#8B5CF6] hover:bg-[#7C3AED] text-white flex items-center gap-1.5 shadow-md shadow-[#8B5CF6]/20 transition-all"
              title="Download Render Image"
            >
              <Download className="w-3.5 h-3.5" />
              <span className="hidden lg:inline">Download</span>
            </a>
          )}

          {/* Full Canvas Focus Toggle */}
          <button
            onClick={toggleFocusMode}
            className={`p-1.5 rounded-lg border text-xs font-medium transition-all ${
              isFocusMode
                ? 'bg-[#8B5CF6] text-white border-[#8B5CF6]'
                : 'bg-[#171B25] border-[#242A38] text-[#94A3B8] hover:text-white'
            }`}
            title={isFocusMode ? 'Exit Full Canvas Focus' : 'Full Canvas Focus Mode'}
          >
            {isFocusMode ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </button>

          {/* Right Panel Toggle */}
          <button
            onClick={() => setIsRightOpen(!isRightOpen)}
            className={`px-2.5 py-1.5 rounded-lg border text-xs font-medium flex items-center gap-1.5 transition-all ${
              isRightOpen
                ? 'bg-[#171B25] border-[#242A38] text-white hover:border-[#8B5CF6]/50'
                : 'bg-[#8B5CF6]/20 border-[#8B5CF6]/40 text-[#A78BFA]'
            }`}
            title={isRightOpen ? 'Collapse Diffusion Settings' : 'Expand Diffusion Settings'}
          >
            <span className="hidden sm:inline">{isRightOpen ? 'Hide Settings' : 'Show Settings'}</span>
            <PanelRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* =========================================================================
          MAIN 3-PANEL STUDIO WORKSPACE (Collapsible Left, Canvas, Collapsible Right)
         ========================================================================= */}
      <div className="flex-1 flex overflow-hidden">
        {/* =========================================================================
            LEFT PANEL: Input Sketch, Taxonomy & Clean Collapsible Directives
           ========================================================================= */}
        <div
          className={`border-r border-[#1E2330] bg-[#12151D] overflow-y-auto shrink-0 transition-all duration-300 ${
            isLeftOpen ? 'w-80 lg:w-88 p-5 space-y-5' : 'w-0 p-0 border-r-0 overflow-hidden'
          }`}
        >
          {isLeftOpen && (
            <>
              {/* Sketch Upload Section */}
              <div className="space-y-2.5">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-semibold text-white uppercase tracking-wider flex items-center gap-1.5">
                    <Upload className="w-3.5 h-3.5 text-[#8B5CF6]" />
                    Input Sketch
                  </label>
                  <div className="flex gap-1.5">
                    <button
                      onClick={() => handleLoadSample('carton')}
                      className="text-[10px] text-[#A78BFA] hover:underline"
                    >
                      Carton
                    </button>
                    <span className="text-[10px] text-[#94A3B8]">•</span>
                    <button
                      onClick={() => handleLoadSample('bottle')}
                      className="text-[10px] text-[#A78BFA] hover:underline"
                    >
                      Bottle
                    </button>
                    <span className="text-[10px] text-[#94A3B8]">•</span>
                    <button
                      onClick={() => handleLoadSample('pouch')}
                      className="text-[10px] text-[#A78BFA] hover:underline"
                    >
                      Pouch
                    </button>
                  </div>
                </div>

                {/* Hidden Global / Reusable File Input */}
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={handleSketchUpload}
                />

                {sketchPreviewUrl ? (
                  <div className="space-y-2">
                    <div className="relative aspect-square rounded-xl bg-black border border-[#242A38] overflow-hidden group">
                      <img src={sketchPreviewUrl} alt="Uploaded sketch" className="w-full h-full object-contain" />
                      <button
                        onClick={handleClearSketch}
                        className="absolute top-2 right-2 p-1.5 rounded-lg bg-black/80 text-white hover:bg-rose-600 transition-colors shadow-md"
                        title="Remove sketch"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                    <button
                      type="button"
                      onClick={() => fileInputRef.current?.click()}
                      className="w-full py-2 px-3 rounded-xl border border-[#242A38] hover:border-[#8B5CF6]/50 bg-[#171B25] text-xs font-semibold text-white hover:text-[#A78BFA] flex items-center justify-center gap-2 transition-all shadow-sm"
                    >
                      <Upload className="w-3.5 h-3.5 text-[#8B5CF6]" />
                      Change / Replace Sketch
                    </button>
                  </div>
                ) : (
                  <div
                    onClick={() => fileInputRef.current?.click()}
                    onDragOver={(e) => {
                      e.preventDefault();
                      setIsDragOverLeft(true);
                    }}
                    onDragLeave={() => setIsDragOverLeft(false)}
                    onDrop={(e) => {
                      e.preventDefault();
                      setIsDragOverLeft(false);
                      const file = e.dataTransfer.files?.[0];
                      if (file) processSketchFile(file);
                    }}
                    className={`flex flex-col items-center justify-center aspect-[4/3] border-2 border-dashed rounded-xl p-4 cursor-pointer transition-all ${
                      isDragOverLeft
                        ? 'border-[#8B5CF6] bg-[#8B5CF6]/15 scale-[1.01]'
                        : 'border-[#242A38] hover:border-[#8B5CF6]/60 bg-[#171B25]/50'
                    }`}
                  >
                    <Upload className="w-7 h-7 text-[#8B5CF6] mb-2" />
                    <span className="text-xs font-semibold text-white">Upload Sketch Image</span>
                    <span className="text-[10px] text-[#94A3B8] mt-1">PNG, JPG, JPEG, WebP</span>
                    <span className="text-[10px] text-[#A78BFA] mt-1.5 font-medium">Click or Drag &amp; Drop</span>
                  </div>
                )}
              </div>

              {/* Core Packaging Taxonomy */}
              <div className="space-y-3.5 p-3.5 rounded-xl bg-[#171B25] border border-[#242A38]">
                <div>
                  <label className="block text-xs font-medium text-[#94A3B8] mb-1">Packaging Category</label>
                  <select
                    value={packageType}
                    onChange={(e) => setPackageType(e.target.value)}
                    className="w-full bg-[#12151D] border border-[#242A38] text-xs rounded-xl px-3 py-2 text-white focus:outline-none focus:border-[#8B5CF6]"
                  >
                    <option value="carton">Folding Carton / Box</option>
                    <option value="bottle">Product Bottle / Dropper</option>
                    <option value="pouch">Flexible Stand-up Pouch</option>
                    <option value="can">Beverage / Spray Can</option>
                    <option value="jar">Cosmetic / Cream Jar</option>
                    <option value="tube">Squeeze Dispenser Tube</option>
                    <option value="sleeve">Cardboard Sleeve / Outer</option>
                  </select>
                </div>

                <div>
                  <label className="block text-xs font-medium text-[#94A3B8] mb-1">Substrate Material</label>
                  <select
                    value={material}
                    onChange={(e) => setMaterial(e.target.value)}
                    className="w-full bg-[#12151D] border border-[#242A38] text-xs rounded-xl px-3 py-2 text-white focus:outline-none focus:border-[#8B5CF6]"
                  >
                    <option value="coated_sbs">Coated SBS Paperboard</option>
                    <option value="kraft">Unbleached Kraft Board</option>
                    <option value="rigid_grayboard">Luxury Rigid Grayboard</option>
                    <option value="clear_flint_glass">Ultra-Clear Flint Glass</option>
                    <option value="amber_glass">UV-Protective Amber Glass</option>
                    <option value="matte_ceramic">Ceramic Porcelain Stoneware</option>
                    <option value="brushed_aluminum">Brushed Aluminum Metal</option>
                    <option value="matte_mylar">Matte Barrier Mylar Foil</option>
                    <option value="translucent_plastic">Frosted Translucent HDPE</option>
                  </select>
                </div>

                <div>
                  <label className="block text-xs font-medium text-[#94A3B8] mb-1">Surface Finish</label>
                  <select
                    value={finish}
                    onChange={(e) => setFinish(e.target.value)}
                    className="w-full bg-[#12151D] border border-[#242A38] text-xs rounded-xl px-3 py-2 text-white focus:outline-none focus:border-[#8B5CF6]"
                  >
                    <option value="matte">Matte Smooth Coating</option>
                    <option value="soft_touch">Velvety Soft-Touch Lamination</option>
                    <option value="gloss_uv">High-Gloss UV Varnish</option>
                    <option value="spot_gloss">Spot High-Gloss Details</option>
                    <option value="metallic_foil">Metallic Hot-Foil Stamping</option>
                    <option value="holographic">Iridescent Holographic Sheen</option>
                  </select>
                </div>

                <div>
                  <label className="block text-xs font-medium text-[#94A3B8] mb-1">Save to Project</label>
                  <select
                    value={selectedProjectId}
                    onChange={(e) => setSelectedProjectId(e.target.value)}
                    className="w-full bg-[#12151D] border border-[#242A38] text-xs rounded-xl px-3 py-2 text-white focus:outline-none focus:border-[#8B5CF6]"
                  >
                    <option value="">No Project (Standalone History)</option>
                    {projectsList.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Collapsible Accordion: Lighting & Viewpoint */}
              <div className="rounded-xl bg-[#171B25] border border-[#242A38] overflow-hidden">
                <button
                  onClick={() => setShowLightingSection(!showLightingSection)}
                  className="w-full p-3 text-xs font-semibold text-white flex items-center justify-between hover:bg-[#242A38]/50 transition-colors"
                >
                  <span className="flex items-center gap-2 text-[#A78BFA]">
                    <Sun className="w-3.5 h-3.5" />
                    Studio Lighting & Angles
                  </span>
                  {showLightingSection ? <ChevronUp className="w-4 h-4 text-[#94A3B8]" /> : <ChevronDown className="w-4 h-4 text-[#94A3B8]" />}
                </button>

                {showLightingSection && (
                  <div className="p-3.5 border-t border-[#242A38] space-y-3">
                    <div>
                      <label className="block text-xs font-medium text-[#94A3B8] mb-1">Studio Lighting Rig</label>
                      <select
                        value={lighting}
                        onChange={(e) => setLighting(e.target.value)}
                        className="w-full bg-[#12151D] border border-[#242A38] text-xs rounded-xl px-3 py-2 text-white focus:outline-none focus:border-[#8B5CF6]"
                      >
                        <option value="studio_softbox">Balanced Three-Point Softbox</option>
                        <option value="dramatic_rim">Dramatic Specular Rim Light</option>
                        <option value="warm_golden">Warm Golden Hour Window Light</option>
                        <option value="cool_high_key">Clean Medical High-Key Lighting</option>
                        <option value="neon_accent">Cyberpunk Colored Neon Rim</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-[#94A3B8] mb-1">Background & Pedestal</label>
                      <select
                        value={background}
                        onChange={(e) => setBackground(e.target.value)}
                        className="w-full bg-[#12151D] border border-[#242A38] text-xs rounded-xl px-3 py-2 text-white focus:outline-none focus:border-[#8B5CF6]"
                      >
                        <option value="seamless_studio">Neutral Light Studio Table</option>
                        <option value="dark_vignette">Minimalist Dark Vignette Gradient</option>
                        <option value="marble_pedestal">Luxury Polished Marble Podium</option>
                        <option value="botanical_organic">Nature Wooden Countertop</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-[#94A3B8] mb-1">Camera Angle</label>
                      <select
                        value={viewpoint}
                        onChange={(e) => setViewpoint(e.target.value)}
                        className="w-full bg-[#12151D] border border-[#242A38] text-xs rounded-xl px-3 py-2 text-white focus:outline-none focus:border-[#8B5CF6]"
                      >
                        <option value="three_quarter">Three-Quarter Hero Perspective</option>
                        <option value="front_elevation">Direct Front Orthographic Elevation</option>
                        <option value="top_down">Overhead Flat-Lay Composition</option>
                      </select>
                    </div>
                  </div>
                )}
              </div>

              {/* Collapsible Accordion: Custom Prompt Directives */}
              <div className="rounded-xl bg-[#171B25] border border-[#242A38] overflow-hidden">
                <button
                  onClick={() => setShowPromptsSection(!showPromptsSection)}
                  className="w-full p-3 text-xs font-semibold text-white flex items-center justify-between hover:bg-[#242A38]/50 transition-colors"
                >
                  <span className="flex items-center gap-2 text-[#A78BFA]">
                    <FileText className="w-3.5 h-3.5" />
                    Custom Directives & Prompts
                  </span>
                  {showPromptsSection ? <ChevronUp className="w-4 h-4 text-[#94A3B8]" /> : <ChevronDown className="w-4 h-4 text-[#94A3B8]" />}
                </button>

                {showPromptsSection && (
                  <div className="p-3.5 border-t border-[#242A38] space-y-3">
                    <div className="flex items-center justify-between">
                      <label className="text-[11px] font-medium text-[#94A3B8]">Positive Prompt</label>
                      <button
                        onClick={generatePromptFromSpecs}
                        className="text-[10px] text-[#8B5CF6] hover:text-[#A78BFA] flex items-center gap-1 font-semibold"
                      >
                        <Sparkles className="w-3 h-3" /> Auto-Synthesize
                      </button>
                    </div>
                    <textarea
                      rows={3}
                      value={prompt}
                      onChange={(e) => setPrompt(e.target.value)}
                      placeholder="e.g. Photorealistic luxury folding carton, matte black cardboard, gold foil typography..."
                      className="w-full bg-[#12151D] border border-[#242A38] text-xs rounded-xl p-3 text-white focus:outline-none focus:border-[#8B5CF6] resize-none"
                    />

                    <div>
                      <label className="block text-[11px] font-medium text-[#94A3B8] mb-1">Negative Directives (Avoid)</label>
                      <textarea
                        rows={2}
                        value={negativePrompt}
                        onChange={(e) => setNegativePrompt(e.target.value)}
                        className="w-full bg-[#12151D] border border-[#242A38] text-xs rounded-xl p-3 text-[#94A3B8] focus:outline-none focus:border-[#8B5CF6] resize-none"
                      />
                    </div>
                  </div>
                )}
              </div>
            </>
          )}
        </div>

        {/* =========================================================================
            CENTER PANEL: Visual Studio Canvas (Render / Maps / Logo Compositor)
           ========================================================================= */}
        <div className="flex-1 bg-[#0B0D12] p-4 lg:p-6 overflow-y-auto flex flex-col items-center justify-center">
          {activeCenterTab === 'render' && (
            <div className="w-full max-w-4xl space-y-4">
              <BeforeAfterSlider
                beforeImage={preprocessedMaps?.maps?.canny_url || sketchPreviewUrl}
                afterImage={activeRenderResult}
                beforeLabel="Input Sketch Structure"
                afterLabel="Photorealistic Packaging Render"
                onUploadClick={() => {
                  setIsLeftOpen(true);
                  fileInputRef.current?.click();
                }}
                onFileDrop={(file) => {
                  setIsLeftOpen(true);
                  processSketchFile(file);
                }}
              />

              {/* Render Metadata Card if completed */}
              {generationMeta && (
                <div className="p-4 rounded-xl bg-[#171B25] border border-[#242A38] text-xs flex flex-wrap items-center justify-between gap-4">
                  <div>
                    <span className="text-[11px] text-[#94A3B8] block">Inference Engine</span>
                    <span className="font-semibold text-white mt-0.5 block">{generationMeta.model_name}</span>
                  </div>
                  <div>
                    <span className="text-[11px] text-[#94A3B8] block">Seed</span>
                    <span className="font-mono text-white mt-0.5 block">{generationMeta.seed}</span>
                  </div>
                  <div>
                    <span className="text-[11px] text-[#94A3B8] block">Steps & Guidance</span>
                    <span className="font-mono text-white mt-0.5 block">
                      {steps} steps • scale {guidanceScale}
                    </span>
                  </div>
                  <div>
                    <span className="text-[11px] text-[#94A3B8] block">Status</span>
                    <span className="text-emerald-400 font-semibold mt-0.5 block capitalize">
                      {generationMeta.status}
                    </span>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeCenterTab === 'maps' && (
            <div className="w-full max-w-4xl space-y-6">
              {/* Responsive, Un-cramped Controls Bar */}
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 bg-[#171B25] p-4 rounded-xl border border-[#242A38]">
                <div className="flex flex-wrap items-center gap-6 w-full sm:w-auto">
                  <div className="flex-1 sm:flex-initial min-w-[140px]">
                    <div className="flex justify-between text-xs text-[#94A3B8] mb-1">
                      <span>Low Threshold</span>
                      <span className="font-mono text-white">{cannyLow}</span>
                    </div>
                    <input
                      type="range"
                      min="10"
                      max="150"
                      value={cannyLow}
                      onChange={(e) => setCannyLow(parseInt(e.target.value, 10))}
                      className="w-full accent-[#8B5CF6]"
                    />
                  </div>
                  <div className="flex-1 sm:flex-initial min-w-[140px]">
                    <div className="flex justify-between text-xs text-[#94A3B8] mb-1">
                      <span>High Threshold</span>
                      <span className="font-mono text-white">{cannyHigh}</span>
                    </div>
                    <input
                      type="range"
                      min="50"
                      max="255"
                      value={cannyHigh}
                      onChange={(e) => setCannyHigh(parseInt(e.target.value, 10))}
                      className="w-full accent-[#8B5CF6]"
                    />
                  </div>
                </div>
                <button
                  onClick={() => triggerPreprocessing()}
                  disabled={isPreprocessing || !sketchAssetId}
                  className="w-full sm:w-auto shrink-0 px-4 py-2.5 rounded-lg text-xs font-semibold bg-[#8B5CF6] hover:bg-[#7C3AED] disabled:opacity-50 text-white flex items-center justify-center gap-1.5 transition-all shadow-md"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${isPreprocessing ? 'animate-spin' : ''}`} />
                  Recompute Maps
                </button>
              </div>

              {preprocessedMaps ? (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="rounded-xl overflow-hidden bg-black border border-[#242A38] p-2 text-center space-y-2">
                    <img src={preprocessedMaps.maps.original_url} alt="Original normalized" className="aspect-square object-contain mx-auto" />
                    <span className="text-xs font-medium text-white block">1. Normalized Sketch</span>
                  </div>
                  <div className="rounded-xl overflow-hidden bg-black border border-[#242A38] p-2 text-center space-y-2">
                    <img src={preprocessedMaps.maps.canny_url} alt="Canny edge map" className="aspect-square object-contain mx-auto" />
                    <span className="text-xs font-medium text-white block">2. Canny Edge Map</span>
                  </div>
                  <div className="rounded-xl overflow-hidden bg-black border border-[#242A38] p-2 text-center space-y-2">
                    <img src={preprocessedMaps.maps.lineart_url} alt="Lineart map" className="aspect-square object-contain mx-auto" />
                    <span className="text-xs font-medium text-white block">3. Line-Art Inversion</span>
                  </div>
                  <div className="rounded-xl overflow-hidden bg-black border border-[#242A38] p-2 text-center space-y-2">
                    <img src={preprocessedMaps.maps.depth_url} alt="Pseudo-depth map" className="aspect-square object-contain mx-auto" />
                    <span className="text-xs font-medium text-white block">4. Pseudo-Depth Map</span>
                  </div>
                </div>
              ) : (
                <div className="p-12 text-center text-xs text-[#94A3B8] bg-[#171B25] rounded-xl border border-[#242A38]">
                  Upload a sketch to view edge and depth maps.
                </div>
              )}
            </div>
          )}

          {activeCenterTab === 'logo' && (
            <div className="w-full">
              <LogoEditor
                baseAssetId={rawRenderAssetId || activeRenderAssetId}
                baseImageUrl={rawRenderResult || activeRenderResult}
                logoAssetId={logoAssetId}
                logoImageUrl={logoPreviewUrl}
                onCompositeCreated={(res) => {
                  setActiveRenderResult(res.url);
                  setActiveRenderAssetId(res.asset_id);
                }}
                showNotification={showNotification}
              />
            </div>
          )}
        </div>

        {/* =========================================================================
            RIGHT PANEL: Generation Settings, Hardware Status & Trigger Control
           ========================================================================= */}
        <div
          className={`border-l border-[#1E2330] bg-[#12151D] overflow-y-auto shrink-0 transition-all duration-300 flex flex-col justify-between ${
            isRightOpen ? 'w-72 lg:w-80 p-5 space-y-5' : 'w-0 p-0 border-l-0 overflow-hidden'
          }`}
        >
          {isRightOpen && (
            <>
              <div className="space-y-5">
                <div className="flex items-center justify-between border-b border-[#1E2330] pb-3">
                  <h2 className="text-xs font-semibold text-white uppercase tracking-wider flex items-center gap-1.5">
                    <Sliders className="w-3.5 h-3.5 text-[#8B5CF6]" />
                    Diffusion Controls
                  </h2>
                </div>

                {/* Primary Action Button (Top) */}
                <div>
                  {isGenerating ? (
                    <button
                      onClick={handleCancelGeneration}
                      className="w-full py-3 px-4 rounded-xl text-xs font-semibold bg-rose-600 hover:bg-rose-700 text-white flex items-center justify-center gap-2 shadow-lg shadow-rose-600/20 transition-all"
                    >
                      <StopCircle className="w-4 h-4 animate-pulse" />
                      Cancel Rendering
                    </button>
                  ) : (
                    <button
                      onClick={handleGenerate}
                      disabled={isGenerating || !sketchAssetId}
                      className="w-full py-3.5 px-4 rounded-xl text-xs font-bold bg-[#8B5CF6] hover:bg-[#7C3AED] disabled:opacity-40 text-white flex items-center justify-center gap-2 shadow-lg shadow-[#8B5CF6]/25 transition-all"
                    >
                      <Sparkles className="w-4 h-4" />
                      Generate Packaging Render
                    </button>
                  )}
                </div>

                {/* Core Slider 1: Inference Steps */}
                <div className="p-3.5 rounded-xl bg-[#171B25] border border-[#242A38]">
                  <div className="flex justify-between text-xs text-white mb-1.5">
                    <span className="font-medium">Inference Steps</span>
                    <span className="font-mono text-[#A78BFA] font-semibold">{steps}</span>
                  </div>
                  <input
                    type="range"
                    min="15"
                    max="50"
                    value={steps}
                    onChange={(e) => setSteps(parseInt(e.target.value, 10))}
                    className="w-full accent-[#8B5CF6]"
                  />
                  <p className="text-[10px] text-[#94A3B8] mt-1">Recommended: 25-30 steps on GTX 1650 Ti / Colab</p>
                </div>

                {/* Core Slider 2: ControlNet Silhouette Edge Strength */}
                <div className="p-3.5 rounded-xl bg-[#171B25] border border-[#242A38]">
                  <div className="flex justify-between text-xs text-white mb-1.5">
                    <span className="font-medium">Sketch Edge Lock</span>
                    <span className="font-mono text-[#A78BFA] font-semibold">{controlnetScale}</span>
                  </div>
                  <input
                    type="range"
                    min="0.3"
                    max="1.5"
                    step="0.05"
                    value={controlnetScale}
                    onChange={(e) => setControlnetScale(parseFloat(e.target.value))}
                    className="w-full accent-[#8B5CF6]"
                  />
                  <p className="text-[10px] text-[#94A3B8] mt-1">Higher values strictly enforce sketch contour lines</p>
                </div>

                {/* Collapsible Accordion: Advanced Diffusion Parameters */}
                <div className="rounded-xl bg-[#171B25] border border-[#242A38] overflow-hidden">
                  <button
                    onClick={() => setShowAdvancedDiffusion(!showAdvancedDiffusion)}
                    className="w-full p-3 text-xs font-semibold text-white flex items-center justify-between hover:bg-[#242A38]/50 transition-colors"
                  >
                    <span className="flex items-center gap-2 text-[#A78BFA]">
                      <SlidersHorizontal className="w-3.5 h-3.5" />
                      Advanced Diffusion
                    </span>
                    {showAdvancedDiffusion ? <ChevronUp className="w-4 h-4 text-[#94A3B8]" /> : <ChevronDown className="w-4 h-4 text-[#94A3B8]" />}
                  </button>

                  {showAdvancedDiffusion && (
                    <div className="p-3.5 border-t border-[#242A38] space-y-4">
                      {/* Guidance Scale */}
                      <div>
                        <div className="flex justify-between text-xs text-[#94A3B8] mb-1">
                          <span>Guidance Scale (CFG)</span>
                          <span className="font-mono text-white">{guidanceScale}</span>
                        </div>
                        <input
                          type="range"
                          min="1.0"
                          max="15.0"
                          step="0.5"
                          value={guidanceScale}
                          onChange={(e) => setGuidanceScale(parseFloat(e.target.value))}
                          className="w-full accent-[#8B5CF6]"
                        />
                      </div>

                      {/* Random Seed */}
                      <div>
                        <div className="flex justify-between text-xs text-[#94A3B8] mb-1">
                          <span>Seed</span>
                          <button
                            onClick={() => setSeed(Math.floor(Math.random() * 999999))}
                            className="text-[10px] text-[#A78BFA] hover:underline"
                          >
                            Randomize
                          </button>
                        </div>
                        <input
                          type="number"
                          placeholder="Random (Leave empty)"
                          value={seed ?? ''}
                          onChange={(e) => setSeed(e.target.value ? parseInt(e.target.value, 10) : null)}
                          className="w-full bg-[#12151D] border border-[#242A38] text-xs rounded-xl px-3 py-2 text-white font-mono focus:outline-none focus:border-[#8B5CF6]"
                        />
                      </div>

                      {/* Resolution Selector */}
                      <div>
                        <label className="block text-xs font-medium text-[#94A3B8] mb-1.5">Output Resolution</label>
                        <div className="grid grid-cols-3 gap-2">
                          {[512, 768, 1024].map((res) => (
                            <button
                              key={res}
                              type="button"
                              onClick={() => setResolution(res)}
                              className={`py-1.5 rounded-lg text-xs font-semibold transition-all ${
                                resolution === res
                                  ? 'bg-[#8B5CF6] text-white shadow-sm'
                                  : 'bg-[#12151D] text-[#94A3B8] hover:text-white border border-[#242A38]'
                              }`}
                            >
                              {res}px
                            </button>
                          ))}
                        </div>
                        <p className="text-[10px] text-[#94A3B8] mt-1">512px optimized for 4GB VRAM & fast Colab inference</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Live Generation Progress Card */}
              {isGenerating && (
                <div className="p-4 rounded-xl bg-[#171B25] border border-[#8B5CF6]/40 space-y-2 mt-4">
                  <div className="flex justify-between text-xs font-medium">
                    <span className="text-white flex items-center gap-1.5">
                      <Sparkles className="w-3.5 h-3.5 text-[#8B5CF6] animate-spin" />
                      Inference Running
                    </span>
                    <span className="text-[#A78BFA] font-mono">{jobProgress}%</span>
                  </div>
                  <div className="w-full bg-[#12151D] rounded-full h-2 overflow-hidden border border-[#242A38]">
                    <div
                      className="bg-gradient-to-r from-[#8B5CF6] to-[#A78BFA] h-2 rounded-full transition-all duration-300"
                      style={{ width: `${jobProgress}%` }}
                    />
                  </div>
                  <p className="text-[10px] text-[#94A3B8] truncate">{jobStageMessage}</p>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
