import React, { useState, useRef } from 'react';
import {
  Upload,
  Sliders,
  RotateCw,
  Eye,
  Sparkles,
  Download,
  Check,
  AlertCircle,
  Move,
  Maximize2,
  Blend,
  CornerDownRight,
  Target,
} from 'lucide-react';
import { api } from '../services/api';

export default function LogoEditor({
  baseAssetId,
  baseImageUrl,
  logoAssetId: initialLogoAssetId,
  logoImageUrl: initialLogoImageUrl,
  onCompositeCreated,
  showNotification,
}) {
  const [logoAssetId, setLogoAssetId] = useState(initialLogoAssetId || null);
  const [logoImageUrl, setLogoImageUrl] = useState(initialLogoImageUrl || null);
  const [posX, setPosX] = useState(0.5);
  const [posY, setPosY] = useState(0.5);
  const [scale, setScale] = useState(0.35);
  const [rotation, setRotation] = useState(0);
  const [opacity, setOpacity] = useState(1.0);
  const [blendMode, setBlendMode] = useState('multiply');
  const [useCielab, setUseCielab] = useState(true);
  const [isProcessing, setIsProcessing] = useState(false);
  const [compositeResultUrl, setCompositeResultUrl] = useState(null);

  const canvasRef = useRef(null);

  const processLogoFile = async (file) => {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      showNotification('Please select a valid image file (PNG, SVG, JPG).', 'error');
      return;
    }

    try {
      showNotification('Uploading brand logo asset...', 'info');
      const res = await api.uploadAsset(file, 'logo');
      if (res.success && res.asset) {
        setLogoAssetId(res.asset.id);
        setLogoImageUrl(res.asset.url);
        setCompositeResultUrl(null);
        showNotification('Logo uploaded successfully!', 'success');
      }
    } catch (err) {
      showNotification(`Logo upload failed: ${err.message}`, 'error');
    }
  };

  const handleLogoUpload = async (e) => {
    const file = e.target.files?.[0];
    if (file) {
      await processLogoFile(file);
    }
    e.target.value = '';
  };

  const handleApplyComposite = async () => {
    if (!baseAssetId) {
      showNotification('Please generate or select a render first.', 'error');
      return;
    }
    if (!logoAssetId) {
      showNotification('Please upload a brand logo first.', 'error');
      return;
    }

    try {
      setIsProcessing(true);
      showNotification('Compositing logo with precision warping & texture blending...', 'info');

      const res = await api.createComposite({
        base_asset_id: baseAssetId,
        logo_asset_id: logoAssetId,
        x: posX,
        y: posY,
        scale: scale,
        rotation: rotation,
        opacity: opacity,
        blend_mode: blendMode,
        use_cielab_texture: useCielab,
        save_as_asset: true,
      });

      if (res.success) {
        setCompositeResultUrl(res.url);
        showNotification('Composite created and saved successfully!', 'success');
        if (onCompositeCreated) {
          onCompositeCreated(res);
        }
      }
    } catch (err) {
      showNotification(`Compositing failed: ${err.message}`, 'error');
    } finally {
      setIsProcessing(false);
    }
  };

  // Direct Click-on-Canvas Positioning
  const handleCanvasClick = (e) => {
    if (!canvasRef.current || !baseImageUrl || compositeResultUrl) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const clickX = Math.max(0.05, Math.min(0.95, (e.clientX - rect.left) / rect.width));
    const clickY = Math.max(0.05, Math.min(0.95, (e.clientY - rect.top) / rect.height));
    setPosX(Math.round(clickX * 100) / 100);
    setPosY(Math.round(clickY * 100) / 100);
    setCompositeResultUrl(null);
  };

  return (
    <div className="w-full max-w-6xl mx-auto flex flex-col xl:flex-row items-start justify-center gap-6">
      {/* =====================================================================
          LEFT: Visual Interactive Canvas (Sticky / Fits on Screen Without Scroll)
         ===================================================================== */}
      <div className="w-full xl:w-[460px] shrink-0 flex flex-col items-center gap-3">
        {/* Canvas Square Frame */}
        <div
          ref={canvasRef}
          onClick={handleCanvasClick}
          className={`relative w-full aspect-square max-w-[440px] max-h-[440px] bg-black rounded-2xl border border-[#242A38] overflow-hidden flex items-center justify-center shadow-2xl transition-all ${
            !compositeResultUrl && baseImageUrl && logoImageUrl ? 'cursor-crosshair' : 'cursor-default'
          }`}
          title={!compositeResultUrl && logoImageUrl ? 'Click anywhere to reposition logo' : ''}
        >
          {baseImageUrl ? (
            <img
              src={compositeResultUrl || baseImageUrl}
              alt="Base Packaging Render"
              className="w-full h-full object-contain pointer-events-none select-none"
            />
          ) : (
            <div className="text-[#94A3B8] text-sm flex flex-col items-center p-8 text-center">
              <AlertCircle className="w-10 h-10 mb-2 opacity-40 text-[#8B5CF6]" />
              <span className="font-medium text-white">No render available in active workspace</span>
              <span className="text-xs text-[#94A3B8] mt-1">Generate a packaging render first, then apply your logo here</span>
            </div>
          )}

          {/* Live Logo Placement Overlay */}
          {logoImageUrl && !compositeResultUrl && baseImageUrl && (
            <div
              className="absolute pointer-events-none transition-all duration-75"
              style={{
                left: `${posX * 100}%`,
                top: `${posY * 100}%`,
                transform: `translate(-50%, -50%) rotate(${rotation}deg)`,
                width: `${scale * 100}%`,
                opacity: opacity,
                mixBlendMode: blendMode === 'multiply' ? 'multiply' : blendMode === 'overlay' ? 'overlay' : 'normal',
              }}
            >
              <img
                src={logoImageUrl}
                alt="Logo overlay"
                className="w-full h-auto object-contain filter drop-shadow-md border border-[#8B5CF6] rounded border-dashed"
              />
            </div>
          )}

          {/* Top Badges */}
          <div className="absolute top-3 left-3 pointer-events-none">
            <span className="px-2.5 py-1 rounded-lg text-[11px] font-semibold bg-[#171B25]/90 text-white border border-[#242A38] backdrop-blur-md shadow-md">
              {compositeResultUrl ? '✓ Final Composited Output' : 'Live Interactive Placement'}
            </span>
          </div>

          {compositeResultUrl && (
            <div className="absolute top-3 right-3 flex gap-2">
              <a
                href={compositeResultUrl}
                download="packvision_composite.png"
                className="px-2.5 py-1 rounded-lg text-xs font-semibold bg-[#8B5CF6] hover:bg-[#7C3AED] text-white flex items-center gap-1 shadow-lg shadow-[#8B5CF6]/30 transition-all"
              >
                <Download className="w-3.5 h-3.5" />
                Download
              </a>
              <button
                onClick={() => setCompositeResultUrl(null)}
                className="px-2.5 py-1 rounded-lg text-xs font-medium bg-[#171B25]/90 text-[#94A3B8] hover:text-white border border-[#242A38] backdrop-blur-md transition-all"
              >
                Re-adjust
              </button>
            </div>
          )}
        </div>

        {/* Quick Position Presets Bar */}
        <div className="w-full max-w-[440px] flex items-center justify-between gap-1 text-[11px] bg-[#171B25] p-2 rounded-xl border border-[#242A38]">
          <span className="text-[#94A3B8] flex items-center gap-1 font-medium pl-1">
            <Target className="w-3.5 h-3.5 text-[#8B5CF6]" /> Presets:
          </span>
          <div className="flex items-center gap-1">
            <button
              onClick={() => { setPosX(0.5); setPosY(0.25); setCompositeResultUrl(null); }}
              className="px-2 py-1 rounded-lg bg-[#12151D] hover:bg-[#8B5CF6] text-white font-medium transition-colors"
            >
              Top (25%)
            </button>
            <button
              onClick={() => { setPosX(0.5); setPosY(0.50); setCompositeResultUrl(null); }}
              className="px-2 py-1 rounded-lg bg-[#12151D] hover:bg-[#8B5CF6] text-white font-medium transition-colors"
            >
              Center (50%)
            </button>
            <button
              onClick={() => { setPosX(0.5); setPosY(0.75); setCompositeResultUrl(null); }}
              className="px-2 py-1 rounded-lg bg-[#12151D] hover:bg-[#8B5CF6] text-white font-medium transition-colors"
            >
              Bottom (75%)
            </button>
          </div>
        </div>
      </div>

      {/* =====================================================================
          RIGHT: Structured Control Deck (Side-by-Side with Canvas)
         ===================================================================== */}
      <div className="flex-1 w-full bg-[#171B25] border border-[#242A38] rounded-2xl p-5 space-y-4 shadow-xl">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#242A38] pb-3">
          <div className="flex items-center gap-2">
            <Sliders className="w-4 h-4 text-[#8B5CF6]" />
            <h3 className="font-semibold text-sm text-white">Brand Logo & Label Compositor</h3>
          </div>
          <span className="text-xs text-[#94A3B8]">
            CIELAB photometric texture blending & perspective alignment
          </span>
        </div>

        {/* 2x2 Controls Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
          {/* 1. Brand Logo Asset */}
          <div className="p-3.5 rounded-xl bg-[#12151D] border border-[#242A38] flex flex-col justify-between">
            <span className="font-semibold text-white flex items-center gap-1.5 text-[11px] uppercase tracking-wider text-[#A78BFA] mb-2">
              <Upload className="w-3.5 h-3.5" /> Brand Logo File
            </span>
            <label className="flex-1 flex flex-col items-center justify-center border-2 border-dashed border-[#242A38] hover:border-[#8B5CF6]/60 rounded-xl p-3 cursor-pointer bg-[#171B25] transition-colors min-h-[72px]">
              <Upload className="w-4 h-4 text-[#8B5CF6] mb-1" />
              <span className="text-xs text-white font-medium truncate max-w-[200px]">
                {logoImageUrl ? 'Change Logo File' : 'Upload PNG / SVG Logo'}
              </span>
              <input type="file" accept="image/*" className="hidden" onChange={handleLogoUpload} />
            </label>
          </div>

          {/* 2. Position Controls (X and Y Sliders) */}
          <div className="space-y-3 p-3.5 rounded-xl bg-[#12151D] border border-[#242A38]">
            <span className="font-semibold text-white flex items-center gap-1.5 text-[11px] uppercase tracking-wider text-[#A78BFA]">
              <Move className="w-3.5 h-3.5" /> Position
            </span>
            <div>
              <div className="flex justify-between text-[#94A3B8] mb-1">
                <span>Horizontal (X)</span>
                <span className="font-mono text-white font-semibold">{Math.round(posX * 100)}%</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="1.0"
                step="0.01"
                value={posX}
                onChange={(e) => {
                  setPosX(parseFloat(e.target.value));
                  setCompositeResultUrl(null);
                }}
                className="w-full accent-[#8B5CF6] cursor-pointer"
              />
            </div>
            <div>
              <div className="flex justify-between text-[#94A3B8] mb-1">
                <span className="text-white font-semibold">Vertical (Y)</span>
                <span className="font-mono text-[#A78BFA] font-bold">{Math.round(posY * 100)}%</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="1.0"
                step="0.01"
                value={posY}
                onChange={(e) => {
                  setPosY(parseFloat(e.target.value));
                  setCompositeResultUrl(null);
                }}
                className="w-full accent-[#8B5CF6] cursor-pointer"
              />
            </div>
          </div>

          {/* 3. Scale & Rotation */}
          <div className="space-y-3 p-3.5 rounded-xl bg-[#12151D] border border-[#242A38]">
            <span className="font-semibold text-white flex items-center gap-1.5 text-[11px] uppercase tracking-wider text-[#A78BFA]">
              <Maximize2 className="w-3.5 h-3.5" /> Geometry
            </span>
            <div>
              <div className="flex justify-between text-[#94A3B8] mb-1">
                <span>Scale Size</span>
                <span className="font-mono text-white font-semibold">{Math.round(scale * 100)}%</span>
              </div>
              <input
                type="range"
                min="0.05"
                max="0.80"
                step="0.01"
                value={scale}
                onChange={(e) => {
                  setScale(parseFloat(e.target.value));
                  setCompositeResultUrl(null);
                }}
                className="w-full accent-[#8B5CF6] cursor-pointer"
              />
            </div>
            <div>
              <div className="flex justify-between text-[#94A3B8] mb-1">
                <span>Rotation</span>
                <span className="font-mono text-white font-semibold">{rotation}°</span>
              </div>
              <input
                type="range"
                min="-180"
                max="180"
                step="1"
                value={rotation}
                onChange={(e) => {
                  setRotation(parseInt(e.target.value, 10));
                  setCompositeResultUrl(null);
                }}
                className="w-full accent-[#8B5CF6] cursor-pointer"
              />
            </div>
          </div>

          {/* 4. Appearance & Blend Modes */}
          <div className="space-y-3 p-3.5 rounded-xl bg-[#12151D] border border-[#242A38]">
            <span className="font-semibold text-white flex items-center gap-1.5 text-[11px] uppercase tracking-wider text-[#A78BFA]">
              <Blend className="w-3.5 h-3.5" /> Appearance
            </span>
            <div>
              <div className="flex justify-between text-[#94A3B8] mb-1">
                <span>Ink Blend Mode</span>
              </div>
              <select
                value={blendMode}
                onChange={(e) => {
                  setBlendMode(e.target.value);
                  setCompositeResultUrl(null);
                }}
                className="w-full bg-[#171B25] border border-[#242A38] rounded-lg px-2.5 py-1.5 text-white focus:outline-none focus:border-[#8B5CF6]"
              >
                <option value="multiply">Multiply (Printed Ink)</option>
                <option value="normal">Normal (Direct Stamped)</option>
                <option value="overlay">Overlay (Emboss Shading)</option>
                <option value="screen">Screen (Metallic Foil Glow)</option>
              </select>
            </div>
            <div>
              <div className="flex justify-between text-[#94A3B8] mb-1">
                <span>Opacity</span>
                <span className="font-mono text-white font-semibold">{Math.round(opacity * 100)}%</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.05"
                value={opacity}
                onChange={(e) => {
                  setOpacity(parseFloat(e.target.value));
                  setCompositeResultUrl(null);
                }}
                className="w-full accent-[#8B5CF6] cursor-pointer"
              />
            </div>
          </div>
        </div>

        {/* Apply Action Bar */}
        <div className="pt-2 flex flex-wrap items-center justify-between gap-3 border-t border-[#242A38]">
          <span className="text-[11px] text-[#94A3B8]">
            Click directly on the packaging canvas to position the logo anywhere.
          </span>
          <button
            onClick={handleApplyComposite}
            disabled={isProcessing || !logoAssetId || !baseAssetId}
            className="w-full sm:w-auto px-6 py-3 rounded-xl text-xs font-semibold bg-[#8B5CF6] hover:bg-[#7C3AED] disabled:opacity-40 text-white flex items-center justify-center gap-2 shadow-lg shadow-[#8B5CF6]/20 transition-all cursor-pointer"
          >
            <Sparkles className={`w-4 h-4 ${isProcessing ? 'animate-spin' : ''}`} />
            {isProcessing ? 'Applying Photometric Blend...' : 'Apply Brand Logo to Render'}
          </button>
        </div>
      </div>
    </div>
  );
}
