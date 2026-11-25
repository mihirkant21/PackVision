import React, { useState, useRef, useEffect } from 'react';
import { Eye, Split, Sparkles, Layers } from 'lucide-react';

export default function BeforeAfterSlider({
  beforeImage,
  afterImage,
  beforeLabel = 'Original Sketch / Edge Map',
  afterLabel = 'AI Photoreal Render',
}) {
  const [sliderPos, setSliderPos] = useState(50);
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef(null);

  const handleMove = (clientX) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    const percent = (x / rect.width) * 100;
    setSliderPos(percent);
  };

  const handleTouchMove = (e) => {
    if (!isDragging) return;
    handleMove(e.touches[0].clientX);
  };

  const handleMouseMove = (e) => {
    if (!isDragging) return;
    handleMove(e.clientX);
  };

  const handleMouseUp = () => setIsDragging(false);

  useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      window.addEventListener('touchmove', handleTouchMove);
      window.addEventListener('touchend', handleMouseUp);
    }
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('touchmove', handleTouchMove);
      window.removeEventListener('touchend', handleMouseUp);
    };
  }, [isDragging]);

  if (!beforeImage && !afterImage) {
    return (
      <div className="w-full h-full min-h-[420px] flex flex-col items-center justify-center text-[#94A3B8] bg-[#12151D] rounded-xl border border-[#242A38] p-8">
        <Layers className="w-12 h-12 text-[#8B5CF6] opacity-50 mb-3" />
        <p className="font-medium text-sm">Upload a sketch and generate a render to view comparison</p>
      </div>
    );
  }

  // If only afterImage is available (packaging render generated)
  if (!beforeImage && afterImage) {
    return (
      <div className="relative w-full h-[520px] max-h-[75vh] select-none overflow-hidden rounded-xl bg-black border border-[#242A38] shadow-2xl flex items-center justify-center">
        <img
          src={afterImage}
          alt={afterLabel}
          className="w-full h-full object-contain"
        />
        <div className="absolute top-4 left-4 pointer-events-none">
          <span className="px-3 py-1 rounded-md text-xs font-semibold bg-[#171B25]/90 text-white border border-[#242A38] shadow-md backdrop-blur-sm">
            {afterLabel}
          </span>
        </div>
      </div>
    );
  }

  // If only beforeImage is available (sketch uploaded, awaiting render)
  if (beforeImage && !afterImage) {
    return (
      <div className="relative w-full h-[520px] max-h-[75vh] select-none overflow-hidden rounded-xl bg-black border border-[#242A38] shadow-2xl flex items-center justify-center">
        <img
          src={beforeImage}
          alt={beforeLabel}
          className="w-full h-full object-contain"
        />
        <div className="absolute top-4 left-4 pointer-events-none">
          <span className="px-3 py-1 rounded-md text-xs font-semibold bg-[#171B25]/90 text-white border border-[#242A38] shadow-md backdrop-blur-sm">
            {beforeLabel}
          </span>
        </div>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="relative w-full h-[520px] max-h-[75vh] select-none overflow-hidden rounded-xl bg-black border border-dark-border shadow-2xl cursor-ew-resize"
      onMouseDown={() => setIsDragging(true)}
      onTouchStart={() => setIsDragging(true)}
    >
      {/* After image (background, full width) */}
      <img
        src={afterImage}
        alt={afterLabel}
        className="absolute inset-0 w-full h-full object-contain pointer-events-none"
      />

      {/* Before image (clipped by sliderPos) */}
      <div
        className="absolute inset-0 overflow-hidden"
        style={{ width: `${sliderPos}%` }}
      >
        <img
          src={beforeImage}
          alt={beforeLabel}
          className="absolute inset-0 w-full h-full object-contain pointer-events-none max-w-none"
          style={{ width: containerRef.current ? `${containerRef.current.offsetWidth}px` : '100%' }}
        />
      </div>

      {/* Divider line & handle */}
      <div
        className="absolute top-0 bottom-0 w-0.5 bg-brand-primary shadow-[0_0_12px_#8B5CF6] pointer-events-none"
        style={{ left: `${sliderPos}%` }}
      >
        <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-brand-primary text-white flex items-center justify-center shadow-lg border-2 border-white pointer-events-auto cursor-grab active:cursor-grabbing">
          <Split className="w-4 h-4 rotate-90" />
        </div>
      </div>

      {/* Labels */}
      <div className="absolute top-4 left-4 pointer-events-none">
        <span className="px-3 py-1 rounded-md text-xs font-semibold bg-dark-card/90 text-text-primary border border-dark-border shadow-md backdrop-blur-sm">
          {beforeLabel}
        </span>
      </div>
      <div className="absolute top-4 right-4 pointer-events-none">
        <span className="px-3 py-1 rounded-md text-xs font-semibold bg-brand-primary/90 text-white shadow-md backdrop-blur-sm flex items-center gap-1.5">
          <Sparkles className="w-3.5 h-3.5" />
          {afterLabel}
        </span>
      </div>

      {/* Bottom hint */}
      <div className="absolute bottom-3 left-1/2 -translate-x-1/2 pointer-events-none">
        <span className="px-3 py-1 rounded-full text-[11px] text-text-secondary bg-black/60 backdrop-blur-sm">
          Drag slider to compare structure & finish
        </span>
      </div>
    </div>
  );
}
