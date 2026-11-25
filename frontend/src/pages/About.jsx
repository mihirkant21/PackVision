import React from 'react';
import { Info, Box, Shield, Cpu, BookOpen, ExternalLink, CheckCircle } from 'lucide-react';

export default function About() {
  const models = [
    {
      name: 'Stable Diffusion 1.5 Base',
      id: 'runwayml/stable-diffusion-v1-5',
      license: 'CreativeML Open RAIL-M',
      purpose: 'Latent diffusion foundation model synthesizing photorealistic textures, lighting, and materials',
      free: true,
    },
    {
      name: 'ControlNet Canny v1.1',
      id: 'lllyasviel/control_v11p_sd15_canny',
      license: 'Apache 2.0',
      purpose: 'Locks dielines, carton folds, label silhouettes, and structural packaging geometry',
      free: true,
    },
    {
      name: 'OpenCV Computer Vision',
      id: 'opencv-python (v4.8+)',
      license: 'Apache 2.0',
      purpose: 'Sketch aspect-ratio normalization, Canny edge detection, and homography perspective transformation',
      free: true,
    },
    {
      name: 'CIELAB Ink Compositor',
      id: 'PackVision Deterministic Engine',
      license: 'MIT',
      purpose: 'Stage-2 brand logo overlay preserving vector sharpness, substrate grain, and micro-shadows',
      free: true,
    },
  ];

  return (
    <div className="p-8 space-y-8 max-w-5xl mx-auto">
      {/* Header */}
      <div className="border-b border-[#1E2330] pb-6">
        <h1 className="text-2xl font-bold text-white flex items-center gap-2.5 font-mono">
          <Info className="w-6 h-6 text-[#8B5CF6]" />
          About PackVision AI
        </h1>
        <p className="text-xs text-[#94A3B8] mt-1">
          Open-Source Sketch-to-Photorealistic Packaging Product Renderer & Design Synthesis System
        </p>
      </div>

      {/* Project Mission & Local-Only Philosophy */}
      <div className="p-6 rounded-2xl bg-[#171B25] border border-[#242A38] space-y-4 shadow-xl">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <Shield className="w-5 h-5 text-[#8B5CF6]" />
          100% Free & Local-Only Architecture
        </h2>
        <p className="text-xs text-[#94A3B8] leading-relaxed">
          PackVision AI was purpose-built to operate entirely on consumer laptop hardware without requiring any cloud hosting, paid API subscriptions (such as OpenAI, Stability AI, or Midjourney), or third-party servers. All user sketches, prompt configurations, SQLite database records, and rendered high-resolution images are created and stored strictly on your local disk.
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 pt-2">
          <div className="p-3 rounded-xl bg-[#12151D] border border-[#242A38] text-xs">
            <span className="text-[#8B5CF6] font-semibold block">Frontend</span>
            <span className="text-white mt-0.5 block">React 18 + Vite</span>
            <span className="text-[10px] text-[#94A3B8]">localhost:5173</span>
          </div>
          <div className="p-3 rounded-xl bg-[#12151D] border border-[#242A38] text-xs">
            <span className="text-[#8B5CF6] font-semibold block">Backend API</span>
            <span className="text-white mt-0.5 block">Python + FastAPI</span>
            <span className="text-[10px] text-[#94A3B8]">localhost:8000</span>
          </div>
          <div className="p-3 rounded-xl bg-[#12151D] border border-[#242A38] text-xs">
            <span className="text-[#8B5CF6] font-semibold block">Database</span>
            <span className="text-white mt-0.5 block">SQLite 3</span>
            <span className="text-[10px] text-[#94A3B8]">packvision.db</span>
          </div>
          <div className="p-3 rounded-xl bg-[#12151D] border border-[#242A38] text-xs">
            <span className="text-[#8B5CF6] font-semibold block">Inference Priority</span>
            <span className="text-white mt-0.5 block">GTX 1650 Ti / Colab</span>
            <span className="text-[10px] text-[#94A3B8]">4GB VRAM Low-Memory</span>
          </div>
        </div>
      </div>

      {/* Model Inventory & Licenses */}
      <div className="space-y-4">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <BookOpen className="w-5 h-5 text-[#8B5CF6]" />
          Models, Frameworks & Licensing
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {models.map((m, idx) => (
            <div key={idx} className="p-5 rounded-2xl bg-[#171B25] border border-[#242A38] space-y-2">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-sm text-white">{m.name}</h3>
                <span className="px-2.5 py-0.5 rounded-full text-[10px] font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                  {m.license}
                </span>
              </div>
              <div className="font-mono text-[11px] text-[#A78BFA]">{m.id}</div>
              <p className="text-xs text-[#94A3B8] leading-relaxed pt-1">{m.purpose}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Packaging Categories Supported */}
      <div className="p-6 rounded-2xl bg-[#171B25] border border-[#242A38] space-y-3">
        <h2 className="text-base font-bold text-white flex items-center gap-2">
          <Box className="w-5 h-5 text-[#8B5CF6]" />
          Supported Packaging Industrial Forms
        </h2>
        <div className="flex flex-wrap gap-2 text-xs">
          {[
            'Folding Cartons & Rigid Boxes',
            'Cosmetic & Beverage Glass Bottles',
            'Flexible Pouches & Sachets',
            'Aluminum Beverage & Aerosol Cans',
            'Skin Care & Cream Jars',
            'Plastic & Aluminum Squeeze Tubes',
            'Cardboard Outer Sleeves & Trays',
          ].map((cat, idx) => (
            <span
              key={idx}
              className="px-3 py-1.5 rounded-xl bg-[#12151D] border border-[#242A38] text-white flex items-center gap-1.5"
            >
              <CheckCircle className="w-3.5 h-3.5 text-[#8B5CF6]" />
              {cat}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
