import React, { useEffect, useState } from 'react';
import {
  Sparkles,
  FolderKanban,
  History,
  ArrowRight,
  Box,
  Layers,
  Clock,
  CheckCircle2,
  Cpu,
  Plus,
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import { api } from '../services/api';

export default function Dashboard() {
  const { setActiveView, openProjectInStudio, loadGenerationInStudio, showNotification } = useApp();
  const [stats, setStats] = useState({ totalGenerations: 0, totalProjects: 0, completedCount: 0 });
  const [recentProjects, setRecentProjects] = useState([]);
  const [recentGenerations, setRecentGenerations] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      const [genRes, projRes] = await Promise.all([
        api.listGenerations({ limit: 6 }),
        api.listProjects({ limit: 4 }),
      ]);

      setRecentGenerations(genRes.items || []);
      setRecentProjects(projRes.items || []);

      const completed = (genRes.items || []).filter((g) => g.status === 'completed').length;
      setStats({
        totalGenerations: genRes.total || 0,
        totalProjects: projRes.total || 0,
        completedCount: completed,
      });
    } catch (err) {
      console.error('Error loading dashboard data:', err);
    } finally {
      setLoading(false);
    }
  };

  const sampleTemplates = [
    {
      title: 'Cosmetic Luxury Carton',
      packageType: 'carton',
      material: 'coated_sbs',
      finish: 'matte',
      desc: 'Rigid cardboard box with matte black texture and gold foil lettering',
    },
    {
      title: 'Cold-Brew Glass Bottle',
      packageType: 'bottle',
      material: 'glass',
      finish: 'glossy',
      desc: 'Amber tinted glass bottle with frosted condensation and studio softbox light',
    },
    {
      title: 'Kraft Coffee Pouch',
      packageType: 'pouch',
      material: 'kraft_paper',
      finish: 'matte',
      desc: 'Eco-friendly recyclable kraft pouch with resealable zip and clean pedestal',
    },
    {
      title: 'Beverage Soda Can',
      packageType: 'can',
      material: 'aluminum',
      finish: 'metallic',
      desc: 'Brushed aluminum cylinder with reflective rim highlights and drop shadow',
    },
  ];

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto">
      {/* Hero Welcome Banner */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-[#171B25] to-[#12151D] border border-[#242A38] p-8 shadow-2xl">
        <div className="relative z-10 max-w-2xl space-y-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#8B5CF6]/15 text-[#A78BFA] text-xs font-semibold border border-[#8B5CF6]/30">
            <Sparkles className="w-3.5 h-3.5" />
            Stable Diffusion 1.5 + ControlNet Canny Studio
          </div>
          <h1 className="text-3xl font-bold tracking-tight text-white font-mono">
            Transform Packaging Sketches into Commercial Product Photographs
          </h1>
          <p className="text-sm text-[#94A3B8] leading-relaxed">
            Preserve carton geometries, dielines, bottle contours, and brand logos while synthesizing photorealistic
            materials, studio lighting rigs, contact shadows, and luxury finishes.
          </p>
          <div className="pt-2 flex flex-wrap gap-3">
            <button
              onClick={() => setActiveView('studio')}
              className="px-5 py-2.5 rounded-xl font-semibold text-sm bg-[#8B5CF6] hover:bg-[#7C3AED] text-white flex items-center gap-2 shadow-lg shadow-[#8B5CF6]/25 transition-all"
            >
              <Sparkles className="w-4 h-4" />
              Open Studio Workbench
            </button>
            <button
              onClick={() => setActiveView('projects')}
              className="px-5 py-2.5 rounded-xl font-medium text-sm bg-[#171B25] hover:bg-[#242A38] text-white border border-[#242A38] transition-all"
            >
              Browse Projects
            </button>
          </div>
        </div>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="p-5 rounded-xl bg-[#171B25] border border-[#242A38]">
          <span className="text-xs text-[#94A3B8] font-medium block">Total Renders</span>
          <div className="text-2xl font-bold text-white mt-1 font-mono">{stats.totalGenerations}</div>
          <span className="text-[11px] text-[#8B5CF6] mt-1 block">Local SQLite database</span>
        </div>
        <div className="p-5 rounded-xl bg-[#171B25] border border-[#242A38]">
          <span className="text-xs text-[#94A3B8] font-medium block">Active Projects</span>
          <div className="text-2xl font-bold text-white mt-1 font-mono">{stats.totalProjects}</div>
          <span className="text-[11px] text-emerald-400 mt-1 block">Organized workspace</span>
        </div>
        <div className="p-5 rounded-xl bg-[#171B25] border border-[#242A38]">
          <span className="text-xs text-[#94A3B8] font-medium block">Packaging Types</span>
          <div className="text-2xl font-bold text-white mt-1 font-mono">7 Categories</div>
          <span className="text-[11px] text-[#A78BFA] mt-1 block">Cartons, Bottles, Pouches, Cans</span>
        </div>
        <div className="p-5 rounded-xl bg-[#171B25] border border-[#242A38]">
          <span className="text-xs text-[#94A3B8] font-medium block">AI Engine</span>
          <div className="text-2xl font-bold text-white mt-1 font-mono">SD 1.5 + Canny</div>
          <span className="text-[11px] text-[#8B5CF6] mt-1 block">FP16 Low-VRAM Pipeline</span>
        </div>
      </div>

      {/* Quick Start Packaging Templates */}
      <div>
        <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
          <Box className="w-5 h-5 text-[#8B5CF6]" />
          Quick-Start Packaging Presets
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {sampleTemplates.map((tmpl, idx) => (
            <div
              key={idx}
              onClick={() => {
                openProjectInStudio({
                  name: tmpl.title,
                  package_type: tmpl.packageType,
                  material: tmpl.material,
                  finish: tmpl.finish,
                  prompt: tmpl.desc,
                });
              }}
              className="p-5 rounded-xl bg-[#171B25] border border-[#242A38] hover:border-[#8B5CF6]/50 cursor-pointer card-hover flex flex-col justify-between"
            >
              <div>
                <span className="px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-[#8B5CF6]/15 text-[#A78BFA] uppercase tracking-wide">
                  {tmpl.packageType}
                </span>
                <h3 className="font-semibold text-sm text-white mt-2.5">{tmpl.title}</h3>
                <p className="text-xs text-[#94A3B8] mt-1.5 line-clamp-2 leading-relaxed">{tmpl.desc}</p>
              </div>
              <div className="pt-4 flex items-center text-xs text-[#8B5CF6] font-medium gap-1">
                <span>Start in Studio</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Recent Generates & Projects Split */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Recent Renders */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <History className="w-5 h-5 text-[#8B5CF6]" />
              Recent Packaging Renders
            </h2>
            <button
              onClick={() => setActiveView('history')}
              className="text-xs text-[#8B5CF6] hover:text-[#A78BFA] font-medium flex items-center gap-1"
            >
              View all <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          {recentGenerations.length === 0 ? (
            <div className="p-8 rounded-xl bg-[#171B25] border border-[#242A38] text-center text-xs text-[#94A3B8]">
              No generations recorded yet. Upload a sketch in the studio to start!
            </div>
          ) : (
            <div className="grid grid-cols-3 gap-3">
              {recentGenerations.slice(0, 6).map((gen) => {
                const outputAsset = gen.outputs && gen.outputs[0] ? gen.outputs[0].asset : null;
                const primaryUrl = outputAsset?.url || gen.input_asset?.url;
                const thumbUrl = outputAsset?.thumbnail_url || outputAsset?.url || gen.input_asset?.thumbnail_url;
                return (
                  <div
                    key={gen.id}
                    onClick={() => loadGenerationInStudio(gen)}
                    className="relative aspect-square rounded-xl overflow-hidden bg-black border border-[#242A38] hover:border-[#8B5CF6] cursor-pointer group card-hover flex items-center justify-center"
                  >
                    {thumbUrl ? (
                      <img
                        src={thumbUrl}
                        alt={gen.package_type}
                        onError={(e) => {
                          if (primaryUrl && e.target.src !== primaryUrl) {
                            e.target.src = primaryUrl;
                          }
                        }}
                        className="w-full h-full object-contain"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-xs text-[#94A3B8]">
                        Processing
                      </div>
                    )}
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity p-2 flex flex-col justify-end">
                      <span className="text-[11px] font-semibold text-white capitalize">{gen.package_type}</span>
                      <span className="text-[10px] text-[#A78BFA]">{gen.material}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Recent Projects */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <FolderKanban className="w-5 h-5 text-[#8B5CF6]" />
              Active Projects
            </h2>
            <button
              onClick={() => setActiveView('projects')}
              className="text-xs text-[#8B5CF6] hover:text-[#A78BFA] font-medium flex items-center gap-1"
            >
              View all <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          {recentProjects.length === 0 ? (
            <div className="p-8 rounded-xl bg-[#171B25] border border-[#242A38] text-center text-xs text-[#94A3B8]">
              No projects created yet. Create a project to save your generation workflows!
            </div>
          ) : (
            <div className="space-y-2.5">
              {recentProjects.map((p) => (
                <div
                  key={p.id}
                  onClick={() => openProjectInStudio(p)}
                  className="p-4 rounded-xl bg-[#171B25] border border-[#242A38] hover:border-[#8B5CF6] cursor-pointer flex items-center justify-between card-hover"
                >
                  <div>
                    <h3 className="font-semibold text-sm text-white">{p.name}</h3>
                    <div className="flex items-center gap-2 mt-1 text-xs text-[#94A3B8]">
                      <span className="capitalize">{p.package_type || 'Custom'}</span>
                      <span>•</span>
                      <span>{p.generations_count || 0} renders</span>
                    </div>
                  </div>
                  <ArrowRight className="w-4 h-4 text-[#94A3B8] group-hover:text-[#8B5CF6]" />
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
