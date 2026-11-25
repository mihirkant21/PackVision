import React, { useState, useEffect } from 'react';
import {
  FolderKanban,
  Plus,
  Trash2,
  ArrowRight,
  Box,
  Calendar,
  X,
  Layers,
  Search,
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import { api } from '../services/api';

export default function Projects() {
  const { openProjectInStudio, showNotification } = useApp();
  const [projects, setProjects] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [isCreateOpen, setIsCreateOpen] = useState(false);

  // New Project Form State
  const [newName, setNewName] = useState('');
  const [newDesc, setNewDesc] = useState('');
  const [newCategory, setNewCategory] = useState('carton');
  const [newMaterial, setNewMaterial] = useState('coated_sbs');
  const [newFinish, setNewFinish] = useState('matte');
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    loadProjects();
  }, [search]);

  const loadProjects = async () => {
    try {
      setLoading(true);
      const res = await api.listProjects({ search: search || undefined });
      setProjects(res.items || []);
    } catch (err) {
      showNotification(`Failed to load projects: ${err.message}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleCreateProject = async (e) => {
    e.preventDefault();
    if (!newName.trim()) return;

    try {
      setIsSubmitting(true);
      const res = await api.createProject({
        name: newName.trim(),
        description: newDesc.trim() || null,
        package_type: newCategory,
        material: newMaterial,
        finish: newFinish,
      });

      showNotification('Project created successfully!', 'success');
      setIsCreateOpen(false);
      setNewName('');
      setNewDesc('');
      loadProjects();
    } catch (err) {
      showNotification(`Failed to create project: ${err.message}`, 'error');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleDeleteProject = async (id, name, e) => {
    e.stopPropagation();
    if (!window.confirm(`Are you sure you want to delete project "${name}"?`)) return;

    try {
      await api.deleteProject(id);
      showNotification(`Project "${name}" deleted.`, 'info');
      setProjects((prev) => prev.filter((p) => p.id !== id));
    } catch (err) {
      showNotification(`Failed to delete project: ${err.message}`, 'error');
    }
  };

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-[#1E2330] pb-6">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2.5 font-mono">
            <FolderKanban className="w-6 h-6 text-[#8B5CF6]" />
            Packaging Projects
          </h1>
          <p className="text-xs text-[#94A3B8] mt-1">
            Organize sketches, prompt configurations, and photoreal client renders
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="relative">
            <Search className="w-4 h-4 text-[#94A3B8] absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search projects..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="bg-[#171B25] border border-[#242A38] text-xs rounded-xl pl-9 pr-3 py-2 text-white focus:outline-none focus:border-[#8B5CF6] w-56"
            />
          </div>
          <button
            onClick={() => setIsCreateOpen(true)}
            className="px-4 py-2 rounded-xl text-xs font-semibold bg-[#8B5CF6] hover:bg-[#7C3AED] text-white flex items-center gap-1.5 shadow-lg shadow-[#8B5CF6]/25 transition-all"
          >
            <Plus className="w-4 h-4" />
            New Project
          </button>
        </div>
      </div>

      {/* Projects Grid */}
      {loading ? (
        <div className="py-20 text-center text-xs text-[#94A3B8]">Loading projects...</div>
      ) : projects.length === 0 ? (
        <div className="p-16 rounded-2xl bg-[#171B25] border border-[#242A38] text-center space-y-4">
          <div className="w-12 h-12 rounded-2xl bg-[#8B5CF6]/10 text-[#8B5CF6] flex items-center justify-center mx-auto">
            <Box className="w-6 h-6" />
          </div>
          <h3 className="font-semibold text-base text-white">No Projects Found</h3>
          <p className="text-xs text-[#94A3B8] max-w-sm mx-auto">
            Create your first packaging design project to save prompt variations and render outputs locally.
          </p>
          <button
            onClick={() => setIsCreateOpen(true)}
            className="px-4 py-2 rounded-xl text-xs font-semibold bg-[#8B5CF6] text-white inline-flex items-center gap-1.5"
          >
            <Plus className="w-4 h-4" />
            Create Project
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {projects.map((p) => (
            <div
              key={p.id}
              onClick={() => openProjectInStudio(p)}
              className="p-5 rounded-2xl bg-[#171B25] border border-[#242A38] hover:border-[#8B5CF6]/50 cursor-pointer flex flex-col justify-between group card-hover shadow-lg"
            >
              <div>
                <div className="flex items-center justify-between mb-3">
                  <span className="px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-[#8B5CF6]/15 text-[#A78BFA] uppercase tracking-wide">
                    {p.package_type || 'Custom'}
                  </span>
                  <button
                    onClick={(e) => handleDeleteProject(p.id, p.name, e)}
                    className="p-1.5 rounded-lg text-[#94A3B8] hover:text-rose-400 hover:bg-[#12151D] transition-colors"
                    title="Delete project"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
                <h3 className="font-bold text-base text-white group-hover:text-[#A78BFA] transition-colors">
                  {p.name}
                </h3>
                <p className="text-xs text-[#94A3B8] mt-1.5 line-clamp-2 leading-relaxed">
                  {p.description || 'No description provided.'}
                </p>
              </div>

              <div className="pt-5 border-t border-[#1E2330] mt-4 flex items-center justify-between text-xs text-[#94A3B8]">
                <div className="flex items-center gap-1.5">
                  <Layers className="w-3.5 h-3.5 text-[#8B5CF6]" />
                  <span>{p.generations_count || 0} renders</span>
                </div>
                <div className="flex items-center text-[#8B5CF6] font-medium gap-1 group-hover:translate-x-1 transition-transform">
                  <span>Open in Studio</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Modal: Create New Project */}
      {isCreateOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm animate-fade-in">
          <div className="bg-[#171B25] border border-[#242A38] w-full max-w-md rounded-2xl shadow-2xl overflow-hidden">
            <div className="flex items-center justify-between px-6 py-4 border-b border-[#242A38]">
              <h3 className="font-bold text-base text-white">Create New Project</h3>
              <button
                onClick={() => setIsCreateOpen(false)}
                className="p-1 rounded-lg text-[#94A3B8] hover:text-white"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <form onSubmit={handleCreateProject} className="p-6 space-y-4">
              <div>
                <label className="block text-xs font-medium text-[#94A3B8] mb-1">Project Name *</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. Nordic Gin Botanical Bottle"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  className="w-full bg-[#12151D] border border-[#242A38] text-xs rounded-xl px-3 py-2.5 text-white focus:outline-none focus:border-[#8B5CF6]"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-[#94A3B8] mb-1">Description / Client Brief</label>
                <textarea
                  rows={2}
                  placeholder="e.g. Minimalist glass bottle with wooden cap for craft distillery"
                  value={newDesc}
                  onChange={(e) => setNewDesc(e.target.value)}
                  className="w-full bg-[#12151D] border border-[#242A38] text-xs rounded-xl p-3 text-white focus:outline-none focus:border-[#8B5CF6] resize-none"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-medium text-[#94A3B8] mb-1">Packaging Form</label>
                  <select
                    value={newCategory}
                    onChange={(e) => setNewCategory(e.target.value)}
                    className="w-full bg-[#12151D] border border-[#242A38] text-xs rounded-xl px-2.5 py-2 text-white focus:outline-none focus:border-[#8B5CF6]"
                  >
                    <option value="carton">Folding Carton</option>
                    <option value="bottle">Bottle / Flask</option>
                    <option value="pouch">Flexible Pouch</option>
                    <option value="can">Beverage Can</option>
                    <option value="jar">Cosmetic Jar</option>
                    <option value="tube">Squeeze Tube</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-medium text-[#94A3B8] mb-1">Substrate Material</label>
                  <select
                    value={newMaterial}
                    onChange={(e) => setNewMaterial(e.target.value)}
                    className="w-full bg-[#12151D] border border-[#242A38] text-xs rounded-xl px-2.5 py-2 text-white focus:outline-none focus:border-[#8B5CF6]"
                  >
                    <option value="coated_sbs">Coated SBS</option>
                    <option value="kraft_paper">Kraft Paper</option>
                    <option value="glass">Glass</option>
                    <option value="aluminum">Aluminum</option>
                    <option value="plastic">Polymer</option>
                  </select>
                </div>
              </div>

              <div className="pt-4 flex justify-end gap-2.5">
                <button
                  type="button"
                  onClick={() => setIsCreateOpen(false)}
                  className="px-4 py-2 rounded-xl text-xs font-medium bg-[#12151D] text-[#94A3B8] hover:text-white border border-[#242A38]"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isSubmitting || !newName.trim()}
                  className="px-5 py-2 rounded-xl text-xs font-semibold bg-[#8B5CF6] hover:bg-[#7C3AED] disabled:opacity-50 text-white shadow-lg"
                >
                  Create Project
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
