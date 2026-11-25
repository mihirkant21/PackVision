import React, { useState, useEffect } from 'react';
import {
  History as HistoryIcon,
  Search,
  Filter,
  Download,
  Award,
  Sparkles,
  Trash2,
  RefreshCw,
  Box,
  Layers,
  ArrowRight,
  AlertTriangle,
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import { api } from '../services/api';

export default function HistoryPage() {
  const { loadGenerationInStudio, openEvaluationModal, showNotification } = useApp();
  const [generations, setGenerations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [statusFilter, setStatusFilter] = useState('');
  const [selectedGen, setSelectedGen] = useState(null);
  const [isClearing, setIsClearing] = useState(false);

  useEffect(() => {
    loadHistory();
  }, [statusFilter]);

  const loadHistory = async () => {
    try {
      setLoading(true);
      const res = await api.listGenerations({
        limit: 100,
        status: statusFilter || undefined,
      });
      setGenerations(res.items || []);
    } catch (err) {
      showNotification(`Failed to load history: ${err.message}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id, e) => {
    e.stopPropagation();
    if (!window.confirm('Are you sure you want to delete this packaging render?')) return;
    try {
      await api.deleteGeneration(id);
      showNotification('Generation record deleted.', 'info');
      setGenerations((prev) => prev.filter((g) => g.id !== id));
      if (selectedGen?.id === id) setSelectedGen(null);
    } catch (err) {
      showNotification(`Failed to delete: ${err.message}`, 'error');
    }
  };

  const handleClearAll = async () => {
    if (!window.confirm('Are you sure you want to delete ALL generation history? This will delete all render records from your database.')) {
      return;
    }
    try {
      setIsClearing(true);
      const res = await api.clearAllGenerations();
      setGenerations([]);
      setSelectedGen(null);
      showNotification(res.message || 'All generation history cleared successfully.', 'info');
    } catch (err) {
      showNotification(`Failed to clear history: ${err.message}`, 'error');
    } finally {
      setIsClearing(false);
    }
  };

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-[#1E2330] pb-6">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2.5 font-mono">
            <HistoryIcon className="w-6 h-6 text-[#8B5CF6]" />
            Generation History
          </h1>
          <p className="text-xs text-[#94A3B8] mt-1">
            Browse, inspect, and manage all packaging renders stored in your local SQLite database ({generations.length} total)
          </p>
        </div>

        {/* Actions & Filters */}
        <div className="flex flex-wrap items-center gap-3">
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="bg-[#171B25] border border-[#242A38] text-xs rounded-xl px-3 py-2 text-white focus:outline-none focus:border-[#8B5CF6]"
          >
            <option value="">All Statuses</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
            <option value="queued">Queued</option>
          </select>

          <button
            onClick={loadHistory}
            className="p-2 rounded-xl bg-[#171B25] border border-[#242A38] text-[#94A3B8] hover:text-white transition-colors"
            title="Refresh history"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>

          {generations.length > 0 && (
            <button
              onClick={handleClearAll}
              disabled={isClearing}
              className="px-3.5 py-2 rounded-xl text-xs font-semibold bg-rose-600/15 hover:bg-rose-600 border border-rose-600/30 hover:border-rose-600 text-rose-400 hover:text-white flex items-center gap-1.5 transition-all shadow-sm"
              title="Delete all generation history"
            >
              <Trash2 className="w-3.5 h-3.5" />
              {isClearing ? 'Clearing...' : 'Clear All History'}
            </button>
          )}
        </div>
      </div>

      {/* Gallery Grid */}
      {loading ? (
        <div className="py-20 text-center text-xs text-[#94A3B8]">Loading generation history...</div>
      ) : generations.length === 0 ? (
        <div className="p-16 rounded-2xl bg-[#171B25] border border-[#242A38] text-center space-y-3">
          <Box className="w-10 h-10 text-[#8B5CF6] mx-auto opacity-50" />
          <h3 className="font-semibold text-base text-white">No Generations Found</h3>
          <p className="text-xs text-[#94A3B8]">Start a render in the studio to build your history log.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {generations.map((gen) => {
            const outputAsset = gen.outputs && gen.outputs[0] ? gen.outputs[0].asset : null;
            const primaryUrl = outputAsset?.url || gen.input_asset?.url;
            const thumbUrl = outputAsset?.thumbnail_url || outputAsset?.url || gen.input_asset?.thumbnail_url;
            const isCompleted = gen.status === 'completed';

            return (
              <div
                key={gen.id}
                onClick={() => loadGenerationInStudio(gen)}
                className="rounded-2xl bg-[#171B25] border border-[#242A38] hover:border-[#8B5CF6]/50 cursor-pointer overflow-hidden flex flex-col justify-between group card-hover shadow-lg"
              >
                {/* Image Preview Container */}
                <div className="relative aspect-[4/3] bg-black overflow-hidden flex items-center justify-center">
                  {thumbUrl ? (
                    <img
                      src={thumbUrl}
                      alt={gen.package_type}
                      onError={(e) => {
                        // Fallback to raw output image if thumbnail fails
                        if (primaryUrl && e.target.src !== primaryUrl) {
                          e.target.src = primaryUrl;
                        }
                      }}
                      className="w-full h-full object-contain group-hover:scale-105 transition-transform duration-300"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-xs text-[#94A3B8]">
                      {gen.status}
                    </div>
                  )}

                  <div className="absolute top-3 left-3">
                    <span className="px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-black/75 text-white uppercase tracking-wider backdrop-blur-md border border-white/10">
                      {gen.package_type}
                    </span>
                  </div>

                  <div className="absolute top-3 right-3 flex items-center gap-1.5">
                    <span
                      className={`px-2 py-0.5 rounded-full text-[10px] font-semibold backdrop-blur-md capitalize ${
                        isCompleted
                          ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                          : 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                      }`}
                    >
                      {gen.status}
                    </span>
                    <button
                      onClick={(e) => handleDelete(gen.id, e)}
                      className="p-1.5 rounded-lg bg-black/75 text-[#94A3B8] hover:text-white hover:bg-rose-600 transition-colors backdrop-blur-md"
                      title="Delete this render"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>

                {/* Details Footer */}
                <div className="p-4 space-y-3">
                  <div>
                    <div className="flex items-center justify-between text-xs text-[#94A3B8]">
                      <span className="capitalize">{gen.material || 'Material'} • {gen.finish || 'Finish'}</span>
                      <span className="font-mono text-[10px]">{new Date(gen.created_at).toLocaleDateString()}</span>
                    </div>
                    <p className="text-xs text-white mt-1 line-clamp-2 leading-relaxed">
                      {gen.prompt || 'Packaging prototype render'}
                    </p>
                  </div>

                  <div className="pt-3 border-t border-[#1E2330] flex items-center justify-between text-xs">
                    {outputAsset && gen.input_asset && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          openEvaluationModal(gen.input_asset.id, outputAsset.id, gen.prompt);
                        }}
                        className="text-[11px] text-[#8B5CF6] hover:text-[#A78BFA] flex items-center gap-1 font-medium"
                      >
                        <Award className="w-3.5 h-3.5" />
                        SSIM Metric
                      </button>
                    )}

                    <div className="flex items-center text-[#8B5CF6] font-medium gap-1 ml-auto group-hover:translate-x-1 transition-transform">
                      <span>Open in Studio</span>
                      <ArrowRight className="w-3.5 h-3.5" />
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
