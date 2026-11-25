import React, { useEffect, useState } from 'react';
import { X, Award, CheckCircle2, AlertTriangle, ShieldCheck, BarChart3, RefreshCw } from 'lucide-react';
import { api } from '../services/api';

export default function EvaluationModal({ sketchAssetId, renderAssetId, prompt, onClose }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    runEvaluation();
  }, [sketchAssetId, renderAssetId]);

  const runEvaluation = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await api.evaluateRender({
        sketch_asset_id: sketchAssetId,
        render_asset_id: renderAssetId,
        prompt: prompt || '',
      });
      setData(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm animate-fade-in">
      <div className="bg-dark-card border border-dark-border w-full max-w-lg rounded-2xl shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-dark-border bg-dark-surface/50">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-brand-primary/10 text-brand-primary">
              <Award className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-semibold text-text-primary">Structural Preservation & Metric Report</h2>
              <p className="text-xs text-text-secondary">Comparing sketch dielines vs rendered packaging geometry</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-text-secondary hover:text-white hover:bg-dark-surface transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6">
          {loading ? (
            <div className="py-12 flex flex-col items-center justify-center space-y-3">
              <RefreshCw className="w-8 h-8 text-brand-primary animate-spin" />
              <p className="text-sm text-text-secondary">Computing structural SSIM & contour IoU...</p>
            </div>
          ) : error ? (
            <div className="py-6 flex flex-col items-center text-center space-y-3">
              <AlertTriangle className="w-10 h-10 text-amber-500" />
              <p className="text-sm font-medium text-text-primary">Evaluation Error</p>
              <p className="text-xs text-text-secondary max-w-sm">{error}</p>
            </div>
          ) : data && data.metrics ? (
            <div className="space-y-5">
              {/* Score Hero */}
              <div className="p-4 rounded-xl bg-dark-surface border border-dark-border flex items-center justify-between">
                <div>
                  <span className="text-xs font-medium text-text-secondary uppercase tracking-wider">
                    Preservation Score
                  </span>
                  <div className="text-3xl font-bold text-brand-primary mt-0.5">
                    {data.metrics.structural_preservation_score}%
                  </div>
                  <div className="flex items-center gap-1.5 mt-1 text-xs text-emerald-400">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    <span>Fidelity Grade: {data.metrics.structural_rating}</span>
                  </div>
                </div>
                <div className="w-16 h-16 rounded-full border-4 border-brand-primary/30 border-t-brand-primary flex items-center justify-center font-mono font-bold text-sm text-white">
                  {Math.round(data.metrics.structural_preservation_score)}
                </div>
              </div>

              {/* Metrics Grid */}
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3.5 rounded-lg bg-dark-surface/60 border border-dark-border">
                  <span className="text-[11px] text-text-secondary block">Edge SSIM (Structure)</span>
                  <span className="text-lg font-mono font-bold text-text-primary mt-1 block">
                    {data.metrics.edge_ssim}
                  </span>
                  <span className="text-[10px] text-text-muted mt-0.5 block">
                    Source: {data.provenance?.edge_ssim}
                  </span>
                </div>
                <div className="p-3.5 rounded-lg bg-dark-surface/60 border border-dark-border">
                  <span className="text-[11px] text-text-secondary block">Contour IoU (Edges)</span>
                  <span className="text-lg font-mono font-bold text-text-primary mt-1 block">
                    {data.metrics.edge_iou}
                  </span>
                  <span className="text-[10px] text-text-muted mt-0.5 block">
                    Source: {data.provenance?.edge_iou}
                  </span>
                </div>
              </div>

              {/* Summary note */}
              <div className="p-3.5 rounded-lg bg-dark-surface/40 border border-dark-border text-xs text-text-secondary leading-relaxed">
                <div className="flex items-center gap-1.5 font-medium text-text-primary mb-1">
                  <ShieldCheck className="w-4 h-4 text-brand-primary" />
                  Preservation Assessment
                </div>
                {data.summary}
              </div>
            </div>
          ) : null}
        </div>

        {/* Footer */}
        <div className="px-6 py-3.5 border-t border-dark-border bg-dark-surface/30 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg text-xs font-medium bg-dark-surface border border-dark-border hover:bg-dark-border text-text-primary transition-colors"
          >
            Close Report
          </button>
        </div>
      </div>
    </div>
  );
}
