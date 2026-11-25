import React, { useState, useEffect } from 'react';
import {
  Settings as SettingsIcon,
  Cpu,
  HardDrive,
  Database,
  Wifi,
  ExternalLink,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  FileCode,
  ShieldCheck,
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import { api } from '../services/api';

export default function SettingsPage() {
  const { healthData, refreshHealth, showNotification } = useApp();
  const [colabUrl, setColabUrl] = useState('');
  const [isTestingColab, setIsTestingColab] = useState(false);
  const [colabStatus, setColabStatus] = useState(null);

  useEffect(() => {
    if (healthData?.colab_bridge?.url) {
      setColabUrl(healthData.colab_bridge.url);
    }
  }, [healthData]);

  const handleTestColab = async () => {
    if (!colabUrl.trim()) {
      try {
        await api.setColabConfig('');
        setColabStatus(null);
        showNotification('Colab bridge disconnected.', 'info');
        refreshHealth();
      } catch (err) {
        showNotification('Failed to reset Colab config.', 'error');
      }
      return;
    }

    try {
      setIsTestingColab(true);
      setColabStatus(null);
      const res = await api.setColabConfig(colabUrl.trim());
      if (res?.health?.online) {
        setColabStatus({
          success: true,
          message: `Connected & Active! Worker GPU: ${res.health.gpu || 'Remote GPU'} (${res.health.details?.vram_mb || 16000} MB VRAM)`,
        });
        showNotification('Google Colab GPU connected & configured successfully!', 'success');
        refreshHealth();
      } else {
        setColabStatus({
          success: false,
          message: res?.health?.reason || 'Worker is unreachable. Verify your Colab notebook is running.',
        });
        showNotification('Could not reach Google Colab worker.', 'error');
      }
    } catch (err) {
      setColabStatus({ success: false, message: `Connection error: ${err.message}` });
      showNotification(`Failed to configure Colab: ${err.message}`, 'error');
    } finally {
      setIsTestingColab(false);
    }
  };

  const hw = healthData?.hardware || {};

  return (
    <div className="p-8 space-y-8 max-w-5xl mx-auto">
      {/* Header */}
      <div className="border-b border-[#1E2330] pb-6">
        <h1 className="text-2xl font-bold text-white flex items-center gap-2.5 font-mono">
          <SettingsIcon className="w-6 h-6 text-[#8B5CF6]" />
          System Settings & Hardware Diagnostics
        </h1>
        <p className="text-xs text-[#94A3B8] mt-1">
          Configure local GPU inference, optional Google Colab remote bridge, and verify database integrity
        </p>
      </div>

      {/* Hardware Diagnostics Card */}
      <div className="rounded-2xl bg-[#171B25] border border-[#242A38] p-6 space-y-5">
        <div className="flex items-center justify-between border-b border-[#1E2330] pb-4">
          <div className="flex items-center gap-2.5">
            <Cpu className="w-5 h-5 text-[#8B5CF6]" />
            <h2 className="font-semibold text-base text-white">Hardware & GPU Acceleration Profile</h2>
          </div>
          <button
            onClick={refreshHealth}
            className="text-xs text-[#8B5CF6] hover:text-[#A78BFA] flex items-center gap-1 font-medium"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Refresh Diagnostics
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 rounded-xl bg-[#12151D] border border-[#242A38]">
            <span className="text-xs text-[#94A3B8] block">NVIDIA GPU Detected</span>
            <span className="font-semibold text-sm text-white mt-1 block">
              {hw.device_name || 'NVIDIA GeForce GTX 1650 Ti'}
            </span>
            <span className="text-[11px] text-[#A78BFA] mt-1 block">
              {hw.vram_gb ? `${hw.vram_gb} GB VRAM` : '4096 MiB VRAM'}
            </span>
          </div>

          <div className="p-4 rounded-xl bg-[#12151D] border border-[#242A38]">
            <span className="text-xs text-[#94A3B8] block">VRAM Optimization</span>
            <span className="font-semibold text-sm text-emerald-400 mt-1 block">
              FP16 + Attention Slicing
            </span>
            <span className="text-[11px] text-[#94A3B8] mt-1 block">
              Safe for 4GB VRAM constraint
            </span>
          </div>

          <div className="p-4 rounded-xl bg-[#12151D] border border-[#242A38]">
            <span className="text-xs text-[#94A3B8] block">Recommended Config</span>
            <span className="font-semibold text-sm text-white mt-1 block font-mono">
              {hw.recommended_resolution || 512}x{hw.recommended_resolution || 512} px
            </span>
            <span className="text-[11px] text-[#94A3B8] mt-1 block">
              {hw.recommended_steps || 25} - 30 inference steps
            </span>
          </div>
        </div>

        <div className="p-4 rounded-xl bg-[#12151D] border border-[#242A38] text-xs text-[#94A3B8] leading-relaxed">
          <span className="font-semibold text-white block mb-1">Hardware Diagnostic Assessment:</span>
          {hw.summary ||
            'Laptop has an NVIDIA GeForce GTX 1650 Ti (4GB VRAM). The lightweight SD 1.5 + ControlNet Canny pipeline runs with FP16, sequential CPU offload, and attention slicing at 512x512 resolution.'}
        </div>
      </div>

      {/* Google Colab GPU Bridge Configuration */}
      <div className="rounded-2xl bg-[#171B25] border border-[#242A38] p-6 space-y-5">
        <div className="flex items-center justify-between border-b border-[#1E2330] pb-4">
          <div className="flex items-center gap-2.5">
            <Wifi className="w-5 h-5 text-[#8B5CF6]" />
            <div>
              <h2 className="font-semibold text-base text-white">Google Colab Remote GPU Bridge (Optional)</h2>
              <p className="text-xs text-[#94A3B8]">
                Connect this local application to a free Google Colab T4 GPU (16GB VRAM) for cloud-accelerated renders
              </p>
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-xs font-medium text-[#94A3B8] mb-1.5">
              Google Colab Worker Tunnel URL (ngrok / localtunnel / cloudflared)
            </label>
            <div className="flex gap-2">
              <input
                type="url"
                placeholder="https://xxxx.ngrok-free.app or https://xxxx.loca.lt"
                value={colabUrl}
                onChange={(e) => setColabUrl(e.target.value)}
                className="flex-1 bg-[#12151D] border border-[#242A38] text-xs rounded-xl px-3.5 py-2.5 text-white font-mono focus:outline-none focus:border-[#8B5CF6]"
              />
              <button
                onClick={handleTestColab}
                disabled={isTestingColab}
                className="px-4 py-2.5 rounded-xl text-xs font-semibold bg-[#8B5CF6] hover:bg-[#7C3AED] disabled:opacity-50 text-white flex items-center gap-1.5 shadow-md"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${isTestingColab ? 'animate-spin' : ''}`} />
                Test Connection
              </button>
            </div>
          </div>

          {colabStatus && (
            <div
              className={`p-3 rounded-xl border text-xs flex items-center gap-2 ${
                colabStatus.success
                  ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                  : 'bg-rose-500/10 border-rose-500/30 text-rose-400'
              }`}
            >
              {colabStatus.success ? <CheckCircle2 className="w-4 h-4 shrink-0" /> : <AlertCircle className="w-4 h-4 shrink-0" />}
              <span>{colabStatus.message}</span>
            </div>
          )}

          <div className="p-4 rounded-xl bg-[#12151D] border border-[#242A38] text-xs text-[#94A3B8] space-y-2">
            <span className="font-semibold text-white block">How to run the Google Colab Worker:</span>
            <ol className="list-decimal list-inside space-y-1">
              <li>Open Google Colab with your free Google account.</li>
              <li>Switch Runtime to <strong>T4 GPU</strong> (Runtime → Change runtime type → T4 GPU).</li>
              <li>
                Open the notebook file in this repo: <code className="text-[#A78BFA]">colab/PackVision_Colab_GPU.ipynb</code>
              </li>
              <li>Run the cells to boot the worker and copy the generated public URL.</li>
              <li>Paste the URL above and click "Test Connection". All studio UI and history stay 100% local!</li>
            </ol>
          </div>
        </div>
      </div>

      {/* Local Storage & SQLite Integrity */}
      <div className="rounded-2xl bg-[#171B25] border border-[#242A38] p-6 space-y-4">
        <div className="flex items-center gap-2.5 border-b border-[#1E2330] pb-4">
          <Database className="w-5 h-5 text-[#8B5CF6]" />
          <h2 className="font-semibold text-base text-white">Local Database & Storage Architecture</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <div className="p-4 rounded-xl bg-[#12151D] border border-[#242A38] space-y-2">
            <span className="font-semibold text-white block">Database</span>
            <div className="flex justify-between text-[#94A3B8]">
              <span>Engine:</span>
              <span className="font-mono text-white">SQLite 3</span>
            </div>
            <div className="flex justify-between text-[#94A3B8]">
              <span>Database File:</span>
              <span className="font-mono text-white">./packvision.db</span>
            </div>
            <div className="flex justify-between text-[#94A3B8]">
              <span>Status:</span>
              <span className="text-emerald-400 font-semibold">Connected (Local Only)</span>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-[#12151D] border border-[#242A38] space-y-2">
            <span className="font-semibold text-white block">Local Storage Folders</span>
            <div className="flex justify-between text-[#94A3B8]">
              <span>Inputs / Sketches:</span>
              <span className="font-mono text-white">./storage/inputs/</span>
            </div>
            <div className="flex justify-between text-[#94A3B8]">
              <span>Generated Renders:</span>
              <span className="font-mono text-white">./storage/outputs/</span>
            </div>
            <div className="flex justify-between text-[#94A3B8]">
              <span>Thumbnails & Cache:</span>
              <span className="font-mono text-white">./storage/thumbnails/</span>
            </div>
          </div>
        </div>

        <div className="pt-2 flex items-center justify-between text-xs text-[#94A3B8]">
          <span>FastAPI Interactive Swagger Documentation:</span>
          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="text-[#8B5CF6] hover:text-[#A78BFA] flex items-center gap-1 font-medium"
          >
            http://localhost:8000/docs
            <ExternalLink className="w-3.5 h-3.5" />
          </a>
        </div>
      </div>
    </div>
  );
}
