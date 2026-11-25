import React, { useState } from 'react';
import {
  LayoutDashboard,
  Sparkles,
  FolderKanban,
  History,
  Settings,
  Info,
  Box,
  Cpu,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import { useApp } from '../context/AppContext';

export default function Sidebar() {
  const { activeView, setActiveView, healthData } = useApp();
  const [isCollapsed, setIsCollapsed] = useState(false);

  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'studio', label: 'Studio Workbench', icon: Sparkles },
    { id: 'projects', label: 'My Projects', icon: FolderKanban },
    { id: 'history', label: 'Generation History', icon: History },
    { id: 'settings', label: 'Settings & Hardware', icon: Settings },
    { id: 'about', label: 'About & Models', icon: Info },
  ];

  const deviceName = healthData?.hardware?.device_name || 'Detecting...';
  const isColab = healthData?.colab_bridge?.online;

  return (
    <aside
      className={`bg-[#12151D] border-r border-[#1E2330] flex flex-col justify-between shrink-0 select-none transition-all duration-300 ${
        isCollapsed ? 'w-20' : 'w-64'
      }`}
    >
      {/* Top Branding & Toggle */}
      <div>
        <div className={`p-4 border-b border-[#1E2330] flex items-center ${isCollapsed ? 'justify-center' : 'justify-between'}`}>
          <div
            className="flex items-center gap-3 cursor-pointer overflow-hidden"
            onClick={() => setActiveView('dashboard')}
            title="PackVision AI Dashboard"
          >
            <div className="w-9 h-9 shrink-0 rounded-xl bg-gradient-to-tr from-[#8B5CF6] to-[#A78BFA] flex items-center justify-center text-white shadow-lg shadow-[#8B5CF6]/20">
              <Box className="w-5 h-5" />
            </div>
            {!isCollapsed && (
              <div className="truncate">
                <h1 className="font-bold text-base tracking-tight text-white flex items-center gap-1.5 font-mono">
                  PackVision<span className="text-[#8B5CF6]">AI</span>
                </h1>
                <p className="text-[11px] text-[#94A3B8] font-medium tracking-wide truncate">
                  Packaging Design Studio
                </p>
              </div>
            )}
          </div>

          <button
            onClick={() => setIsCollapsed(!isCollapsed)}
            className={`p-1.5 rounded-lg text-[#94A3B8] hover:text-white hover:bg-[#171B25] transition-colors ${
              isCollapsed ? 'mt-3' : ''
            }`}
            title={isCollapsed ? 'Expand Sidebar' : 'Collapse Sidebar'}
          >
            {isCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
          </button>
        </div>

        {/* Navigation Links */}
        <nav className="p-2 space-y-1 mt-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeView === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveView(item.id)}
                title={isCollapsed ? item.label : undefined}
                className={`w-full flex items-center rounded-xl text-sm font-medium transition-all ${
                  isCollapsed ? 'justify-center p-3' : 'gap-3 px-3.5 py-2.5'
                } ${
                  isActive
                    ? 'bg-[#8B5CF6]/15 text-[#A78BFA] border border-[#8B5CF6]/30 shadow-sm'
                    : 'text-[#94A3B8] hover:text-white hover:bg-[#171B25]'
                }`}
              >
                <Icon className={`w-4 h-4 shrink-0 ${isActive ? 'text-[#8B5CF6]' : 'text-[#94A3B8]'}`} />
                {!isCollapsed && <span className="truncate">{item.label}</span>}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Bottom Hardware Diagnostics Pill */}
      <div className="p-3 border-t border-[#1E2330]">
        <div
          className={`rounded-xl bg-[#171B25] border border-[#242A38] text-xs transition-all ${
            isCollapsed ? 'p-2 flex flex-col items-center gap-1' : 'p-3 space-y-2'
          }`}
          title={deviceName}
        >
          <div className={`flex items-center text-[11px] text-[#94A3B8] ${isCollapsed ? 'justify-center' : 'justify-between'}`}>
            {!isCollapsed && (
              <span className="flex items-center gap-1.5 truncate">
                <Cpu className="w-3.5 h-3.5 text-[#8B5CF6] shrink-0" />
                Inference Mode
              </span>
            )}
            <span
              className={`w-2.5 h-2.5 rounded-full shrink-0 ${isColab ? 'bg-cyan-400' : 'bg-emerald-400'}`}
              title={isColab ? 'Google Colab GPU Active' : 'Local GPU/CPU'}
            />
          </div>
          {!isCollapsed && (
            <>
              <div className="font-mono text-[11px] text-white truncate">
                {isColab ? 'Colab Free GPU' : deviceName}
              </div>
              <div className="flex items-center justify-between text-[10px] text-[#94A3B8]">
                <span>Local Studio</span>
                <span className="text-[#8B5CF6]">v1.0.0</span>
              </div>
            </>
          )}
        </div>
      </div>
    </aside>
  );
}
