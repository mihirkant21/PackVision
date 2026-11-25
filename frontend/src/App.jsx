import React from 'react';
import { useApp } from './context/AppContext';
import Sidebar from './layouts/Sidebar';
import Dashboard from './pages/Dashboard';
import Studio from './pages/Studio';
import Projects from './pages/Projects';
import HistoryPage from './pages/History';
import SettingsPage from './pages/Settings';
import About from './pages/About';
import EvaluationModal from './components/EvaluationModal';
import { CheckCircle2, AlertTriangle, Info } from 'lucide-react';

export default function App() {
  const { activeView, notification, evaluationModalData, closeEvaluationModal } = useApp();

  const renderActiveView = () => {
    switch (activeView) {
      case 'dashboard':
        return <Dashboard />;
      case 'studio':
        return <Studio />;
      case 'projects':
        return <Projects />;
      case 'history':
        return <HistoryPage />;
      case 'settings':
        return <SettingsPage />;
      case 'about':
        return <About />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <div className="flex h-screen bg-[#0B0D12] text-[#F8FAFC] overflow-hidden select-none">
      {/* Sidebar Navigation */}
      <Sidebar />

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col overflow-y-auto bg-[#0B0D12]">
        {renderActiveView()}
      </main>

      {/* Global Evaluation Modal */}
      {evaluationModalData && (
        <EvaluationModal
          sketchAssetId={evaluationModalData.sketchAssetId}
          renderAssetId={evaluationModalData.renderAssetId}
          prompt={evaluationModalData.prompt}
          onClose={closeEvaluationModal}
        />
      )}

      {/* Global Toast Notification */}
      {notification && (
        <div className="fixed bottom-6 right-6 z-50 flex items-center gap-2.5 px-4 py-3 rounded-xl bg-[#171B25] border border-[#242A38] text-xs font-medium text-white shadow-2xl animate-bounce-subtle">
          {notification.type === 'success' && <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />}
          {notification.type === 'error' && <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />}
          {notification.type === 'info' && <Info className="w-4 h-4 text-[#8B5CF6] shrink-0" />}
          <span>{notification.message}</span>
        </div>
      )}
    </div>
  );
}
