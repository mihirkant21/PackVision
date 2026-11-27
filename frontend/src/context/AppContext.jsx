import React, { createContext, useContext, useState, useEffect } from 'react';
import { api } from '../services/api';

const AppContext = createContext();

export const AppProvider = ({ children }) => {
  const [activeView, setActiveView] = useState('dashboard');
  const [healthData, setHealthData] = useState(null);
  const [taxonomy, setTaxonomy] = useState(null);
  const [activeProject, setActiveProject] = useState(null);
  const [studioInitialState, setStudioInitialState] = useState(null);
  const [evaluationModalData, setEvaluationModalData] = useState(null);
  const [notification, setNotification] = useState(null);

  // Load health & taxonomy on app startup
  useEffect(() => {
    refreshHealth();
    loadTaxonomy();
  }, []);

  const refreshHealth = async () => {
    try {
      const data = await api.getHealth();
      setHealthData(data);
    } catch (err) {
      console.error('Failed to load service health:', err);
    }
  };

  const loadTaxonomy = async () => {
    try {
      const data = await api.getTaxonomy();
      setTaxonomy(data);
    } catch (err) {
      console.error('Failed to load taxonomy:', err);
    }
  };

  const showNotification = (message, type = 'info') => {
    setNotification({ message, type, id: Date.now() });
    setTimeout(() => {
      setNotification((current) => (current && current.message === message ? null : current));
    }, 4000);
  };

  const openProjectInStudio = (project) => {
    setActiveProject(project);
    setStudioInitialState({
      projectId: project.id,
      packageType: project.package_type || 'bottle',
      material: project.material || 'glass',
      finish: project.finish || 'glossy',
      prompt: project.prompt || '',
      negativePrompt: project.negative_prompt || '',
    });
    setActiveView('studio');
  };

  const loadGenerationInStudio = (gen) => {
    const firstOutput = gen.outputs && gen.outputs[0];
    const outputAsset = firstOutput?.asset || (gen.assets?.find((a) => a.asset_type === 'output')) || null;
    const outputAssetId = firstOutput?.asset_id || outputAsset?.id || gen.output_asset_id || null;
    const outputUrl = outputAsset?.url || (outputAssetId ? `/api/v1/files/${outputAssetId}` : null);

    const inputAsset = gen.input_asset || (gen.assets?.find((a) => a.asset_type === 'sketch')) || null;
    const inputAssetId = gen.input_asset_id || inputAsset?.id || null;
    const inputUrl = inputAsset?.url || (inputAssetId ? `/api/v1/files/${inputAssetId}` : null);

    setStudioInitialState({
      projectId: gen.project_id || null,
      packageType: gen.package_type || 'bottle',
      material: gen.material || 'glass',
      finish: gen.finish || 'glossy',
      viewpoint: gen.viewpoint || 'front',
      background: gen.background || 'studio_clean',
      prompt: gen.prompt || '',
      negativePrompt: gen.negative_prompt || '',
      seed: gen.seed,
      inputAssetId: inputAssetId,
      inputAssetUrl: inputUrl,
      inputAsset: inputAsset,
      logoAsset: gen.logo_asset,
      completedOutputId: outputAssetId,
      completedOutputUrl: outputUrl,
      completedOutput: outputAsset,
      completedGen: gen,
    });
    setActiveView('studio');
  };

  const openEvaluationModal = (sketchAssetId, renderAssetId, prompt = '') => {
    setEvaluationModalData({
      sketchAssetId,
      renderAssetId,
      prompt,
    });
  };

  const closeEvaluationModal = () => {
    setEvaluationModalData(null);
  };

  return (
    <AppContext.Provider
      value={{
        activeView,
        setActiveView,
        healthData,
        refreshHealth,
        taxonomy,
        activeProject,
        setActiveProject,
        studioInitialState,
        setStudioInitialState,
        openProjectInStudio,
        loadGenerationInStudio,
        evaluationModalData,
        openEvaluationModal,
        closeEvaluationModal,
        notification,
        showNotification,
      }}
    >
      {children}
    </AppContext.Provider>
  );
};

export const useApp = () => useContext(AppContext);
