/**
 * PackVision AI - Modular API Client Service
 */

const API_BASE = '/api/v1';

async function request(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  const headers = options.headers || {};

  if (!(options.body instanceof FormData) && !headers['Content-Type']) {
    headers['Content-Type'] = 'application/json';
  }

  const response = await fetch(url, {
    ...options,
    headers,
  });

  if (!response.ok) {
    let errorDetail = `HTTP ${response.status} ${response.statusText}`;
    try {
      const errJson = await response.json();
      if (errJson.detail) {
        errorDetail = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
      }
    } catch (e) {
      // ignore
    }
    throw new Error(errorDetail);
  }

  return response.json();
}

export const api = {
  // Health & Hardware
  getHealth: () => request('/health'),
  setColabConfig: (url) =>
    request('/health/colab-config', {
      method: 'POST',
      body: JSON.stringify({ url }),
    }),

  // Taxonomy
  getTaxonomy: () => request('/taxonomy'),

  // Samples
  getSamples: () => request('/samples'),

  // Uploads
  uploadAsset: async (file, assetType = 'sketch') => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('asset_type', assetType);
    return request('/uploads', {
      method: 'POST',
      body: formData,
    });
  },

  // Preprocessing
  preprocessSketch: async ({ assetId, file, lowThreshold, highThreshold, dilation = 1, targetSize = 512 }) => {
    const formData = new FormData();
    if (assetId) formData.append('asset_id', assetId);
    if (file) formData.append('file', file);
    if (lowThreshold !== undefined && lowThreshold !== null) formData.append('low_threshold', lowThreshold);
    if (highThreshold !== undefined && highThreshold !== null) formData.append('high_threshold', highThreshold);
    formData.append('dilation', dilation);
    formData.append('target_size', targetSize);

    return request('/preprocess', {
      method: 'POST',
      body: formData,
    });
  },

  // Generations
  createGeneration: (payload) =>
    request('/generations', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  getGeneration: (id) => request(`/generations/${id}`),

  listGenerations: (params = {}) => {
    const cleanParams = {};
    Object.entries(params || {}).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== '' && v !== 'undefined' && v !== 'null') {
        cleanParams[k] = v;
      }
    });
    const query = new URLSearchParams(cleanParams).toString();
    return request(`/generations${query ? `?${query}` : ''}`);
  },

  cancelGeneration: (id) =>
    request(`/generations/${id}/cancel`, {
      method: 'POST',
    }),

  retryGeneration: (id) =>
    request(`/generations/${id}/retry`, {
      method: 'POST',
    }),

  deleteGeneration: (id) =>
    request(`/generations/${id}`, {
      method: 'DELETE',
    }),

  clearAllGenerations: () =>
    request('/generations', {
      method: 'DELETE',
    }),

  // Projects
  createProject: (payload) =>
    request('/projects', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  listProjects: (params = {}) => {
    const query = new URLSearchParams(params).toString();
    return request(`/projects${query ? `?${query}` : ''}`);
  },

  getProject: (id) => request(`/projects/${id}`),

  updateProject: (id, payload) =>
    request(`/projects/${id}`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),

  deleteProject: (id) =>
    request(`/projects/${id}`, {
      method: 'DELETE',
    }),

  // Logo Compositing
  createComposite: (payload) =>
    request('/composite', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  // Evaluation
  evaluateRender: (payload) =>
    request('/evaluate', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
};
