// Vertex AI provider for anymodel
// Translates between Anthropic Messages API and Vertex AI OpenAI-compatible API

import { translateRequest, translateResponse, createStreamTranslator } from './openai.mjs';

export default {
  name: 'vertex',

  buildRequest(url, payload, apiKey) {
    const projectId = process.env.VERTEX_PROJECT_ID;
    if (!projectId) {
      throw new Error('VERTEX_PROJECT_ID is not set.');
    }

    const location = process.env.VERTEX_LOCATION || 'us-central1';
    
    // Auth token can be passed via VERTEX_ACCESS_TOKEN or as the apiKey parameter
    const token = apiKey || process.env.VERTEX_ACCESS_TOKEN;
    if (!token) {
      throw new Error('VERTEX_ACCESS_TOKEN is not set. Get one with: gcloud auth print-access-token');
    }

    return {
      hostname: `${location}-aiplatform.googleapis.com`,
      port: 443,
      protocol: 'https:',
      path: `/v1beta1/projects/${projectId}/locations/${location}/endpoints/openapi/chat/completions`,
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'authorization': `Bearer ${token}`,
        'content-length': Buffer.byteLength(payload),
      },
    };
  },

  // Transform the body before sending (Anthropic -> OpenAI format)
  transformRequest(body) {
    return translateRequest(body);
  },

  // Transform response back (OpenAI format -> Anthropic)
  transformResponse(body) {
    return translateResponse(body);
  },

  // Create stream translator (OpenAI SSE -> Anthropic SSE)
  createStreamTranslator,

  displayInfo(model) {
    return model ? `(${model} via Vertex AI)` : '(Vertex AI)';
  },

  detect() {
    return !!(process.env.VERTEX_PROJECT_ID && process.env.VERTEX_ACCESS_TOKEN);
  },
};
