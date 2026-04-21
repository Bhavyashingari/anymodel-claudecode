// Azure OpenAI provider for anymodel
// Translates between Anthropic Messages API and Azure OpenAI Chat Completions API

import { translateRequest, translateResponse, createStreamTranslator } from './openai.mjs';

export default {
  name: 'azure',

  buildRequest(url, payload, apiKey) {
    const endpoint = process.env.AZURE_OPENAI_ENDPOINT;
    if (!endpoint) {
      throw new Error('AZURE_OPENAI_ENDPOINT is not set. Expected format: https://<resource-name>.openai.azure.com');
    }

    const apiVersion = process.env.AZURE_OPENAI_API_VERSION || '2024-02-15-preview';

    // Parse payload to get the deployment name (passed as model inside body)
    let body;
    try {
      body = JSON.parse(payload);
    } catch {
      body = {};
    }
    const deploymentId = body.model || process.env.AZURE_OPENAI_DEPLOYMENT || 'gpt-4o'; // fallback

    const parsedUrl = new URL(endpoint);
    return {
      hostname: parsedUrl.hostname,
      port: parsedUrl.port || (parsedUrl.protocol === 'https:' ? 443 : 80),
      protocol: parsedUrl.protocol,
      path: `/openai/deployments/${deploymentId}/chat/completions?api-version=${apiVersion}`,
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'api-key': apiKey || process.env.AZURE_OPENAI_API_KEY,
        'content-length': Buffer.byteLength(payload),
      },
    };
  },

  // Transform the body before sending (Anthropic → Azure OpenAI)
  transformRequest(body) {
    const openaiBody = translateRequest(body);
    // Azure typically ignores the model in the body, but sending it is fine
    return openaiBody;
  },

  // Transform response back (Azure OpenAI → Anthropic)
  transformResponse(body) {
    return translateResponse(body);
  },

  // Create stream translator (Azure OpenAI SSE → Anthropic SSE)
  createStreamTranslator,

  displayInfo(model) {
    return model ? `(${model} via Azure OpenAI)` : '(Azure OpenAI)';
  },

  detect() {
    return !!(process.env.AZURE_OPENAI_API_KEY && process.env.AZURE_OPENAI_ENDPOINT);
  },
};
