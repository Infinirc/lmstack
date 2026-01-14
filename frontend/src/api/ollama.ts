/**
 * Ollama Library API
 */
import { api } from "./client";

export interface OllamaModel {
  name: string;
  description?: string;
  pulls: number;
  tags: string[];
  updated?: string;
  sizes: string[];
  capabilities: string[];
  readme?: string;
}

export interface OllamaTagInfo {
  name: string;
  full_name: string;
  size?: string;
  quantization?: string;
}

export const ollamaApi = {
  listModels: async (options?: {
    search?: string;
    capability?: string;
    limit?: number;
  }): Promise<OllamaModel[]> => {
    const response = await api.get<OllamaModel[]>("/ollama/models", {
      params: options,
    });
    return response.data;
  },

  getModelInfo: async (modelName: string): Promise<OllamaModel> => {
    const response = await api.get<OllamaModel>(
      `/ollama/model/${encodeURIComponent(modelName)}`,
    );
    return response.data;
  },

  getModelTags: async (
    modelName: string,
  ): Promise<{ model: string; tags: OllamaTagInfo[] }> => {
    const response = await api.get<{ model: string; tags: OllamaTagInfo[] }>(
      `/ollama/tags/${encodeURIComponent(modelName)}`,
    );
    return response.data;
  },

  getPopular: async (limit?: number): Promise<OllamaModel[]> => {
    const response = await api.get<OllamaModel[]>("/ollama/popular", {
      params: { limit },
    });
    return response.data;
  },
};
