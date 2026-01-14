/**
 * HuggingFace API
 */
import { api } from "./client";

export interface HFModelInfo {
  id: string;
  model_id: string;
  author?: string;
  sha?: string;
  pipeline_tag?: string;
  library_name?: string;
  tags: string[];
  downloads: number;
  likes: number;
  private: boolean;
  gated?: string;
  created_at?: string;
  last_modified?: string;
  size_bytes?: number;
  parameter_count?: string;
  description?: string;
}

export interface VRAMEstimate {
  model_id: string;
  parameter_count?: number;
  estimated_vram_gb: number;
  precision: string;
  breakdown: {
    model_weights: number;
    kv_cache: number;
    activations: number;
    overhead: number;
  };
  compatible: boolean;
  messages: string[];
}

export interface HFModelFile {
  filename: string;
  size: number;
  type: string;
}

export interface HFSearchResult {
  id: string;
  author?: string;
  downloads: number;
  likes: number;
  pipeline_tag?: string;
  tags: string[];
}

export const huggingfaceApi = {
  getModelInfo: async (
    modelId: string,
    token?: string,
  ): Promise<HFModelInfo> => {
    const params: Record<string, string> = {};
    if (token) params.token = token;
    const response = await api.get<HFModelInfo>(
      `/huggingface/model/${encodeURIComponent(modelId)}`,
      { params },
    );
    return response.data;
  },

  estimateVRAM: async (
    modelId: string,
    options?: {
      precision?: string;
      context_length?: number;
      gpu_memory_gb?: number;
      token?: string;
    },
  ): Promise<VRAMEstimate> => {
    const response = await api.get<VRAMEstimate>(
      `/huggingface/estimate-vram/${encodeURIComponent(modelId)}`,
      { params: options },
    );
    return response.data;
  },

  listFiles: async (
    modelId: string,
    token?: string,
  ): Promise<HFModelFile[]> => {
    const params: Record<string, string> = {};
    if (token) params.token = token;
    const response = await api.get<HFModelFile[]>(
      `/huggingface/files/${encodeURIComponent(modelId)}`,
      { params },
    );
    return response.data;
  },

  search: async (
    query: string,
    options?: { limit?: number; filter_task?: string },
  ): Promise<HFSearchResult[]> => {
    const response = await api.get<HFSearchResult[]>("/huggingface/search", {
      params: { query, ...options },
    });
    return response.data;
  },

  getPopular: async (limit?: number): Promise<HFSearchResult[]> => {
    const response = await api.get<HFSearchResult[]>("/huggingface/popular", {
      params: { limit },
    });
    return response.data;
  },

  getReadme: async (
    modelId: string,
    token?: string,
  ): Promise<{ content: string | null; message?: string }> => {
    const params: Record<string, string> = {};
    if (token) params.token = token;
    const response = await api.get<{
      content: string | null;
      message?: string;
    }>(`/huggingface/readme/${encodeURIComponent(modelId)}`, { params });
    return response.data;
  },
};
